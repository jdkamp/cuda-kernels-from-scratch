#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda/std/utility>
#include "cuda_utils.cuh"
#include "scan_utils.cuh"
#include "merge.cuh"

constexpr int kThreadsPerBlock = 256;
constexpr int kTileSize = 1024;             // elements sorted by one block
constexpr int kTileThreads = kTileSize / 2; // one thread per pair
constexpr int kElementsPerThread = 8;
constexpr int kElementsPerBlock = kThreadsPerBlock * kElementsPerThread;
constexpr int kRadixBuckets = 2;            // one bit per pass: bucket 0 and bucket 1

// Stage 1: odd-even transposition sort with one kernel launch per phase.
// In the even phase thread t compares pair(2t, 2t+1), in the odd phase
// (2t+1, 2t+2), and swaps if they are out of order. The kernel boundary
// synchronizes the phases across blocks. changed is set if any pair was swapped.
__global__ void odd_even_sort_kernel(int* data, bool* changed, int n, bool is_odd_step) {
    int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x) + (is_odd_step ? 1 : 0);

    if (i < n - 1) {
        if (data[i] > data[i+1]){
            cuda::std::swap(data[i], data[i+1]);
            *changed = true;
        }
    }
}

float odd_even_sort(int* h_data, int n) {
    size_t bytes = n * sizeof(int);
    int num_pairs = n / 2;      // one thread per pair
    int num_blocks = (num_pairs + kThreadsPerBlock - 1) / kThreadsPerBlock;
    int *d_input, *d_data;
    bool *d_changed;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_changed, sizeof(bool));

    cudaMemcpy(d_input, h_data, bytes, cudaMemcpyHostToDevice);

    float ms = time_kernel([&] {
        // Every run sorts the original input, not the resultof the previous run.
        cudaMemcpy(d_data, d_input, bytes, cudaMemcpyDeviceToDevice);

        bool h_changed = true;
        while (h_changed) {
            cudaMemset(d_changed, 0, sizeof(bool));
            odd_even_sort_kernel<<<num_blocks, kThreadsPerBlock>>>(d_data, d_changed, n, false);    // even phase
            odd_even_sort_kernel<<<num_blocks, kThreadsPerBlock>>>(d_data, d_changed, n, true);     // odd phase
            cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
        }
    });

    cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_data);
    cudaFree(d_changed);

    return ms;
}

// Stage 2a: merge sort, step 1. Each block sorts one tile of kTileSize elements
// in shared memory with odd-even transposition sort, one barrier per phase instead
// of one kernel launch per phase. Positions beyond n are padded with INT_MAX, so
// they sort to the end of the tile and are not written back.
// Requires blockDim.x = kTileThreads.
__global__ void sort_tile_kernel(int* data, int n) {
    __shared__ int tile[kTileSize];
    int t = threadIdx.x;
    int tile_start = blockIdx.x * kTileSize;

    // Coalsced load: each thread loads 2 elements, t and t + blockDim.x.
    // Positions beyond n get INT_MAX, so they sort to the end of the tile.
    for (int k = t; k < kTileSize; k += blockDim.x) {
        int index = tile_start + k;
        tile[k] = (index < n) ? data[index] : INT_MAX;
    }

    // kTileSize phases of odd-even
    for (int phase = 0; phase < kTileSize; phase++) {
        __syncthreads();    // previous phase is done (on phase 0: the load is done)
        int i = 2 * t + (phase % 2);
        if (i + 1 < kTileSize && tile[i] > tile[i+1]) {
            cuda::std::swap(tile[i], tile[i+1]);
        }
    }

    __syncthreads();    // sorting is done before the store reads the tile

    // Coalesced store, only positions < n
    for (int k = t; k < kTileSize; k += blockDim.x) {
        int index = tile_start + k;
        if (index < n) {
            data[index] = tile[k];
        }
    }
}

// Stage 2b: merge sort, one merge pass. src consists of sorted runs of length
// width, every pair of neighbouring runs is merged into one sorted run of length
// 2 * width in dst. Each thread computes kElementsPerThread output elements: it
// finds its pair, then its parts of the two runs with co_rank, and merges them
// sequentially. Requires 2 * width to be a multiple of kElementsPerThread.
__global__ void merge_pass_kernel(const int* src, int* dst, int n, int width) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;    // global thread index
    int k_curr = min(tid * kElementsPerThread, n);
    int k_next = min((tid + 1) * kElementsPerThread, n);
    if (k_curr == k_next) {
        return; // nothing to do for thread beyond n
    }

    // Which pair of runs does this threads's output belong to?
    int pair_start = (k_curr / (2 * width)) * (2 * width);

    // The two runs of the pair, at the end of the array they can be shorter or empty
    const int* a = src + pair_start;
    int len_a = min(width, n - pair_start);
    const int* b = a + len_a;
    int len_b = min(width, n - pair_start - len_a);

    // My part of the pair's output, relative to pair_start
    int k_local_curr = k_curr - pair_start;
    int k_local_next = k_next - pair_start;

    // co_rank and merge sequential
    int i_curr = co_rank(k_local_curr, a, len_a, b, len_b);
    int i_next = co_rank(k_local_next, a, len_a, b, len_b);
    int j_curr = k_local_curr - i_curr;
    int j_next = k_local_next - i_next;

    merge_sequential(a + i_curr, i_next - i_curr,
                    b + j_curr, j_next - j_curr,
                    dst + k_curr);

}

// Sorts h_data in place with merge sort and returns the time in ms.
float merge_sort(int* h_data, int n) {
    size_t bytes = n * sizeof(int);
    int num_tiles = (n + kTileSize - 1) / kTileSize;
    int num_merge_blocks = (n + kElementsPerBlock - 1) / kElementsPerBlock;
    int *d_input, *d_buffer_0, *d_buffer_1;
    int* d_result = nullptr;    // the buffer that holds the sorted array after the last pass

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_buffer_0, bytes);
    cudaMalloc(&d_buffer_1, bytes);

    cudaMemcpy(d_input, h_data, bytes, cudaMemcpyHostToDevice);

    float ms = time_kernel([&] {
        // Every run sorts the original input, not the result of the previous run.
        int* src = d_buffer_0;
        int* dst = d_buffer_1;
        cudaMemcpy(src, d_input, bytes, cudaMemcpyDeviceToDevice);

        // Sort every tile of kTileSize elements on its own
        sort_tile_kernel<<<num_tiles, kTileThreads>>>(src, n);

        // Merge neighbouring runs, the run length doubles every pass
        for (int width = kTileSize; width < n; width *= 2) {
            merge_pass_kernel<<<num_merge_blocks, kThreadsPerBlock>>>(src, dst, n, width);
            std::swap(src, dst);    // the output of this pass is the input of the next one
        }
        d_result = src;
    });

    cudaMemcpy(h_data, d_result, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_buffer_0);
    cudaFree(d_buffer_1);

    return ms;
}

// Stage 3: radix sort in a single block, one bit per pass. The host runs 32
// passes. Each pass splits the keys stably by the current bit: keys with bit 0
// first, then keys with bit 1, each group in its previous order. A block_scan
// over the bits gives every key the number of ones before it, which yields its
// new position. Because every pass is stable, the order from the lower bits is
// kept, and after 32 passes the keys are sorted.
// Requires n <= blockDim.x, one block.
__global__ void radix_pass_single_block_kernel(const unsigned* src, unsigned* dst, int n, int bit) {
    __shared__ int total_ones;

    int i = threadIdx.x;
    unsigned key = (i < n) ? src[i] : 0;
    int b = (key >> bit) & 1;

    // Ones before me (exclusive) and up to me (inclusive).
    int ones_inclusive = block_scan(b);
    int ones_before = ones_inclusive - b;

    // The last thread knows how many ones the block has in total.
    if (threadIdx.x == blockDim.x - 1) {
        total_ones = ones_inclusive;
    }
    __syncthreads();    // total_ones is written by the last thread, read by all

    if (i < n) {
        int total_zeros = n - total_ones;
        int zeros_before = i - ones_before;
        int pos = (b == 0) ? zeros_before : total_zeros + ones_before;
        dst[pos] = key;
    }
}

float radix_sort_single_block(unsigned* h_data, int n) {
    size_t bytes = n * sizeof(unsigned);
    unsigned *d_input, *d_buffer_0, *d_buffer_1;
    unsigned *d_result = nullptr;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_buffer_0, bytes);
    cudaMalloc(&d_buffer_1, bytes);

    cudaMemcpy(d_input, h_data, bytes, cudaMemcpyHostToDevice);

    float ms = time_kernel([&] {
        unsigned* src = d_buffer_0;
        unsigned* dst = d_buffer_1;
        cudaMemcpy(src, d_input, bytes, cudaMemcpyDeviceToDevice);
        for (int bit = 0; bit < 32; bit++) {
            radix_pass_single_block_kernel<<<1, kThreadsPerBlock>>>(src, dst, n, bit);
            std::swap(src, dst);
        }
        d_result = src;
    });

    cudaMemcpy(h_data, d_result, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_buffer_0);
    cudaFree(d_buffer_1);

    return ms;
}

// Stage 4: radix sort over many blocks, one bit per pass. With several blocks,
// a key's new position also depends on the keys in all earlier blocks. Each of
// the 32 passes therefore runs three kernels: (a) every block counts its keys
// per bucket (bit 0 and bit 1) into a table, (b) an exclusive scan over the
// table turns the counts into the start of each block's part of each bucket,
// and (c) every block writes its keys there. The table is stored bucket-major,
// so all zeros of all blocks come before all ones and the pass stays stable.

// Stage 4a: count how many keys of each block fall into each bucket and store
// the counts bucket-major: table[bucket * num_blocks + block].
__global__ void radix_count_kernel(const unsigned* src, int n, int bit, int* table) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks = gridDim.x;

    unsigned key = (i < n) ? src[i] : 0;
    int b = (key >> bit) & 1;

    int ones = __syncthreads_count(b);      // number of threads in the block with b != 0

    if (threadIdx.x == 0) {
        int block_size = min(blockDim.x, n - blockIdx.x * blockDim.x);  // keys in this block, the last block can be shorter
        table[0 * num_blocks + blockIdx.x] = block_size - ones;         // zeros
        table[1 * num_blocks + blockIdx.x] = ones;                      // ones
    }
}

// Stage 4b: exclusive scan over the whole table in place, with a single block
// that walks over it in chunks of blockDim.x and carries the running total.
__global__ void radix_scan_kernel(int* table, int size) {
    __shared__ int carry;
    if (threadIdx.x == 0) {
        carry = 0;
    }
    __syncthreads();

    for (int start = 0; start < size; start += blockDim.x) {
        int idx = start + threadIdx.x;
        int value = (idx < size) ? table[idx] : 0;
        int inclusive = block_scan(value);

        if (idx < size) {
            table[idx] = carry + inclusive - value;     // exclusive: carry + everything before me in this chunk
        }
        __syncthreads();    // everyone has read carry before the last thread changes it

        if (threadIdx.x == blockDim.x - 1) {
            carry += inclusive;     // the chunk total: inclusive of the last thread
        }
        __syncthreads();    // the new carry is visible before the next chunk reads it
    }
}

// Stage 4c: each block scans its bits again to find every key's position within
// its bucket, adds the bucket's start for this block from the scanned table, and
// writes the key there. Keys keep their order within a bucket, so the pass is stable.
__global__ void radix_scatter_kernel(const unsigned* src, unsigned* dst, int n, int bit, const int* table) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks = gridDim.x;

    unsigned key = (i < n) ? src[i] : 0;
    int b = (key >> bit) & 1;

    // Position within my bucket in this block: how many keys before me in the
    // block fall into the same bucket?
    int ones_inclusive = block_scan(b);
    int ones_before = ones_inclusive - b;
    int zeros_before = threadIdx.x - ones_before;   // keys before me in the block minus the ones among them

    if (i < n) {
        int bucket_start = table[b * num_blocks + blockIdx.x];          // start of my bucket for this block
        int pos = bucket_start + (b == 0 ? zeros_before : ones_before); // plus my position within it
        dst[pos] = key;
    }
}

float radix_sort(unsigned* h_data, int n) {
    int num_blocks = (n + kThreadsPerBlock - 1) / kThreadsPerBlock;
    int table_size = kRadixBuckets * num_blocks;
    size_t bytes = n * sizeof(unsigned);
    size_t bytes_table = table_size * sizeof(int);
    unsigned *d_input, *d_buffer_0, *d_buffer_1;
    unsigned *d_result = nullptr;
    int *d_table;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_buffer_0, bytes);
    cudaMalloc(&d_buffer_1, bytes);
    cudaMalloc(&d_table, bytes_table);

    cudaMemcpy(d_input, h_data, bytes, cudaMemcpyHostToDevice);

    float ms = time_kernel([&] {
        unsigned* src = d_buffer_0;
        unsigned* dst = d_buffer_1;
        cudaMemcpy(src, d_input, bytes, cudaMemcpyDeviceToDevice);

        for (int bit = 0; bit < 32; bit++) {
            radix_count_kernel<<<num_blocks, kThreadsPerBlock>>>(src, n, bit, d_table);
            radix_scan_kernel<<<1, kThreadsPerBlock>>>(d_table, table_size);
            radix_scatter_kernel<<<num_blocks, kThreadsPerBlock>>>(src, dst, n, bit, d_table);
            std::swap(src, dst);    // the output of this pass is the input of the next one
        }
        d_result = src;
    });

    cudaMemcpy(h_data, d_result, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_buffer_0);
    cudaFree(d_buffer_1);
    cudaFree(d_table);

    return ms;
}

int main() {
    int n_small = 999;          // odd-even: O(n^2), only for small inputs
    int n_radix_small = 200;    // single-block radix sort: n <= kThreadsPerBlock
    int n = (1 << 20) + 3;      // merge sort and multi-block radix sort: not a multiple of
                                // kTileSize or kThreadsPerBlock, several merge passes

    srand(0);   // reproducibility

    // Odd-even sort: random values with duplicates. Reference on the CPU.
    std::vector<int> h_input_small(n_small);
    for (int i = 0; i < n_small; i++) {
        h_input_small[i] = rand() % 1000;
    }
    std::vector<int> h_expected_small = h_input_small;
    std::sort(h_expected_small.begin(), h_expected_small.end());

    std::vector<int> h_output = h_input_small;
    float ms = odd_even_sort(h_output.data(), n_small);
    printf("odd-even:   %.3f ms\n", ms);
    assert(count_mismatches(h_output, h_expected_small) == 0);

    // Radix sort in a single block: unsigned keys from rand().
    // n_radix_small < kThreadsPerBlock, so some threads have no key.
    std::vector<unsigned> h_input_radix(n_radix_small);
    for (int i = 0; i < n_radix_small; i++) {
        h_input_radix[i] = rand();
    }
    std::vector<unsigned> h_expected_radix = h_input_radix;
    std::sort(h_expected_radix.begin(), h_expected_radix.end());

    std::vector<unsigned> h_output_radix = h_input_radix;
    ms = radix_sort_single_block(h_output_radix.data(), n_radix_small);
    printf("radix:      %.3f ms\n", ms);
    assert(count_mismatches(h_output_radix, h_expected_radix) == 0);

    // Merge sort and multi-block radix sort get the same n keys, so their
    // throughput can be compared directly. Reference on the CPU.
    std::vector<unsigned> h_input(n);
    for (int i = 0; i < n; i++) {
        h_input[i] = rand();
    }
    std::vector<unsigned> h_expected = h_input;
    std::sort(h_expected.begin(), h_expected.end());

    // Merge sort: sort tiles in shared memory, then merge passes. It works on int;
    // rand() never exceeds INT_MAX, so the same keys fit and sort the same way.
    std::vector<int> h_output_merge(h_input.begin(), h_input.end());
    std::vector<int> h_expected_merge(h_expected.begin(), h_expected.end());
    ms = merge_sort(h_output_merge.data(), n);
    printf("merge sort: %.3f ms  %.1f Mkeys/s\n", ms, n / (ms * 1e-3) / 1e6);
    assert(count_mismatches(h_output_merge, h_expected_merge) == 0);

    // Radix sort over many blocks
    std::vector<unsigned> h_output_radix_large = h_input;
    ms = radix_sort(h_output_radix_large.data(), n);
    printf("radix (multi-block): %.3f ms  %.1f Mkeys/s\n", ms, n / (ms * 1e-3) / 1e6);
    assert(count_mismatches(h_output_radix_large, h_expected) == 0);

    return 0;
}