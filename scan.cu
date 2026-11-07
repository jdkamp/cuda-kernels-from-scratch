#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "cuda_utils.cuh"
#include "scan_utils.cuh"

constexpr int kBlockDim = 512;
constexpr int kCoarseFactor = 8;                        // elements per thread
constexpr int kSectionSize = kBlockDim * kCoarseFactor; // elements per block

// Reference on CPU. Inclusive scan.
void scan_cpu(const int* input, int* output, int n) {
    output[0] = input[0];
    for (int i = 1; i < n; i++) {
        output[i] = output[i-1] + input[i];
    }
}

// Stage 1: Kogge-Stone scan in one block. In each step every element adds the
// element stride positions to its left, stride doubles each step.
// Requires blockDim = kBlockDim, one block, n <= kBlockDim
__global__ void kogge_stone_scan_kernel(const int* input, int* output, int n) {
    __shared__ int section[kBlockDim];
    int i = threadIdx.x;

    if (i < n) {
        section[i] = input[i];
    } else {
        section[i] = 0;
    }

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();    // previous step's writes are done before anyone reads
        int temp = 0;
        if (i >= stride) {
            temp = section[i] + section[i-stride];
        }
        __syncthreads();    // all reads are done before anyone writes
        if (i >= stride) {
            section[i] = temp;
        }
    }
    if (i < n) {
        output[i] = section[i];
    }
}

float kogge_stone_scan(const int* h_input, int* h_output, int n) {
    size_t bytes = n * sizeof(int);
    int *d_input, *d_output;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, bytes, cudaMemcpyHostToDevice);

    float ms = time_kernel([&] {
        kogge_stone_scan_kernel<<<1, kBlockDim>>>(d_input, d_output, n);
    });

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return ms;
}

// Stage 2: Kogge-Stone scan with double buffering. Each step reads from one
// shared array and writes to the other, then the roles swap. Needs one barrier
// per step instead of two, at the cost of twice the shared memory.
// Requires blockDim = kBlockDim, one block, n <= kBlockDim
__global__ void kogge_stone_double_buffer_scan_kernel(const int* input, int* output, int n) {
    __shared__ int buffer[2][kBlockDim];
    int src = 0;    // Index of the buffer read from.
    int i = threadIdx.x;

    if (i < n) {
        buffer[src][i] = input[i];
    } else {
        buffer[src][i] = 0;
    }

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();    // last step's writes are done, and nobody still reads the buffer we now write
        if (i >= stride) {
            buffer[1-src][i] = buffer[src][i] + buffer[src][i-stride];
        } else {
            buffer[1-src][i] = buffer[src][i];
        }
        src = 1 - src;  // swap roles
    }

    if (i < n) {
        output[i] = buffer[src][i];
    }
}

float kogge_stone_double_buffer_scan(const int* h_input, int* h_output, int n) {
    size_t bytes = n * sizeof(int);
    int *d_input, *d_output;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, bytes, cudaMemcpyHostToDevice);

    float ms = time_kernel([&] {
        kogge_stone_double_buffer_scan_kernel<<<1, kBlockDim>>>(d_input, d_output, n);
    });

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return ms;
}

// Stage 3: block scan with warp shuffles (scan-scan-add). Each warp scans its
// 32 values in registers with __shfl_up_sync, warp 0 scans the warp totals,
// then every warp adds the total of all previous warps. Values stay in
// registers instead of shared memory, only the warp totals go through shared
// memory, and the whole block needs two barriers instead of two per step.
// Requires blockDim = kBlockDim, one block, n <= kBlockDim
__global__ void warp_shuffle_scan_kernel(const int* input, int* output, int n) {
    int i = threadIdx.x;
    int val = (i < n) ? input[i] : 0;
    val = block_scan(val);

    if (i < n) {
        output[i] = val;
    }
}

float warp_shuffle_scan(const int* h_input, int* h_output, int n) {
    size_t bytes = n * sizeof(int);
    int *d_input, *d_output;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, bytes, cudaMemcpyHostToDevice);

    float ms = time_kernel([&] {
        warp_shuffle_scan_kernel<<<1, kBlockDim>>>(d_input, d_output, n);
    });

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return ms;
}

// Stage 4: three-phase scan with thread coarsening. Each thread owns a chunk of
// kCoarseFactor consecutive elements. Every thread scans its chunk
// sequentially. Block_scan over the chunk totals. Every thread
// adds the total of all previous chunks. Loads and stores go through shared
// memory so that global memory accesses stay coalesced.
// Requires blockDim = kBlockDim, one block, n <= kSectionSize
__global__ void coarsened_scan_kernel(const int* input, int* output, int n) {
    __shared__ int section[kSectionSize];
    int t = threadIdx.x;

    // Coalesced load: neighbouring threads read neighbouring addresses.
    for (int k = 0; k < kCoarseFactor; k++) {
        int index = k * blockDim.x + t;
        section[index] = (index < n) ? input[index] : 0;
    }
    __syncthreads();    // chunks contain elements loaded by other threads

    // Sequential scan of this thread's chunk.
    int start = t * kCoarseFactor;
    for (int j = start + 1; j < start + kCoarseFactor; j++) {
        section[j] += section[j - 1];
    }

    // Scan the chunk totals. Each thread reads only its own chunk,
    // so no barrier is needed before this.
    int chunk_total = section[start + kCoarseFactor - 1];
    int offset = block_scan(chunk_total) - chunk_total;    // total of all previous chunks

    // Add the total of all previous chunks.
    for (int j = start; j < start + kCoarseFactor; j++) {
        section[j] += offset;
    }
    __syncthreads();    // the coalesced store reads chunks of other threads

    // Coalesced store.
    for (int k = 0; k < kCoarseFactor; k++) {
        int index = k * blockDim.x + t;
        if (index < n) {
            output[index] = section[index];
        }
    }
}

float coarsened_scan(const int* h_input, int* h_output, int n) {
    size_t bytes = n * sizeof(int);
    int *d_input, *d_output;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, bytes, cudaMemcpyHostToDevice);

    float ms = time_kernel([&] {
        coarsened_scan_kernel<<<1, kBlockDim>>>(d_input, d_output, n);
    });

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return ms;
}

// Stage 5: three-phase scan with register tiling. Same as stage 4, but each
// thread copies its chunk from shared memory into registers once . Shared 
// memory is only used to rearrange the data between the coalesced global 
// accesses and the per-thread chunks.
// Requires blockDim = kBlockDim, one block, n <= kSectionSize
__global__ void register_tiled_scan_kernel(const int* input, int* output, int n) {
    __shared__ int section[kSectionSize];
    int t = threadIdx.x;

    // Coalesced load: neighbouring threads read neighbouring addresses.
    for (int k = 0; k < kCoarseFactor; k++) {
        int index = k * blockDim.x + t;
        section[index] = (index < n) ? input[index] : 0;
    }
    __syncthreads();    // chunks contain elements loaded by other threads

    int start = t * kCoarseFactor;
    int vals[kCoarseFactor];

    // Copy this thread's chunk into registers.
    #pragma unroll
    for (int j = 0; j < kCoarseFactor; j++) {
        vals[j] = section[start + j];
    }

    // Sequential scan of this thread's chunk in registers.
    #pragma unroll
    for (int j = 1; j < kCoarseFactor; j++) {
        vals[j] += vals[j - 1];
    }

    // Scan the chunk totals.
    int chunk_total = vals[kCoarseFactor - 1];
    int offset = block_scan(chunk_total) - chunk_total;    // total of all previous chunks

    // Add the total of all previous chunks.
    #pragma unroll
    for (int j = 0; j < kCoarseFactor; j++) {
        vals[j] += offset;
    }

    // Copy the chunk back to shared memory for the coalesced store.
    #pragma unroll
    for (int j = 0; j < kCoarseFactor; j++) {
        section[start + j] = vals[j];
    }
    __syncthreads();    // the coalesced store reads chunks of other threads

    // Coalesced store.
    for (int k = 0; k < kCoarseFactor; k++) {
        int index = k * blockDim.x + t;
        if (index < n) {
            output[index] = section[index];
        }
    }
}

float register_tiled_scan(const int* h_input, int* h_output, int n) {
    size_t bytes = n * sizeof(int);
    int *d_input, *d_output;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, bytes, cudaMemcpyHostToDevice);

    float ms = time_kernel([&] {
        register_tiled_scan_kernel<<<1, kBlockDim>>>(d_input, d_output, n);
    });

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return ms;
}

int main() {
    // Stages 1-3 handle up to kBlockDim elements. 500 leaves the last threads
    // without an element, so the padding with 0 is tested too.
    int n = 500;
    size_t bytes_moved = 2 * sizeof(int) * n;   // read input, write output
    std::vector<int> h_input(n);
    srand(0);   // reproducibility
    for (int i = 0; i < n; i++) {
        h_input[i] = rand() % 10;
    }

    std::vector<int> expected(n, -1);
    scan_cpu(h_input.data(), expected.data(), n);

    std::vector<int> h_output_kogge_stone(n, -1);
    float ms = kogge_stone_scan(h_input.data(), h_output_kogge_stone.data(), n);
    printf("Kogge-Stone:    %.3f ms  %.2f GB/s\n", ms, bandwidth_gb_per_s(bytes_moved, ms));
    assert(count_mismatches(h_output_kogge_stone, expected) == 0);

    std::vector<int> h_output_double_buffer(n, -1);
    ms = kogge_stone_double_buffer_scan(h_input.data(), h_output_double_buffer.data(), n);
    printf("Double buffer:  %.3f ms  %.2f GB/s\n", ms, bandwidth_gb_per_s(bytes_moved, ms));
    assert(count_mismatches(h_output_double_buffer, expected) == 0);

    std::vector<int> h_output_warp_shuffle(n, -1);
    ms = warp_shuffle_scan(h_input.data(), h_output_warp_shuffle.data(), n);
    printf("Warp shuffle:   %.3f ms  %.2f GB/s\n", ms, bandwidth_gb_per_s(bytes_moved, ms));
    assert(count_mismatches(h_output_warp_shuffle, expected) == 0);

    // From Stage 4: handles up to kSectionSize elements. 4000 is not a multiple of
    // kBlockDim, so the last load round is only partly filled.
    int n_coarse = 4000;
    size_t bytes_moved_coarse = 2 * sizeof(int) * n_coarse;   // read input, write output
    std::vector<int> h_input_coarse(n_coarse);
    for (int i = 0; i < n_coarse; i++) {
        h_input_coarse[i] = rand() % 10;
    }

    std::vector<int> expected_coarse(n_coarse, -1);
    scan_cpu(h_input_coarse.data(), expected_coarse.data(), n_coarse);

    std::vector<int> h_output_coarsened(n_coarse, -1);
    ms = coarsened_scan(h_input_coarse.data(), h_output_coarsened.data(), n_coarse);
    printf("Coarsened:      %.3f ms  %.2f GB/s\n", ms, bandwidth_gb_per_s(bytes_moved_coarse, ms));
    assert(count_mismatches(h_output_coarsened, expected_coarse) == 0);

    std::vector<int> h_output_register_tiled(n_coarse, -1);
    ms = register_tiled_scan(h_input_coarse.data(), h_output_register_tiled.data(), n_coarse);
    printf("Register tiled: %.3f ms  %.2f GB/s\n", ms, bandwidth_gb_per_s(bytes_moved_coarse, ms));
    assert(count_mismatches(h_output_register_tiled, expected_coarse) == 0);

    return 0;
}