#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "cuda_utils.cuh"
#include "scan_utils.cuh"

constexpr int kBlockDim = 512;
constexpr int kCoarseFactor = 8;                                // elements per thread (stage 4)
constexpr int kElementsPerBlock = kBlockDim * kCoarseFactor;

// Predicate: an element is kept if this returns true. Shared by the CPU
// reference and the kernels, so both always filter the same way.
__host__ __device__ bool is_kept(int x) {
    return x % 2 == 0;
}

// Reference on CPU. Writes the kept elements to output in their original
// order and returns how many there are.
int filter_cpu(const int* input, int* output, int n) {
    int count = 0;
    for (int i = 0;  i < n; i++) {
        if (is_kept(input[i])) {
            output[count] = input[i];
            count++;
        }
    }
    return count;
}

// Stage 1: basic unstable filter. Every thread with a kept element gets its
// output slot with one atomicAdd on a global counter, so all kept elements
// compete for the same address and the order of the output is not preserved.
__global__ void filter_basic_kernel(const int* input, int* output, int* count, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && is_kept(input[i])) {
        int pos = atomicAdd(count, 1);
        output[pos] = input[i];
    }
}

float filter_basic(const int* h_input, int* h_output, int* h_count, int n) {
    size_t bytes = n * sizeof(int);
    int num_blocks = (n + kBlockDim - 1) / kBlockDim;
    int *d_input, *d_output, *d_count;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMalloc(&d_count, sizeof(int));

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, bytes, cudaMemcpyHostToDevice);

    float ms = time_kernel([&] {
        cudaMemset(d_count, 0, sizeof(int));    // the kernel counts with atomicAdd
        filter_basic_kernel<<<num_blocks, kBlockDim>>>(d_input, d_output, d_count, n);
    });

    cudaMemcpy(h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_count);

    return ms;
}

// Stage 2: unstable filter with warp-aggregated atomics. The kept lanes of a warp
// find each other with __activemask, the lowest one reserves slots for all
// of them with a single atomicAdd, and every lane computes its own slot from the
// number of kept lanes below it.
__global__ void filter_warp_aggregated_kernel(const int* input, int* output, int* count, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && is_kept(input[i])) {
        unsigned int active = __activemask();   // lanes of this warp that keep an element
        int lane = threadIdx.x % kWarpSize;
        int leader = __ffs(active) - 1;         // lowest lane in active
        int base = 0;
        if (lane == leader) {
            base = atomicAdd(count, __popc(active));    // reserve one slot per active lane
        }
        base = __shfl_sync(active, base, leader);       // broadcast the first slot to all active lanes
        int offset = __popc(active & ((1u << lane) - 1));   // active lanes below this one
        output[base+offset] = input[i];
    }
}

float filter_warp_aggregated(const int* h_input, int* h_output, int* h_count, int n) {
    size_t bytes = n * sizeof(int);
    int num_blocks = (n + kBlockDim - 1) / kBlockDim;
    int *d_input, *d_output, *d_count;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMalloc(&d_count, sizeof(int));

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, bytes, cudaMemcpyHostToDevice);

    float ms = time_kernel([&] {
        cudaMemset(d_count, 0, sizeof(int));    // the kernel counts with atomicAdd
        filter_warp_aggregated_kernel<<<num_blocks, kBlockDim>>>(d_input, d_output, d_count, n);
    });

    cudaMemcpy(h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_count);

    return ms;
}

// Stage 3: unstable filter with block-aggregated atomics. block_scan over the
// kept flags gives every thread its position within the block, and the last
// thread reserves slots for the whole block with a single atomicAdd. One atomic
// per block instead of per warp; the order is kept within a block, but not
// between blocks. Requires blockDim.x to be a multiple of kWarpSize.
__global__ void filter_block_aggregated_kernel(const int* input, int* output, int* count, int n) {
    __shared__ int block_base;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int flag = (i < n && is_kept(input[i])) ? 1 : 0;

    int inclusive = block_scan(flag);
    int offset = inclusive - flag;     // kept elements before me in the block

    if (threadIdx.x == blockDim.x - 1) {
        block_base = atomicAdd(count, inclusive);   // inclusive of the last thread = block total
    }

    __syncthreads();    // block_base is written by the last thread, read by all

    if (flag) {
        output[block_base+offset] = input[i];
    }
}

float filter_block_aggregated(const int* h_input, int* h_output, int* h_count, int n) {
    size_t bytes = n * sizeof(int);
    int num_blocks = (n + kBlockDim - 1) / kBlockDim;
    int *d_input, *d_output, *d_count;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMalloc(&d_count, sizeof(int));

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, bytes, cudaMemcpyHostToDevice);

    float ms = time_kernel([&] {
        cudaMemset(d_count, 0, sizeof(int));    // the kernel counts with atomicAdd
        filter_block_aggregated_kernel<<<num_blocks, kBlockDim>>>(d_input, d_output, d_count, n);
    });

    cudaMemcpy(h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_count);

    return ms;
}

// Stage 4: stage 3 with thread coarsening. Each thread loads kCoarseFactor
// elements into registers (interleaved, so loads stay coalesced) and counts its
// kept ones. block_scan over these counts gives every thread its offset in the
// block, the last thread reserves slots for the whole block with one atomicAdd,
// and every thread writes its kept elements one after another. The barriers and
// the atomic are now shared by kCoarseFactor times more elements.
// Requires blockDim.x = kBlockDim.
__global__ void filter_coarsened_kernel(const int* input, int* output, int* count, int n) {
    __shared__ int block_base;
    int block_start = blockIdx.x * kElementsPerBlock;

    int vals[kCoarseFactor];
    int thread_count = 0;

    // Interleaved load: in round k, neighbouring threads read neighbouring elements.
    #pragma unroll
    for (int k = 0; k < kCoarseFactor; k++) {
        int index = block_start + k * blockDim.x + threadIdx.x;
        vals[k] = (index < n) ? input[index] : 0;
        if (index < n && is_kept(vals[k])) {
            thread_count++;
        }
    }

    int inclusive = block_scan(thread_count);
    int offset = inclusive - thread_count;     // kept elements of the threads before me

    if (threadIdx.x == blockDim.x - 1) {
        block_base = atomicAdd(count, inclusive);   // inclusive of the last thread = block total
    }

    __syncthreads();    // block_base is written by the last thread, read by all

    int pos = block_base + offset;
    #pragma unroll
    for (int k = 0; k < kCoarseFactor; k++) {
        int index = block_start + k * blockDim.x + threadIdx.x;
        if (index < n && is_kept(vals[k])) {
            output[pos] = vals[k];
            pos++;
        }
    }
}

float filter_coarsened(const int* h_input, int* h_output, int* h_count, int n) {
    size_t bytes = n * sizeof(int);
    int num_blocks = (n + kElementsPerBlock - 1) / kElementsPerBlock;   // each block handles kElementsPerBlock
    int *d_input, *d_output, *d_count;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMalloc(&d_count, sizeof(int));

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, bytes, cudaMemcpyHostToDevice);

    float ms = time_kernel([&] {
        cudaMemset(d_count, 0, sizeof(int));    // the kernel counts with atomicAdd
        filter_coarsened_kernel<<<num_blocks, kBlockDim>>>(d_input, d_output, d_count, n);
    });

    cudaMemcpy(h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_count);

    return ms;
}

// Stage 5: stage 4 with privatization. Each thread first writes its kept
// elements into a shared-memory buffer of the block, packed without gaps.
// After a barrier the whole block copies the buffer to the output, so
// neighbouring threads write neighbouring addresses and the global stores
// are coalesced. Requires blockDim.x = kBlockDim.
__global__ void filter_privatization_kernel(const int* input, int* output, int* count, int n) {
    __shared__ int block_base;
    __shared__ int s_buffer[kElementsPerBlock]; // kept elements of the block
    __shared__ int block_total;                 // number of kept elements in the block
    int block_start = blockIdx.x * kElementsPerBlock;

    int vals[kCoarseFactor];
    int thread_count = 0;

    // Interleaved load: in round k, neighbouring threads read neighbouring elements.
    #pragma unroll
    for (int k = 0; k < kCoarseFactor; k++) {
        int index = block_start + k * blockDim.x + threadIdx.x;
        vals[k] = (index < n) ? input[index] : 0;
        if (index < n && is_kept(vals[k])) {
            thread_count++;
        }
    }

    int inclusive = block_scan(thread_count);
    int offset = inclusive - thread_count;     // kept elements of the threads before me

    if (threadIdx.x == blockDim.x - 1) {
        block_base = atomicAdd(count, inclusive);   // inclusive of the last thread = block total
        block_total = inclusive;
    }

    __syncthreads();    // block_base and block_total are written by the last thread, read by all

    int pos = offset;
    #pragma unroll
    for (int k = 0; k < kCoarseFactor; k++) {
        int index = block_start + k * blockDim.x + threadIdx.x;
        if (index < n && is_kept(vals[k])) {
            s_buffer[pos] = vals[k];
            pos++;
        }
    }
    __syncthreads();    // the copy reads entries written by other threads

    #pragma unroll
    for (int k = 0; k < kCoarseFactor; k++) {
        int j = k * blockDim.x + threadIdx.x;     // interleaved, like the load
        if (j < block_total) {
            output[block_base+j] = s_buffer[j];
        }
    }
}

float filter_privatization(const int* h_input, int* h_output, int* h_count, int n) {
    size_t bytes = n * sizeof(int);
    int num_blocks = (n + kElementsPerBlock - 1) / kElementsPerBlock;   // each block handles kElementsPerBlock
    int *d_input, *d_output, *d_count;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMalloc(&d_count, sizeof(int));

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, bytes, cudaMemcpyHostToDevice);

    float ms = time_kernel([&] {
        cudaMemset(d_count, 0, sizeof(int));    // the kernel counts with atomicAdd
        filter_privatization_kernel<<<num_blocks, kBlockDim>>>(d_input, d_output, d_count, n);
    });

    cudaMemcpy(h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_count);

    return ms;
}

int main() {
    int n = (1 << 24) + 3;
    std::vector<int> h_input(n);
    srand(0);   // reproducibility
    for (int i = 0; i < n; i++) {
        h_input[i] = rand() % 10;
    }

    std::vector<int> expected(n, -1);
    int expected_count = filter_cpu(h_input.data(), expected.data(), n);
    expected.resize(expected_count);                            // drop unused -1 entries
    size_t bytes_moved = (n + expected_count) * sizeof(int);   // read all, write the kept ones
    std::vector<int> expected_sorted = expected;
    std::sort(expected_sorted.begin(), expected_sorted.end());  // unstable filters are compared sorted

    // Basic: one atomicAdd per kept element.
    std::vector<int> h_output_basic(n, -1);
    int count_basic = 0;
    float ms = filter_basic(h_input.data(), h_output_basic.data(), &count_basic, n);
    printf("basic:       %.3f ms  %5.1f GB/s  kept %d\n", ms, bandwidth_gb_per_s(bytes_moved, ms), count_basic);
    assert(count_basic == expected_count);
    h_output_basic.resize(count_basic);
    std::sort(h_output_basic.begin(), h_output_basic.end());
    assert(count_mismatches(h_output_basic, expected_sorted) == 0);

    // Warp-aggregated: one atomicAdd per warp
    std::vector<int> h_output_warp_aggregated(n, -1);
    int count_warp_aggregated = 0;
    ms = filter_warp_aggregated(h_input.data(), h_output_warp_aggregated.data(), &count_warp_aggregated, n);
    printf("warp aggr.:  %.3f ms  %5.1f GB/s  kept %d\n", ms, bandwidth_gb_per_s(bytes_moved, ms), count_warp_aggregated);
    assert(count_warp_aggregated == expected_count);
    h_output_warp_aggregated.resize(count_warp_aggregated);
    std::sort(h_output_warp_aggregated.begin(), h_output_warp_aggregated.end());
    assert(count_mismatches(h_output_warp_aggregated, expected_sorted) == 0);

    // Block-aggregated: one atomicAdd per block
    std::vector<int> h_output_block_aggregated(n, -1);
    int count_block_aggregated = 0;
    ms = filter_block_aggregated(h_input.data(), h_output_block_aggregated.data(), &count_block_aggregated, n);
    printf("block aggr.: %.3f ms  %5.1f GB/s  kept %d\n", ms, bandwidth_gb_per_s(bytes_moved, ms), count_block_aggregated);
    assert(count_block_aggregated == expected_count);
    h_output_block_aggregated.resize(count_block_aggregated);
    std::sort(h_output_block_aggregated.begin(), h_output_block_aggregated.end());
    assert(count_mismatches(h_output_block_aggregated, expected_sorted) == 0);

    // Coarsened: one atomicAdd per kElementsPerBlock elements
    std::vector<int> h_output_coarsened(n, -1);
    int count_coarsened = 0;
    ms = filter_coarsened(h_input.data(), h_output_coarsened.data(), &count_coarsened, n);
    printf("coarsened:   %.3f ms  %5.1f GB/s  kept %d\n", ms, bandwidth_gb_per_s(bytes_moved, ms), count_coarsened);
    assert(count_coarsened == expected_count);
    h_output_coarsened.resize(count_coarsened);
    std::sort(h_output_coarsened.begin(), h_output_coarsened.end());
    assert(count_mismatches(h_output_coarsened, expected_sorted) == 0);

    // Privatized: packed in shared memory first, then coalesced stores
    std::vector<int> h_output_privatization(n, -1);
    int count_privatization = 0;
    ms = filter_privatization(h_input.data(), h_output_privatization.data(), &count_privatization, n);
    printf("privatized:  %.3f ms  %5.1f GB/s  kept %d\n", ms, bandwidth_gb_per_s(bytes_moved, ms), count_privatization);
    assert(count_privatization == expected_count);
    h_output_privatization.resize(count_privatization);
    std::sort(h_output_privatization.begin(), h_output_privatization.end());
    assert(count_mismatches(h_output_privatization, expected_sorted) == 0);

    return 0;
}