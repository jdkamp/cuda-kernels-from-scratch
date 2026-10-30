#include <cassert>
#include <cstdio>
#include <vector>
#include "cuda_utils.cuh"

constexpr int kBlockDim = 1024;
constexpr int kCoarseFactor = 2;
constexpr int kWarpSize = 32;


// Stage 1: Reduction tree with global memory, one block.
__global__ void simple_reduction_kernel(float* input, float* output) {
    int i = 2 * threadIdx.x;
    for (int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 0) {
            input[i] += input[i + stride];
        }
        // Wait for the whole block: the next step reads sums written in this step.
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *output = input[0];
    }
}

// Copies to GPU and runs kernel
void simple_reduction(float* h_input, float* h_output, int n) {
    size_t size = sizeof(float);
    float *d_input, *d_output;

    cudaMalloc(&d_input, n * size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, h_input, n * size, cudaMemcpyHostToDevice);

    simple_reduction_kernel<<<1, n / 2>>>(d_input, d_output);

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}


// Stage 2: Reduction tree with global memory, one block. Active threads are contiguous
// to reduce control divergence.
__global__ void convergent_reduction_kernel(float* input, float* output) {
    int i = threadIdx.x;
    for (int stride = blockDim.x; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            input[i] += input[i + stride];
        }
        // Wait for the whole block: the next step reads sums written in this step.
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *output = input[0];
    }
}

// Copies to GPU and runs kernel
void convergent_reduction(float* h_input, float* h_output, int n) {
    size_t size = sizeof(float);
    float *d_input, *d_output;

    cudaMalloc(&d_input, n * size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, h_input, n * size, cudaMemcpyHostToDevice);

    convergent_reduction_kernel<<<1, n / 2>>>(d_input, d_output);

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}


// Stage 3: First step reads global memory, all later steps run in shared memory.
__global__ void shared_mem_reduction_kernel(float* input, float* output) {
    __shared__ float input_s[kBlockDim];
    int i = threadIdx.x;
    input_s[i] = input[i] + input[i + kBlockDim];
    for (int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();    // wait until the whole block has written the previous (or initial) step
        if (threadIdx.x < stride) {
            input_s[i] += input_s[i + stride];
        }
    }

    if (threadIdx.x == 0) {
        *output = input_s[0];
    }
}

// Copies to GPU and runs kernel
void shared_mem_reduction(float* h_input, float* h_output, int n) {
    size_t size = sizeof(float);
    float *d_input, *d_output;

    cudaMalloc(&d_input, n * size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, h_input, n * size, cudaMemcpyHostToDevice);

    shared_mem_reduction_kernel<<<1, n / 2>>>(d_input, d_output);

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}


// Stage 4: Many blocks. Each block reduces its own segment of 2 * kBlockDim
// elements and adds its partial sum with atomicAdd.
__global__ void segmented_reduction_kernel(float* input, float* output) {
    __shared__ float input_s[kBlockDim];
    int segment = 2 * blockDim.x * blockIdx.x;
    int i = segment + threadIdx.x;
    int t = threadIdx.x;
    input_s[t] = input[i] + input[i + kBlockDim];
    for (int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();    // wait until the whole block has written the previous (or initial) step
        if (threadIdx.x < stride) {
            input_s[t] += input_s[t + stride];
        }
    }

    if (threadIdx.x == 0) {
        atomicAdd(output, input_s[0]);
    }
}

// Copies to GPU and runs kernel
float segmented_reduction(float* h_input, float* h_output, int n) {
    size_t size = sizeof(float);
    float *d_input, *d_output;

    cudaMalloc(&d_input, n * size);
    cudaMalloc(&d_output, size);
    cudaMemset(d_output, 0, size);

    cudaMemcpy(d_input, h_input, n * size, cudaMemcpyHostToDevice);

    GpuTimer timer;
    timer.begin();

    segmented_reduction_kernel<<<n / (2 * kBlockDim), kBlockDim>>>(d_input, d_output);

    float ms = timer.end();

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return ms;
}


// Stage 5: Thread coarsening. Each thread first sums 2 * kCoarseFactor elements
// sequentially, kBlockDim apart so adjacent elements can be read, then the tree
// runs as in stage 4.
__global__ void coarsened_reduction_kernel(float* input, float* output) {
    __shared__ float input_s[kBlockDim];
    int segment = kCoarseFactor * 2 * blockDim.x * blockIdx.x;
    int i = segment + threadIdx.x;
    int t = threadIdx.x;
    float sum = 0.0f;
    for (int k = 0; k < kCoarseFactor * 2; k++) {
        sum += input[i + k * kBlockDim];
    }
    input_s[t] = sum;

    for (int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();    // wait until the whole block has written the previous (or initial) step
        if (threadIdx.x < stride) {
            input_s[t] += input_s[t + stride];
        }
    }

    if (threadIdx.x == 0) {
        atomicAdd(output, input_s[0]);
    }
}

// Copies to GPU and runs kernel
float coarsened_reduction(float* h_input, float* h_output, int n) {
    size_t size = sizeof(float);
    float *d_input, *d_output;

    cudaMalloc(&d_input, n * size);
    cudaMalloc(&d_output, size);
    cudaMemset(d_output, 0, size);

    cudaMemcpy(d_input, h_input, n * size, cudaMemcpyHostToDevice);

    GpuTimer timer;
    timer.begin();

    coarsened_reduction_kernel<<<n / (kCoarseFactor * 2 * kBlockDim), kBlockDim>>>(d_input, d_output);

    float ms = timer.end();

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return ms;
}


// Stage 6: Warp shuffle. Each warp reduces its 32 partial sums in registers,
// lane 0 stores the warp sum in shared memory.
__global__ void warp_shuffle_reduction_kernel(float* input, float* output) {
    __shared__ float warp_sums[kBlockDim/kWarpSize];
    int segment = kCoarseFactor * 2 * blockDim.x * blockIdx.x;
    int i = segment + threadIdx.x;
    int lane = threadIdx.x % kWarpSize;
    int warp = threadIdx.x / kWarpSize;
    float sum = 0.0f;

    for (int k = 0; k < kCoarseFactor * 2; k++) {
        sum += input[i + k * kBlockDim];
    }

    for (int offset = kWarpSize / 2 ; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum , offset);
    }

    if (lane == 0) {
        warp_sums[warp] = sum;
    }
    __syncthreads();    // The only barrier: all 32 warp sums must be in shared memory

    if (warp == 0) {
        sum = warp_sums[lane];
        for (int offset = kWarpSize / 2 ; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum , offset);
        }

        if (lane == 0) {
            atomicAdd(output, sum);
        }

    }
}

// Copies to GPU and runs kernel
float warp_shuffle_reduction(float* h_input, float* h_output, int n) {
    size_t size = sizeof(float);
    float *d_input, *d_output;

    cudaMalloc(&d_input, n * size);
    cudaMalloc(&d_output, size);
    cudaMemset(d_output, 0, size);

    cudaMemcpy(d_input, h_input, n * size, cudaMemcpyHostToDevice);

    GpuTimer timer;
    timer.begin();

    warp_shuffle_reduction_kernel<<<n / (kCoarseFactor * 2 * kBlockDim), kBlockDim>>>(d_input, d_output);

    float ms = timer.end();

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return ms;
}




int main() {
    int n = 2048 * 1024;
    int n_single_block = 2 * kBlockDim;   // stages 1-3 run in one block and need exactly this size
    size_t bytes_moved = n * sizeof(float);   // read input, the single output value is negligible

    std::vector<float> h_input(n);
    for (int i = 0; i < n; i++) {
        h_input[i] = i % 7;
    }

    // Reference results on the CPU.
    float expected_single_block = 0.0f;
    for (int i = 0; i < n_single_block; i++) {
        expected_single_block += h_input[i];
    }
    float expected = 0.0f;
    for (int i = 0; i < n; i++) {
        expected += h_input[i];
    }

    // -1 marks a result that was never copied back.
    float h_output = -1.0f;
    simple_reduction(h_input.data(), &h_output, n_single_block);
    printf("1. Simple tree      (n = %d): %.1f\n", n_single_block, h_output);
    assert(h_output == expected_single_block);

    h_output = -1.0f;
    convergent_reduction(h_input.data(), &h_output, n_single_block);
    printf("2. Convergent tree  (n = %d): %.1f\n", n_single_block, h_output);
    assert(h_output == expected_single_block);

    h_output = -1.0f;
    shared_mem_reduction(h_input.data(), &h_output, n_single_block);
    printf("3. Shared memory    (n = %d): %.1f\n", n_single_block, h_output);
    assert(h_output == expected_single_block);

    h_output = -1.0f;
    float ms = segmented_reduction(h_input.data(), &h_output, n);
    printf("4. Segmented        (n = %d): %.1f   (%.3f ms, %.1f GB/s)\n", n, h_output, ms, bandwidth_gb_per_s(bytes_moved, ms));
    assert(h_output == expected);

    h_output = -1.0f;
    ms = coarsened_reduction(h_input.data(), &h_output, n);
    printf("5. Coarsened        (n = %d): %.1f   (%.3f ms, %.1f GB/s)\n", n, h_output, ms, bandwidth_gb_per_s(bytes_moved, ms));
    assert(h_output == expected);

    h_output = -1.0f;
    ms = warp_shuffle_reduction(h_input.data(), &h_output, n);
    printf("6. Warp shuffle     (n = %d): %.1f   (%.3f ms, %.1f GB/s)\n", n, h_output, ms, bandwidth_gb_per_s(bytes_moved, ms));
    assert(h_output == expected);

    return 0;
}
