#include <cassert>
#include <cstdio>
#include <vector>
#include "cuda_utils.cuh"

// Stencil coefficients. Small distinct integers instead of realistic values:
// - integers keep every sum in float, so GPU and CPU must match exactly
// - distinct values per neighbour make a swapped neighbour show up as a mismatch
// kC0: center, kC1/kC2: x+1/x-1, kC3/kC4: y+1/y-1, kC5/kC6: z+1/z-1
constexpr float kC0 = 1;
constexpr float kC1 = 2;
constexpr float kC2 = 3;
constexpr float kC3 = 4;
constexpr float kC4 = 5;
constexpr float kC5 = 6;
constexpr float kC6 = 7;

constexpr int kBlockDim = 8;
constexpr int kInTileDim = 8;
constexpr int kOutTileDim = kInTileDim - 2;
constexpr int kCoarseInTileDim = 32;
constexpr int kCoarseOutTileDim = kCoarseInTileDim - 2;

// Reference on CPU. Computes all interior points, boundary points are not written.
void stencil_cpu(const float* input, float* output, int n) {
    for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < n - 1; j++) {
            for (int k = 1; k < n - 1; k++) {
                int center = i * n * n + j * n + k;
                output[center] = kC0 * input[center]
                               + kC1 * input[center+1]
                               + kC2 * input[center-1]
                               + kC3 * input[center+n]
                               + kC4 * input[center-n]
                               + kC5 * input[center+(n*n)]
                               + kC6 * input[center-(n*n)];
            }
        }
    }
}

// Stage 1: 7-point stencil, one thread per grid point, input from global memory.
// Boundary points are not written.
__global__ void stencil_basic_kernel(const float* input, float* output, int n) {
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= 1 && i < n-1 && j >= 1 && j < n-1 && k >= 1 && k < n-1) {
        int center = i * n * n + j * n + k;
        output[center] = kC0 * input[center]
                       + kC1 * input[center+1]
                       + kC2 * input[center-1]
                       + kC3 * input[center+n]
                       + kC4 * input[center-n]
                       + kC5 * input[center+(n*n)]
                       + kC6 * input[center-(n*n)];
    }
}

float stencil_basic(float* h_input, float* h_output, int n) {
    size_t bytes =n * n * n * sizeof(float);
    dim3 threads_per_block(kBlockDim, kBlockDim, kBlockDim);
    dim3 num_blocks((n + kBlockDim - 1) / kBlockDim,
                    (n + kBlockDim - 1) / kBlockDim,
                    (n + kBlockDim - 1) / kBlockDim);
    float *d_input, *d_output;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, bytes, cudaMemcpyHostToDevice);

    GpuTimer timer;
    timer.begin();

    stencil_basic_kernel<<<num_blocks, threads_per_block >>>(d_input, d_output, n);

    float ms = timer.end();

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return ms;
}

// Stage 2: Shared memory tiling. Each block loads a kInTileDim^3 input tile including
// the halo, one point per thread. Only the inner kOutTileDim^3 threads compute an
// output point. Requires blockDim = (kInTileDim, kInTileDim, kInTileDim).
__global__ void stencil_shared_mem_kernel(const float* input, float* output, int n) {
    __shared__ float input_tile[kInTileDim][kInTileDim][kInTileDim];

    int i = blockIdx.z * kOutTileDim + threadIdx.z - 1;
    int j = blockIdx.y * kOutTileDim + threadIdx.y - 1;
    int k = blockIdx.x * kOutTileDim + threadIdx.x - 1;

    // loading input tiles
    if (i >= 0 && i < n && j >= 0 && j < n && k >= 0 && k < n) {
        input_tile[threadIdx.z][threadIdx.y][threadIdx.x] = input[i * n * n + j * n + k];
    } else {
        input_tile[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;   // ghost cell
    }
    __syncthreads();    // whole tile must be loaded before anyone reads

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    if (i >= 1 && i < n-1 && j >= 1 && j < n-1 && k >= 1 && k < n-1) {                                       // interior point of the grid
        if (tx >= 1 && tx < kInTileDim-1 && ty >= 1 && ty < kInTileDim-1 && tz >= 1 && tz < kInTileDim-1) {  // no halo thread
            output[i * n * n + j * n + k] = kC0 * input_tile[tz][ty][tx]
                                          + kC1 * input_tile[tz][ty][tx+1]
                                          + kC2 * input_tile[tz][ty][tx-1]
                                          + kC3 * input_tile[tz][ty+1][tx]
                                          + kC4 * input_tile[tz][ty-1][tx]
                                          + kC5 * input_tile[tz+1][ty][tx]
                                          + kC6 * input_tile[tz-1][ty][tx];
        }
    }
}

float stencil_shared_mem(float* h_input, float* h_output, int n) {
    size_t bytes =n * n * n * sizeof(float);
    dim3 threads_per_block(kInTileDim, kInTileDim, kInTileDim);
    dim3 num_blocks((n + kOutTileDim - 1) / kOutTileDim,
                    (n + kOutTileDim - 1) / kOutTileDim,
                    (n + kOutTileDim - 1) / kOutTileDim);
    float *d_input, *d_output;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, bytes, cudaMemcpyHostToDevice);

    GpuTimer timer;
    timer.begin();

    stencil_shared_mem_kernel<<<num_blocks, threads_per_block >>>(d_input, d_output, n);

    float ms = timer.end();

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return ms;
}

// Stage 3: Thread coarsening in z. 2D blocks of kCoarseInTileDim^2 threads walk through
// kCoarseOutTileDim planes one after another. Only three planes (prev, curr, next) are
// kept in shared memory and shifted after each plane, so every input plane is loaded once.
// Requires blockDim = (kCoarseInTileDim, kCoarseInTileDim, 1).
__global__ void stencil_coarsened_kernel(const float* input, float* output, int n) {
    __shared__ float prev_tile[kCoarseInTileDim][kCoarseInTileDim];
    __shared__ float curr_tile[kCoarseInTileDim][kCoarseInTileDim];
    __shared__ float next_tile[kCoarseInTileDim][kCoarseInTileDim];

    int i_start = blockIdx.z * kCoarseOutTileDim;   // first output plane of this block
    // Input tile starts one point earlier in x and y (halo).
    int j = blockIdx.y * kCoarseOutTileDim + threadIdx.y - 1;
    int k = blockIdx.x * kCoarseOutTileDim + threadIdx.x - 1;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Load the plane below the first output plane and the first output plane itself.
    if (i_start - 1 >= 0 && i_start - 1 < n && j >= 0 && j < n && k >= 0 && k < n) {
        prev_tile[threadIdx.y][threadIdx.x] = input[(i_start - 1) * n * n + j * n + k];
    } else {
        prev_tile[threadIdx.y][threadIdx.x] = 0.0f;   // ghost cell
    }

    if (i_start >= 0 && i_start < n && j >= 0 && j < n && k >= 0 && k < n) {
        curr_tile[threadIdx.y][threadIdx.x] = input[(i_start) * n * n + j * n + k];
    } else {
        curr_tile[threadIdx.y][threadIdx.x] = 0.0f;   // ghost cell
    }

    for (int i = i_start; i < i_start + kCoarseOutTileDim; i++) {
        // Load the plane above the current one.
        if (i+1 >= 0 && i+1 < n && j >= 0 && j < n && k >= 0 && k < n) {
            next_tile[threadIdx.y][threadIdx.x] = input[(i+1)* n * n + j * n + k];
        } else {
            next_tile[threadIdx.y][threadIdx.x] = 0.0f;   // ghost cell
        }

        __syncthreads();    // all three planes must be loaded before anyone reads

        if (i >= 1 && i < n-1 && j >= 1 && j < n-1 && k >= 1 && k < n-1) {                                       // interior point of the grid
            if (tx >= 1 && tx < kCoarseInTileDim-1 && ty >= 1 && ty < kCoarseInTileDim-1) {   // no halo thread
                output[i * n * n + j * n + k] = kC0 * curr_tile[ty][tx]
                                            + kC1 * curr_tile[ty][tx+1]
                                            + kC2 * curr_tile[ty][tx-1]
                                            + kC3 * curr_tile[ty+1][tx]
                                            + kC4 * curr_tile[ty-1][tx]
                                            + kC5 * next_tile[ty][tx]
                                            + kC6 * prev_tile[ty][tx];
            }
        }

        __syncthreads();    // all reads must be done before the planes are shifted

        // Shift the planes up by one. Order matters: curr must be copied before it is overwritten.
        prev_tile[threadIdx.y][threadIdx.x] = curr_tile[threadIdx.y][threadIdx.x];
        curr_tile[threadIdx.y][threadIdx.x] = next_tile[threadIdx.y][threadIdx.x];
    }
}

float stencil_coarsened(float* h_input, float* h_output, int n) {
    size_t bytes =n * n * n * sizeof(float);
    dim3 threads_per_block(kCoarseInTileDim, kCoarseInTileDim, 1);
    dim3 num_blocks((n + kCoarseOutTileDim - 1) / kCoarseOutTileDim,
                    (n + kCoarseOutTileDim - 1) / kCoarseOutTileDim,
                    (n + kCoarseOutTileDim - 1) / kCoarseOutTileDim);
    float *d_input, *d_output;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, bytes, cudaMemcpyHostToDevice);

    GpuTimer timer;
    timer.begin();

    stencil_coarsened_kernel<<<num_blocks, threads_per_block >>>(d_input, d_output, n);

    float ms = timer.end();

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return ms;
}

// Stage 4: Register tiling. Same plane walk as stage 3, but the previous and next plane
// are only read at the thread's own position, so they live in registers. Only the current
// plane stays in shared memory because neighbouring threads read it for x and y.
// Requires blockDim = (kCoarseInTileDim, kCoarseInTileDim, 1).
__global__ void stencil_register_tiling_kernel(const float* input, float* output, int n) {
    __shared__ float curr_tile[kCoarseInTileDim][kCoarseInTileDim];

    int i_start = blockIdx.z * kCoarseOutTileDim;   // first output plane of this block
    // Input tile starts one point earlier in x and y (halo).
    int j = blockIdx.y * kCoarseOutTileDim + threadIdx.y - 1;
    int k = blockIdx.x * kCoarseOutTileDim + threadIdx.x - 1;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Plane below the first output plane: only needed by this thread.
    float prev = 0.0f;
    if (i_start - 1 >= 0 && i_start - 1 < n && j >= 0 && j < n && k >= 0 && k < n) {
        prev = input[(i_start - 1) * n * n + j * n + k];
    }
    // First output plane: in a register for this thread, in shared memory for its neighbours.
    float curr = 0.0f;
    if (i_start >= 0 && i_start < n && j >= 0 && j < n && k >= 0 && k < n) {
        curr = input[i_start * n * n + j * n + k];
    }
    curr_tile[ty][tx] = curr;

    for (int i = i_start; i < i_start + kCoarseOutTileDim; i++) {
        // Plane above the current one: only needed by this thread.
        float next = 0.0f;
        if (i+1 >= 0 && i+1 < n && j >= 0 && j < n && k >= 0 && k < n) {
            next = input[(i+1) * n * n + j * n + k];
        }

        __syncthreads();    // curr_tile must be complete before anyone reads neighbours

        if (i >= 1 && i < n-1 && j >= 1 && j < n-1 && k >= 1 && k < n-1) {                    // interior point of the grid
            if (tx >= 1 && tx < kCoarseInTileDim-1 && ty >= 1 && ty < kCoarseInTileDim-1) {   // no halo thread
                output[i * n * n + j * n + k] = kC0 * curr_tile[ty][tx]
                                            + kC1 * curr_tile[ty][tx+1]
                                            + kC2 * curr_tile[ty][tx-1]
                                            + kC3 * curr_tile[ty+1][tx]
                                            + kC4 * curr_tile[ty-1][tx]
                                            + kC5 * next
                                            + kC6 * prev;
            }
        }

        __syncthreads();    // all reads must be done before curr_tile is overwritten

        // Shift the planes up by one in registers, then publish the new current plane to the neighbours.
        prev = curr;
        curr = next;
        curr_tile[ty][tx] = curr;
    }
}

float stencil_register_tiling(float* h_input, float* h_output, int n) {
    size_t bytes =n * n * n * sizeof(float);
    dim3 threads_per_block(kCoarseInTileDim, kCoarseInTileDim, 1);
    dim3 num_blocks((n + kCoarseOutTileDim - 1) / kCoarseOutTileDim,
                    (n + kCoarseOutTileDim - 1) / kCoarseOutTileDim,
                    (n + kCoarseOutTileDim - 1) / kCoarseOutTileDim);
    float *d_input, *d_output;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, bytes, cudaMemcpyHostToDevice);

    GpuTimer timer;
    timer.begin();

    stencil_register_tiling_kernel<<<num_blocks, threads_per_block >>>(d_input, d_output, n);

    float ms = timer.end();

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return ms;
}

int main() {
    int n = 512;
    std::vector<float> h_input(n*n*n);
    for (int i = 0; i < n*n*n; i++) {
        h_input[i] = i % 7;
    }
    std::vector<float> expected(n*n*n, -1.0f);
    stencil_cpu(h_input.data(), expected.data(), n);

    std::vector<float> h_output(n*n*n, -1.0f);
    float ms = stencil_basic(h_input.data(), h_output.data(), n);
    printf("basic:      %.3f ms\n", ms);
    assert(count_mismatches(h_output, expected) == 0);

    std::vector<float> h_output_shared_mem(n*n*n, -1.0f);
    ms = stencil_shared_mem(h_input.data(), h_output_shared_mem.data(), n);
    printf("shared mem: %.3f ms\n", ms);
    assert(count_mismatches(h_output_shared_mem, expected) == 0);

    std::vector<float> h_output_coarsened(n*n*n, -1.0f);
    ms = stencil_coarsened(h_input.data(), h_output_coarsened.data(), n);
    printf("coarsened:  %.3f ms\n", ms);
    assert(count_mismatches(h_output_coarsened, expected) == 0);

    std::vector<float> h_output_register_tiling(n*n*n, -1.0f);
    ms = stencil_register_tiling(h_input.data(), h_output_register_tiling.data(), n);
    printf("register:   %.3f ms\n", ms);
    assert(count_mismatches(h_output_register_tiling, expected) == 0);

    return 0;
}