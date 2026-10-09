#include <cassert>
#include <cstdio>
#include <vector>
#include "cuda_utils.cuh"

constexpr int kBlockDim = 16;   // threads per block in x and y
constexpr int kTileWidth = 16;


// Simple matrix multiplication p = m * n for square width x width matrices.
// One thread per element of p. Matrices are flat arrays in row-major order.
__global__ void matrix_mul_kernel(float* m, float* n, float* p, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < width) && (col < width)) {   // blocks at the right and bottom edge may reach beyond matrix
        float p_value = 0.0f;
        // Dot product of row of m and column of n.
        for (int k = 0; k < width; k++) {
            p_value += m[row*width+k] * n[k*width+col];
        }
        p[row*width+col] = p_value;
    }
}

float matrix_mul(float* h_m, float* h_n, float* h_p, int width) {
    size_t size = width * width * sizeof(float);
    dim3 threads_per_block(kBlockDim, kBlockDim);
    dim3 num_blocks((width + threads_per_block.x - 1) / threads_per_block.x,
                    (width + threads_per_block.y - 1) / threads_per_block.y);
    float *d_m, *d_n, *d_p;

    cudaMalloc(&d_m, size);
    cudaMalloc(&d_n, size);
    cudaMalloc(&d_p, size);

    cudaMemcpy(d_m, h_m, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_n, h_n, size, cudaMemcpyHostToDevice);

    GpuTimer timer;
    timer.begin();

    matrix_mul_kernel<<<num_blocks, threads_per_block>>>(d_m, d_n, d_p, width);

    float ms = timer.end();


    cudaMemcpy(h_p, d_p, size, cudaMemcpyDeviceToHost);

    cudaFree(d_m);
    cudaFree(d_n);
    cudaFree(d_p);

    return ms;
}

// Tiled matrix multiplication p = m * n. Each block computes one kTileWidth x kTileWidth
// tile of p. In every phase the block loads one tile of m and n into shared memory, and
// each thread adds kTileWidth products to its partial sum.
// Requires blockDim == (kTileWidth, kTileWidth).
__global__ void tiled_matrix_mul_kernel(float* m, float* n, float* p, int width) {
    __shared__ float m_tile[kTileWidth][kTileWidth];
    __shared__ float n_tile[kTileWidth][kTileWidth];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * kTileWidth + ty;
    int col = bx * kTileWidth + tx;

    float p_value = 0.0f;
    for (int phase = 0; phase < (width + kTileWidth - 1) / kTileWidth; phase++) {
        // Each thread loads one element of each tile. Positions outside the
        // matrix are filled with 0 so they do not change the sum.
        if (row < width && (phase * kTileWidth + tx) < width) {
            m_tile[ty][tx] = m[row*width + phase*kTileWidth + tx];
        } else {
            m_tile[ty][tx] = 0.0f;
        }
        if ((phase * kTileWidth + ty) < width && col < width) {
            n_tile[ty][tx] = n[(phase*kTileWidth + ty)*width + col];
        } else {
            n_tile[ty][tx] = 0.0f;
        }

        __syncthreads();    // both tiles must be completed before reading

        for (int k = 0; k < kTileWidth; k++) {
            p_value += m_tile[ty][k] * n_tile[k][tx];
        }
        __syncthreads();    // every thread must finish before the next phase overwrites the tiles
    }

    if (row < width && col < width) {
        p[row*width + col] = p_value;
    }
}

float tiled_matrix_mul(float* h_m, float* h_n, float* h_p, int width) {
    size_t size = width * width * sizeof(float);
    dim3 threads_per_block(kTileWidth, kTileWidth);
    dim3 num_blocks((width + kTileWidth - 1) / kTileWidth,
                    (width + kTileWidth - 1) / kTileWidth);
    float *d_m, *d_n, *d_p;

    cudaMalloc(&d_m, size);
    cudaMalloc(&d_n, size);
    cudaMalloc(&d_p, size);

    cudaMemcpy(d_m, h_m, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_n, h_n, size, cudaMemcpyHostToDevice);

    GpuTimer timer;
    timer.begin();

    tiled_matrix_mul_kernel<<<num_blocks, threads_per_block>>>(d_m, d_n, d_p, width);

    float ms = timer.end();

    cudaMemcpy(h_p, d_p, size, cudaMemcpyDeviceToHost);

    cudaFree(d_m);
    cudaFree(d_n);
    cudaFree(d_p);

    return ms;
}

int main() {
    int width = 1000;

    std::vector<float> h_m(width * width);
    std::vector<float> h_n(width * width);
    std::vector<float> h_p(width * width, -1.0f);

    for (int i = 0; i < width * width; i++) {
        h_m[i] = i % 7;
        h_n[i] = i % 5;
    }

    // Reference result on the CPU.
    std::vector<float> expected(width * width);
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            float acc = 0.0f;
            for (int k = 0; k < width; k++) {
                acc += h_m[row*width+k] * h_n[k*width+col];
            }
            expected[row*width+col] = acc;
        }
    }


    // Basic matrix multiplication. -1 marks elements that no thread has written.
    float ms = matrix_mul(h_m.data(), h_n.data(), h_p.data(), width);
    printf("basic:       %.3f ms\n", ms);
    assert(count_mismatches(h_p, expected) == 0);

    // Tiled matrix multiplication
    std::vector<float> h_p_tiled(width * width, -1.0f);
    ms = tiled_matrix_mul(h_m.data(), h_n.data(), h_p_tiled.data(), width);
    printf("tiled:       %.3f ms\n", ms);
    assert(count_mismatches(h_p_tiled, expected) == 0);

    return 0;
}
