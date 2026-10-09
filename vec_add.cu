#include <cassert>
#include <cstdio>
#include <vector>
#include "cuda_utils.cuh"

constexpr int kThreadsPerBlock = 256;


// Add two vectors element-wise: out[i] = a[i] + b[i]. One thread per element.
__global__ void vec_add_kernel(float* a, float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {    // the last block may have more threads than elements
        out[i] = a[i] + b[i];
    }
}

float vec_add(float* h_a, float* h_b, float* h_out, int n) {
    size_t bytes = n * sizeof(float);
    int num_blocks = (n + kThreadsPerBlock - 1) / kThreadsPerBlock;   // round up
    float *d_a, *d_b, *d_out;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    GpuTimer timer;
    timer.begin();

    vec_add_kernel<<<num_blocks, kThreadsPerBlock>>>(d_a, d_b, d_out, n);

    float ms = timer.end();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    return ms;
}

// Same addition with a fixed-size grid: each thread loops over the array
// in steps of the total thread count, so any n works with any grid size.
__global__ void vec_add_grid_stride_kernel(float* a, float* b, float* out, int n) {
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        out[i] = a[i] + b[i];
    }
}

float vec_add_grid_stride(float* h_a, float* h_b, float* h_out, int n) {
    size_t bytes = n * sizeof(float);
    float *d_a, *d_b, *d_out;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int num_sms = prop.multiProcessorCount;   // 40 on a T4
    int num_blocks = num_sms * 4;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    GpuTimer timer;
    timer.begin();

    vec_add_grid_stride_kernel<<<num_blocks, kThreadsPerBlock>>>(d_a, d_b, d_out, n);

    float ms = timer.end();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    return ms;
}

// Each thread adds four elements at once with 16-byte float4 loads and stores.
__global__ void vec_add_float4_kernel(float* a, float* b, float* out, int n) {
    const float4* a4 = reinterpret_cast<const float4*>(a);
    const float4* b4 = reinterpret_cast<const float4*>(b);
    float4* out4 = reinterpret_cast<float4*>(out);

    int n4 = n / 4; // number of float4 groups
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n4) {
        float4 va = a4[i];
        float4 vb = b4[i];

        float4 result;
        result.x = va.x + vb.x;
        result.y = va.y + vb.y;
        result.z = va.z + vb.z;
        result.w = va.w + vb.w;

        out4[i] = result;
    }

    // The last n % 4 elements do not fill a whole float4, thread n4 adds them one by one.
    if (i == n4) {
        for (int j = 4 * n4; j < n; j++) {
            out[j] = a[j] + b[j];
        }
    }

}

float vec_add_float4(float* h_a, float* h_b, float* h_out, int n) {
    size_t bytes = n * sizeof(float);
    float *d_a, *d_b, *d_out;

    int num_groups = (n + 3) / 4;   // round up so the remainder thread always exists
    int num_blocks = (num_groups + kThreadsPerBlock - 1) / kThreadsPerBlock;


    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    GpuTimer timer;
    timer.begin();

    vec_add_float4_kernel<<<num_blocks, kThreadsPerBlock>>>(d_a, d_b, d_out, n);

    float ms = timer.end();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    return ms;
}

int main() {
    int n = (1 << 24) + 3;    // 2^24 + 3

    std::vector<float> h_a(n);
    std::vector<float> h_b(n);

    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    // Reference result on the CPU.
    std::vector<float> expected(n);
    for (int i = 0; i < n; i++) {
        expected[i] = h_a[i] + h_b[i];
    }

    // Basic vector addition. -1 marks elements that no thread has written.
    std::vector<float> h_out(n, -1.0f);
    float ms = vec_add(h_a.data(), h_b.data(), h_out.data(), n);
    printf("basic:       %.3f ms\n", ms);
    assert(count_mismatches(h_out, expected) == 0);

    // Grid-stride vector addition
    std::vector<float> h_out_grid_stride(n, -1.0f);
    ms = vec_add_grid_stride(h_a.data(), h_b.data(), h_out_grid_stride.data(), n);
    printf("grid-stride: %.3f ms\n", ms);
    assert(count_mismatches(h_out_grid_stride, expected) == 0);

    // Float4 vector addition
    std::vector<float> h_out_float4(n, -1.0f);
    ms = vec_add_float4(h_a.data(), h_b.data(), h_out_float4.data(), n);
    printf("float4:      %.3f ms\n", ms);
    assert(count_mismatches(h_out_float4, expected) == 0);

    return 0;
}
