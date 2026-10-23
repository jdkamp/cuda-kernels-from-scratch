#include <cassert>
#include <cstdio>
#include <vector>
#include "cuda_utils.cuh"

constexpr int kFilterRadius = 1;                                // filter is (2*r+1) x (2*r+1)
constexpr int kBlockDim = 16;                                   // threads per block in x and y (stages 1 and 2)
constexpr int kInTileDim = 32;                                  // input tile including halo = threads per block
constexpr int kOutTileDim = (kInTileDim - 2 * kFilterRadius);   // output pixels per block

__constant__ float c_filter[2*kFilterRadius+1][2*kFilterRadius+1];


// Reference on the CPU.
void convolution_2d_cpu(float* input, float* filter, float* output, int r, int width, int height) {
    for (int out_row = 0; out_row < height; out_row++) {
        for (int out_col = 0; out_col < width; out_col++) {
            float p_value = 0.0f;
            for (int f_row = 0; f_row < 2*r+1; f_row++) {
                for (int f_col = 0; f_col < 2*r+1; f_col++) {
                    int in_row = out_row - r + f_row;
                    int in_col = out_col - r + f_col;
                    if (in_row >= 0 && in_row < height && in_col >= 0 && in_col < width) {
                        p_value += filter[f_row*(2*r+1)+f_col] * input[in_row*width+in_col];
                    }
                }
            }
            output[out_row*width+out_col] = p_value;
        }
    }
}

// Stage 1: 2D convolution, one thread per output pixel. Image and filter are
// in global memory.
__global__ void convolution_2d_basic_kernel(float* input, float* filter, float* output, int r, int width, int height) {
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    float p_value = 0.0f;

    for (int f_row = 0; f_row < 2*r+1; f_row++) {
        for (int f_col = 0; f_col < 2*r+1; f_col++) {
            int in_row = out_row - r + f_row;   // - r centers the filter on the output pixel
            int in_col = out_col - r + f_col;
            if (in_row >= 0 && in_row < height && in_col >= 0 && in_col < width) {
                p_value += filter[f_row*(2*r+1)+f_col] * input[in_row*width+in_col];
            }
        }
    }

    if (out_row < height && out_col < width) {
        output[out_row*width+out_col] = p_value;
    }
}

float convolution_2d_basic(float* h_input, float* h_filter, float* h_output, int r, int width, int height) {
    size_t image_size = width * height * sizeof(float);
    size_t filter_size = (2*r+1) * (2*r+1) * sizeof(float);
    float *d_input, *d_filter, *d_output;

    cudaMalloc(&d_input, image_size);
    cudaMalloc(&d_output, image_size);
    cudaMalloc(&d_filter, filter_size);

    cudaMemcpy(d_input, h_input, image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, filter_size, cudaMemcpyHostToDevice);

    dim3 threads_per_block(kBlockDim, kBlockDim);
    dim3 num_blocks((width  + threads_per_block.x - 1) / threads_per_block.x,
                    (height + threads_per_block.y - 1) / threads_per_block.y);

    GpuTimer timer;
    timer.begin();

    convolution_2d_basic_kernel<<<num_blocks, threads_per_block>>>(d_input, d_filter, d_output, r, width, height);

    float ms = timer.end();

    cudaMemcpy(h_output, d_output, image_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);

    return ms;
}

// Stage 2: same as stage 1, but the filter is in constant memory.
// Requires the radius to be fixed at compile time.
__global__ void convolution_2d_const_mem_kernel(float* input, float* output, int width, int height) {
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    float p_value = 0.0f;

    for (int f_row = 0; f_row < 2*kFilterRadius+1; f_row++) {
        for (int f_col = 0; f_col < 2*kFilterRadius+1; f_col++) {
            int in_row = out_row - kFilterRadius + f_row;
            int in_col = out_col - kFilterRadius + f_col;
            if (in_row >= 0 && in_row < height && in_col >= 0 && in_col < width) {
                p_value += c_filter[f_row][f_col] * input[in_row*width+in_col];
            }
        }
    }

    if (out_row < height && out_col < width) {
        output[out_row*width+out_col] = p_value;
    }
}

float convolution_2d_const_mem(float* h_input, float* h_filter, float* h_output, int width, int height) {
    size_t image_size = width * height * sizeof(float);
    size_t filter_size = (2*kFilterRadius+1) * (2*kFilterRadius+1) * sizeof(float);
    float *d_input, *d_output;

    cudaMalloc(&d_input, image_size);
    cudaMalloc(&d_output, image_size);
    cudaMemcpyToSymbol(c_filter, h_filter, filter_size);  // copies into the __constant__ variable

    cudaMemcpy(d_input, h_input, image_size, cudaMemcpyHostToDevice);

    dim3 threads_per_block(kBlockDim, kBlockDim);
    dim3 num_blocks((width  + threads_per_block.x - 1) / threads_per_block.x,
                    (height + threads_per_block.y - 1) / threads_per_block.y);

    GpuTimer timer;
    timer.begin();

    convolution_2d_const_mem_kernel<<<num_blocks, threads_per_block>>>(d_input, d_output, width, height);

    float ms = timer.end();

    cudaMemcpy(h_output, d_output, image_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return ms;
}

// Stage 3: tiled convolution. Each block has kInTileDim x kInTileDim threads and
// loads one input tile including the halo into shared memory, one pixel per thread.
// Only the inner kOutTileDim x kOutTileDim threads compute an output pixel.
__global__ void convolution_2d_tiled_const_mem_kernel(float* input, float* output, int width, int height) {
    __shared__ float input_tile[kInTileDim][kInTileDim];
    // Output tiles are adjacent, input tiles overlap: the input tile starts kFilterRadius earlier.
    int col = blockIdx.x * kOutTileDim + threadIdx.x - kFilterRadius;
    int row = blockIdx.y * kOutTileDim + threadIdx.y - kFilterRadius;

    // loading input tiles
    if (row >= 0 && row < height && col >= 0 && col < width) {
        input_tile[threadIdx.y][threadIdx.x] = input[row*width+col];
    } else {
        input_tile[threadIdx.y][threadIdx.x] = 0.0f;   // ghost cell
    }
    __syncthreads();    // whole tile must be loaded before anyone reads

    // Output elements
    int tile_col = threadIdx.x - kFilterRadius;
    int tile_row = threadIdx.y - kFilterRadius;

    if (col >= 0 && col < width && row >= 0 && row < height) {                                      // pixels inside the image
        if (tile_col >= 0 && tile_col < kOutTileDim && tile_row >= 0 && tile_row < kOutTileDim) { // no halo thread
            float p_value = 0.0f;
            for (int f_row = 0; f_row < 2*kFilterRadius+1; f_row++) {
                for (int f_col = 0; f_col < 2*kFilterRadius+1; f_col++) {
                    p_value += c_filter[f_row][f_col] * input_tile[tile_row+f_row][tile_col+f_col]; // Neighbourhood starts at [tile_row][tile_col], ghost cells are already 0.
                }
            }
            output[row*width+col] = p_value;
        }
    }
}

float convolution_2d_tiled_const_mem(float* h_input, float* h_filter, float* h_output, int width, int height) {
    size_t image_size = width * height * sizeof(float);
    size_t filter_size = (2*kFilterRadius+1) * (2*kFilterRadius+1) * sizeof(float);
    float *d_input, *d_output;

    cudaMalloc(&d_input, image_size);
    cudaMalloc(&d_output, image_size);
    cudaMemcpyToSymbol(c_filter, h_filter, filter_size);

    cudaMemcpy(d_input, h_input, image_size, cudaMemcpyHostToDevice);

    // Blocks have kInTileDim threads per side but produce only kOutTileDim pixels,
    // so the grid steps in kOutTileDim.
    dim3 threads_per_block(kInTileDim, kInTileDim);
    dim3 num_blocks((width  + kOutTileDim - 1) / kOutTileDim,
                    (height + kOutTileDim - 1) / kOutTileDim);

    GpuTimer timer;
    timer.begin();

    convolution_2d_tiled_const_mem_kernel<<<num_blocks, threads_per_block>>>(d_input, d_output, width, height);

    float ms = timer.end();

    cudaMemcpy(h_output, d_output, image_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return ms;
}


int main() {
    // Not square and no multiple of kBlockDim or kOutTileDim, so every kernel needs
    // several blocks in both directions and the tiled kernel loads halos from neighbouring tiles.
    int width = 70;
    int height = 45;
    int r = kFilterRadius;
    int filter_dim = 2*r+1;

    std::vector<float> h_input(width*height);
    for (int i = 0; i < width*height; i++) {
        h_input[i] = i % 7;
    }
    std::vector<float> h_filter(filter_dim*filter_dim);
    for (int k = 0; k < filter_dim*filter_dim; k++) {
        h_filter[k] = k + 1;
    }

    // Reference result on the CPU.
    std::vector<float> expected(width*height);
    convolution_2d_cpu(h_input.data(), h_filter.data(), expected.data(), r, width, height);

    // Basic, filter in global memory. -1 marks pixels that no thread has written.
    std::vector<float> h_output_basic(width*height, -1.0f);
    float ms = convolution_2d_basic(h_input.data(), h_filter.data(), h_output_basic.data(), r, width, height);
    printf("basic:           %.3f ms\n", ms);
    assert(count_mismatches(h_output_basic, expected) == 0);

    // Filter in constant memory
    std::vector<float> h_output_const(width*height, -1.0f);
    ms = convolution_2d_const_mem(h_input.data(), h_filter.data(), h_output_const.data(), width, height);
    printf("constant memory: %.3f ms\n", ms);
    assert(count_mismatches(h_output_const, expected) == 0);

    // Filter in constant memory and tiled
    std::vector<float> h_output_tiled(width*height, -1.0f);
    ms = convolution_2d_tiled_const_mem(h_input.data(), h_filter.data(), h_output_tiled.data(), width, height);
    printf("tiled:           %.3f ms\n", ms);
    assert(count_mismatches(h_output_tiled, expected) == 0);

    return 0;
}
