#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "cuda_utils.cuh"

constexpr int kNumBins = 7;
constexpr int kLettersPerBin = 4;
constexpr int kThreadsPerBlock = 1024;
constexpr int kCoarseFactor = 32;

// Reference on the CPU. Counts lowercase letters in groups of kLettersPerBin, 
// all other characters are ignored.
void histogram_cpu(const char* data, int n, unsigned int* histo) {
    for (int i = 0; i < n; i++) {
        int position = data[i] - 'a';
        if (position >= 0 && position <= 25) {
            histo[position/kLettersPerBin]++;
        }
    }
}

// Stage 1: One thread per character, every thread increments the global histogram
// with atomicAdd.
__global__ void histogram_basic_kernel(const char* data, int n, unsigned int* histo) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int position = data[i] - 'a';
        if (position >= 0 && position <= 25) {
            atomicAdd(&histo[position/kLettersPerBin], 1);
        }
    }
}

float histogram_basic(const char* h_data, int n, unsigned int* h_histo) {
    size_t bytes_data = n * sizeof(char);
    size_t bytes_histo = kNumBins * sizeof(unsigned int);
    int num_blocks = (n + kThreadsPerBlock - 1) / kThreadsPerBlock;
    char *d_data;
    unsigned int *d_histo;

    cudaMalloc(&d_data, bytes_data);
    cudaMalloc(&d_histo, bytes_histo);

    cudaMemcpy(d_data, h_data, bytes_data, cudaMemcpyHostToDevice);
    cudaMemcpy(d_histo, h_histo, bytes_histo, cudaMemcpyHostToDevice);

    GpuTimer timer;
    timer.begin();

    histogram_basic_kernel<<<num_blocks, kThreadsPerBlock>>>(d_data, n, d_histo);

    float ms = timer.end();

    cudaMemcpy(h_histo, d_histo, bytes_histo, cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_histo);

    return ms;
}

// Stage 2: Privatization. Each block counts into its own histogram in shared memory
// and adds it to the global histogram once at the end.
__global__ void histogram_private_kernel(const char* data, int n, unsigned int* histo) {
    __shared__ unsigned int histo_s[kNumBins];

    if (threadIdx.x < kNumBins) {
        histo_s[threadIdx.x] = 0;
    }

    __syncthreads();    // all bins must be 0 before anyone counts

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int position = data[i] - 'a';
        if (position >= 0 && position <= 25) {
            atomicAdd(&histo_s[position/kLettersPerBin], 1);
        }
    }

    __syncthreads();    // all counts must be done before the merge

    if (threadIdx.x < kNumBins) {
        atomicAdd(&histo[threadIdx.x], histo_s[threadIdx.x]);
    }
}

float histogram_private(const char* h_data, int n, unsigned int* h_histo) {
    size_t bytes_data = n * sizeof(char);
    size_t bytes_histo = kNumBins * sizeof(unsigned int);
    int num_blocks = (n + kThreadsPerBlock - 1) / kThreadsPerBlock;
    char *d_data;
    unsigned int *d_histo;

    cudaMalloc(&d_data, bytes_data);
    cudaMalloc(&d_histo, bytes_histo);

    cudaMemcpy(d_data, h_data, bytes_data, cudaMemcpyHostToDevice);
    cudaMemcpy(d_histo, h_histo, bytes_histo, cudaMemcpyHostToDevice);

    GpuTimer timer;
    timer.begin();

    histogram_private_kernel<<<num_blocks, kThreadsPerBlock>>>(d_data, n, d_histo);

    float ms = timer.end();

    cudaMemcpy(h_histo, d_histo, bytes_histo, cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_histo);

    return ms;
}

// Stage 3: Privatization + coarsening with contiguous partitioning.
// Each thread counts kCoarseFactor consecutive characters.
__global__ void histogram_contiguous_kernel(const char* data, int n, unsigned int* histo) {
    __shared__ unsigned int histo_s[kNumBins];

    if (threadIdx.x < kNumBins) {
        histo_s[threadIdx.x] = 0;
    }

    __syncthreads();    // all bins must be 0 before anyone counts

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid*kCoarseFactor; i < (tid+1)*kCoarseFactor; i++) {
        if (i < n) {
            int position = data[i] - 'a';
            if (position >= 0 && position <= 25) {
                atomicAdd(&histo_s[position/kLettersPerBin], 1);
            }
        }
    }

    __syncthreads();    // all counts must be done before the merge

    if (threadIdx.x < kNumBins) {
        atomicAdd(&histo[threadIdx.x], histo_s[threadIdx.x]);
    }
}

float histogram_contiguous(const char* h_data, int n, unsigned int* h_histo) {
    size_t bytes_data = n * sizeof(char);
    size_t bytes_histo = kNumBins * sizeof(unsigned int);
    int num_blocks = (n + kThreadsPerBlock * kCoarseFactor - 1) / (kThreadsPerBlock * kCoarseFactor);
    char *d_data;
    unsigned int *d_histo;

    cudaMalloc(&d_data, bytes_data);
    cudaMalloc(&d_histo, bytes_histo);

    cudaMemcpy(d_data, h_data, bytes_data, cudaMemcpyHostToDevice);
    cudaMemcpy(d_histo, h_histo, bytes_histo, cudaMemcpyHostToDevice);

    GpuTimer timer;
    timer.begin();

    histogram_contiguous_kernel<<<num_blocks, kThreadsPerBlock>>>(d_data, n, d_histo);

    float ms = timer.end();

    cudaMemcpy(h_histo, d_histo, bytes_histo, cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_histo);

    return ms;
}

// Stage 4: Privatization + coarsening with interleaved partitioning.
// In each loop iteration the threads of a warp read consecutive characters (coalesced).
__global__ void histogram_interleaved_kernel(const char* data, int n, unsigned int* histo) {
    __shared__ unsigned int histo_s[kNumBins];

    if (threadIdx.x < kNumBins) {
        histo_s[threadIdx.x] = 0;
    }

    __syncthreads();    // all bins must be 0 before anyone counts

    int tid = blockIdx.x * blockDim.x + threadIdx.x;    // global thread index
    int stride = blockDim.x * gridDim.x;                // total number of threads in the grid
    for (int i = tid; i < n; i += stride) {
        int position = data[i] - 'a';
        if (position >= 0 && position <= 25) {
            atomicAdd(&histo_s[position/kLettersPerBin], 1);
        }
    }

    __syncthreads();    // all counts must be done before the merge

    if (threadIdx.x < kNumBins) {
        atomicAdd(&histo[threadIdx.x], histo_s[threadIdx.x]);
    }
}

float histogram_interleaved(const char* h_data, int n, unsigned int* h_histo) {
    size_t bytes_data = n * sizeof(char);
    size_t bytes_histo = kNumBins * sizeof(unsigned int);
    int num_blocks = (n + kThreadsPerBlock * kCoarseFactor - 1) / (kThreadsPerBlock * kCoarseFactor);
    char *d_data;
    unsigned int *d_histo;

    cudaMalloc(&d_data, bytes_data);
    cudaMalloc(&d_histo, bytes_histo);

    cudaMemcpy(d_data, h_data, bytes_data, cudaMemcpyHostToDevice);
    cudaMemcpy(d_histo, h_histo, bytes_histo, cudaMemcpyHostToDevice);

    GpuTimer timer;
    timer.begin();

    histogram_interleaved_kernel<<<num_blocks, kThreadsPerBlock>>>(d_data, n, d_histo);

    float ms = timer.end();

    cudaMemcpy(h_histo, d_histo, bytes_histo, cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_histo);

    return ms;
}

// Stage 5: Privatization + interleaved coarsening + aggregation. Each thread collects
// runs of the same bin locally and writes them with a single atomicAdd.
__global__ void histogram_aggregation_kernel(const char* data, int n, unsigned int* histo) {
    __shared__ unsigned int histo_s[kNumBins];

    if (threadIdx.x < kNumBins) {
        histo_s[threadIdx.x] = 0;
    }

    __syncthreads();    // all bins must be 0 before anyone counts

    unsigned int accumulator = 0;
    int prev_bin = -1;  // no bin yet

    int tid = blockIdx.x * blockDim.x + threadIdx.x;    // global thread index
    int stride = blockDim.x * gridDim.x;                // total number of threads in the grid
    for (int i = tid; i < n; i += stride) {
        int position = data[i] - 'a';
        if (position >= 0 && position <= 25) {
            int bin = position / kLettersPerBin;
            if (bin == prev_bin) {
                accumulator++;
            } else {
                if (accumulator > 0) {
                    atomicAdd(&histo_s[prev_bin], accumulator);   // write the finished run
                }
                accumulator = 1;    // new run, the current character already counts
                prev_bin = bin;
            }
        }
    }
    if (accumulator > 0) {
        atomicAdd(&histo_s[prev_bin], accumulator);   // last bin must be added
    }

    __syncthreads();    // all counts must be done before the merge

    if (threadIdx.x < kNumBins) {
        atomicAdd(&histo[threadIdx.x], histo_s[threadIdx.x]);
    }
}

float histogram_aggregation(const char* h_data, int n, unsigned int* h_histo) {
    size_t bytes_data = n * sizeof(char);
    size_t bytes_histo = kNumBins * sizeof(unsigned int);
    int num_blocks = (n + kThreadsPerBlock * kCoarseFactor - 1) / (kThreadsPerBlock * kCoarseFactor);
    char *d_data;
    unsigned int *d_histo;

    cudaMalloc(&d_data, bytes_data);
    cudaMalloc(&d_histo, bytes_histo);

    cudaMemcpy(d_data, h_data, bytes_data, cudaMemcpyHostToDevice);
    cudaMemcpy(d_histo, h_histo, bytes_histo, cudaMemcpyHostToDevice);

    GpuTimer timer;
    timer.begin();

    histogram_aggregation_kernel<<<num_blocks, kThreadsPerBlock>>>(d_data, n, d_histo);

    float ms = timer.end();

    cudaMemcpy(h_histo, d_histo, bytes_histo, cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_histo);

    return ms;
}

int main() {
    int n = (1 << 24) + 3;

    std::vector<char> h_data(n);
    srand(0);   // reproducibility
    for (int i = 0; i < n; i++) {
        if (rand() % 10 == 0) {
            h_data[i] = ' ';
        } else {
            h_data[i] = 'a' + rand() % 26;
        }
    }

    std::vector<unsigned int> expected(kNumBins, 0);
    histogram_cpu(h_data.data(), n, expected.data());

    std::vector<unsigned int> h_histo_basic(kNumBins, 0);
    float ms = histogram_basic(h_data.data(), n, h_histo_basic.data());
    printf("basic:       %.3f ms\n", ms);
    assert(count_mismatches(h_histo_basic, expected) == 0);

    std::vector<unsigned int> h_histo_private(kNumBins, 0);
    ms = histogram_private(h_data.data(), n, h_histo_private.data());
    printf("private:     %.3f ms\n", ms);
    assert(count_mismatches(h_histo_private, expected) == 0);

    std::vector<unsigned int> h_histo_contiguous(kNumBins, 0);
    ms = histogram_contiguous(h_data.data(), n, h_histo_contiguous.data());
    printf("contiguous:  %.3f ms\n", ms);
    assert(count_mismatches(h_histo_contiguous, expected) == 0);

    std::vector<unsigned int> h_histo_interleaved(kNumBins, 0);
    ms = histogram_interleaved(h_data.data(), n, h_histo_interleaved.data());
    printf("interleaved: %.3f ms\n", ms);
    assert(count_mismatches(h_histo_interleaved, expected) == 0);

    std::vector<unsigned int> h_histo_aggregation(kNumBins, 0);
    ms = histogram_aggregation(h_data.data(), n, h_histo_aggregation.data());
    printf("aggregation: %.3f ms\n", ms);
    assert(count_mismatches(h_histo_aggregation, expected) == 0);

    return 0;
}
