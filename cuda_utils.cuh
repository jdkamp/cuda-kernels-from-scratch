#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <vector>
#include <cassert>

// Times a kernel with CUDA events.
struct GpuTimer {
    cudaEvent_t start, stop;

    GpuTimer()  { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~GpuTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }

    void begin() { cudaEventRecord(start); }
    float end() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// Returns the number of elements where result and expected differ.
template <typename T>
int count_mismatches(const std::vector<T>& result, const std::vector<T>& expected) {
    assert(result.size() == expected.size());
    int mismatch = 0;
    for (size_t i = 0; i < result.size(); i++){
        if (result[i] != expected[i]) mismatch++;
    }
    return mismatch;
}

#endif  // CUDA_UTILS_CUH
