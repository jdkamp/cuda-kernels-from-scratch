#pragma once

#include <vector>

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
int count_mismatches(const std::vector<float>& result, const std::vector<float>& expected) {
    int mismatch = 0;
    for (size_t i = 0; i < result.size(); i++){
        if (result[i] != expected[i]) mismatch++;
    }
    return mismatch;
}