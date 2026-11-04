#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <algorithm>
#include <cassert>
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

// Runs launch() once untimed as a warm-up (module loading, clock ramp-up, caches),
// then timed_runs times with CUDA events, and returns the median time in ms.
// launch() must give the same result every time it runs, so kernels that
// accumulate into their output (atomicAdd) need the output reset inside launch().
template <typename Launch>
float time_kernel(Launch launch, int timed_runs = 20) {
    launch();
    cudaDeviceSynchronize();

    GpuTimer timer;
    std::vector<float> times(timed_runs);
    for (int run = 0; run < timed_runs; run++) {
        timer.begin();
        launch();
        times[run] = timer.end();
    }

    std::sort(times.begin(), times.end());
    return times[timed_runs / 2];
}

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

// Effective bandwidth in GB/s: bytes read plus bytes written, divided by the time.
inline double bandwidth_gb_per_s(size_t bytes, float ms) {
    return static_cast<double>(bytes) / (ms * 1e-3) / 1e9;
}

// Compute throughput in GFLOP/s: floating-point operations divided by the time.
inline double gflop_per_s(double flops, float ms) {
    return flops / (ms * 1e-3) / 1e9;
}

// Theoretical peak memory bandwidth of the current GPU in GB/s.
inline double peak_bandwidth_gb_per_s() {
    int device = 0;
    int clock_khz = 0;
    int bus_width_bits = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, device);
    cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, device);
    return 2.0 * clock_khz * 1e3 * bus_width_bits / 8 / 1e9;    // DDR: two transfers per clock.
}
#endif  // CUDA_UTILS_CUH
