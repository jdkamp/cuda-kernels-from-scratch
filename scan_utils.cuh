#ifndef SCAN_UTILS_CUH
#define SCAN_UTILS_CUH
constexpr int kWarpSize = 32;

// Inclusive scan of val across the 32 lanes of a warp.
__device__ int warp_scan(int val) {
    int lane = threadIdx.x % kWarpSize;
    for (int offset = 1; offset < kWarpSize; offset *= 2) {
        int left = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) {
            val += left;
        }
    }
    return val;
}

// Inclusive scan of val across the whole block: scan within each warp,
// scan the warp totals, add the total of all previous warps.
// Requires blockDim = kBlockDim, kNumWarps <= kWarpSize
__device__ int block_scan(int val) {
    __shared__ int warp_sums[kWarpSize];
    int lane = threadIdx.x % kWarpSize;
    int warp = threadIdx.x / kWarpSize;
    int num_warps = blockDim.x / kWarpSize;

    val = warp_scan(val);
    if (lane == kWarpSize - 1) {
        warp_sums[warp] = val;
    }

    __syncthreads();

    if (warp == 0) {
        int warp_sum = (threadIdx.x<num_warps) ? warp_sums[threadIdx.x] : 0;
        warp_sum = warp_scan(warp_sum);
        if (threadIdx.x < num_warps) {
            warp_sums[threadIdx.x] = warp_sum;
        }
    }

    __syncthreads();

    if (warp > 0) {
        val += warp_sums[warp-1];
    }

    return val;
}

#endif // SCAN_UTILS_CUH