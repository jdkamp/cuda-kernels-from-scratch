# cuda-kernels-from-scratch

While learning GPU programming I implemented the core parallel patterns in CUDA from scratch, testing them and measuring how fast they run. Each file takes one pattern from a basic kernel through optimizations, such as shared memory, thread coarsening and warp-level primitives, and checks and times every stage. 

## Kernels

| File | Pattern | Stages |
|---|---|---|
| [`vec_add.cu`](vec_add.cu) | Vector addition | One thread per element, grid-stride loop, `float4` loads and stores |
| [`matrix_mul.cu`](matrix_mul.cu) | Matrix multiplication | Naive, tiled in shared memory |
| [`reduction.cu`](reduction.cu) | Sum reduction | Simple tree, convergent tree, shared memory, segmented with `atomicAdd`, thread coarsening, warp shuffle |
| [`convolution.cu`](convolution.cu) | 2D convolution | Basic, filter in constant memory, tiled with halo cells |
| [`stencil.cu`](stencil.cu) | 3D 7-point stencil | Basic, shared memory tiles, thread coarsening in z, register tiling |
| [`histogram.cu`](histogram.cu) | Histogram | Global atomics, privatization in shared memory, contiguous and interleaved coarsening, aggregation |
| [`scan.cu`](scan.cu) | Inclusive scan (single block) | Kogge-Stone, double buffering, warp shuffle (scan-scan-add), thread coarsening, register tiling |

## Results

Measured on a NVIDIA Tesla T4. Each time is the median of 20 runs after one untimed warm-up run, measured with CUDA events around the kernel only, not the copies between host and device.

Bandwidth is the effective bandwidth: the minimum number of bytes the kernel has to read and write, divided by the time. For kernels that do a lot of arithmetic per byte, GFLOP/s is given instead.

### Vector addition

n = 2^24 + 3 floats, 201 MB moved.

| Kernel | Time | Bandwidth |
|---|---|---|
| basic                    | 0.772 ms | 261 GB/s |
| grid-stride (160 blocks) | 0.893 ms | 226 GB/s |
| `float4`                 | 0.807 ms | 250 GB/s |

Vector addition is memory-bound. The basic kernel already reaches 261 GB/s, 82 % of the T4's theoretical peak of 320 GB/s, so 16-byte `float4` accesses have nothing left to gain. The grid-stride loop with only 4 blocks per SM keeps fewer loads in flight and hides memory latency less well.

### Matrix multiplication

1000 × 1000 floats, 2 x 10^9 floating-point operations.

| Kernel | Time | Throughput |
|---|---|---|
| naive           | 4.433 ms | 451 GFLOP/s |
| tiled (16 × 16) | 2.460 ms | 813 GFLOP/s |

Tiling reduces global memory loads by the tile width and makes the kernel 1.8x faster.

### Reduction

n = 2^21 floats (8.4 MB). Stages 1-3 run in a single block and are only checked for correctness, not timed. The first three stages (simple tree, convergent tree, shared memory) run in a single block on 2048 elements and are only checked for correctness, not timed.

| Kernel | Time | Bandwidth |
|---|---|---|
| segmented + `atomicAdd` | 0.144 ms | 58 GB/s |
| thread coarsening       | 0.086 ms | 98 GB/s |
| warp shuffle            | 0.045 ms | 185 GB/s |

Coarsening lets each thread add elements in registers before the tree starts, and warp shuffles replace the last shared-memory steps and their barriers. Together they make the kernel 3.2x faster.

### Convolution

4000 × 3000 float image, 7 × 7 filter (`kFilterRadius = 3`), 96 MB moved, 1.18 x 10^9 floating-point operations.

| Kernel | Time | Throughput |
|---|---|---|
| basic           | 3.266 ms | 360 GFLOP/s |
| constant memory | 2.346 ms | 501 GFLOP/s |
| tiled           | 1.723 ms | 682 GFLOP/s |

Each input pixel is used 49 times, so the kernel does far more work per byte of DRAM traffic than a memory-bound kernel. Moving the filter to constant memory helps, because all threads of a warp read the same filter value, which the constant cache broadcasts in one access. Tiling the image in shared memory makes the kernel 1.9x faster than basic, even though only 26 × 26 of its 32 × 32 threads compute an output.

### 3D stencil

512^3 grid, 7-point stencil, 1.07 GB moved.

| Kernel | Time | Bandwidth |
|---|---|---|
| basic                  | 8.027 ms | 134 GB/s |
| shared memory tiles    | 11.452 ms | 94 GB/s |
| thread coarsening in z | 6.789 ms | 158 GB/s |
| register tiling        | 6.523 ms | 165 GB/s |

The shared-memory version is slower than the basic one. Its 8×8×8 input tile only has 6×6×6 inner points, so 58 % of the threads only load halo cells and never compute, while the basic kernel already gets most of its neighbour reads from the L1 cache. Coarsening in z fixes this with a 2D tile that moves through the grid, and register tiling, which keeps the z neighbours in registers, is the fastest stage at 1.23x the speed of the basic kernel.

### Histogram

n = 2^24 + 3 random lowercase letters and spaces, 7 bins of 4 letters each.

| Kernel | Time | Bandwidth |
|---|---|---|
| global atomics                | 7.987 ms | 2.1 GB/s |
| privatization (shared memory) | 0.679 ms | 24.7 GB/s |
| contiguous coarsening         | 0.606 ms | 27.7 GB/s |
| interleaved coarsening        | 0.373 ms | 44.9 GB/s |
| aggregation                   | 0.414 ms | 40.6 GB/s |

With only 7 bins, all threads compete for the same bins, so the basic kernel is limited by atomic contention. A private histogram per block in shared memory is 11.8x faster. Interleaved coarsening beats contiguous coarsening because neighbouring threads read neighbouring bytes, so loads are coalesced. Aggregation (counting runs of the same bin before one atomic) does not pay off on random text, where runs are short.

### Scan

All stages scan within a single block of 512 threads. Stages 1-3 handle one element per thread (n = 500), stages 4-5 handle eight (n = 4000).

| Kernel | n | Time | Bandwidth |
|---|---|---|---|
| Kogge-Stone                    | 500  | 0.007 ms | 0.54 GB/s |
| Kogge-Stone, double buffering  | 500  | 0.006 ms | 0.65 GB/s |
| warp shuffle (scan-scan-add)   | 500  | 0.005 ms | 0.74 GB/s |
| thread coarsening (three-phase) | 4000 | 0.010 ms | 3.13 GB/s |
| register tiling                | 4000 | 0.007 ms | 4.39 GB/s |

With a single block on one of 40 SMs and at most 16 KB of data, these times are dominated by the kernel launch and do not show the differences between the algorithms.

## Running

Requirements: an NVIDIA GPU and the CUDA toolkit (`nvcc`). Each file is a standalone program:

```
mkdir -p build
nvcc -arch=sm_75 -o build/reduction reduction.cu && ./build/reduction
```

`-arch=sm_75` matches the T4.

