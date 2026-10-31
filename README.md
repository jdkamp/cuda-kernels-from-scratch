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

## Results

Measured on a NVIDIA Tesla T4. Times cover the kernel only, not the copies between host and device.

Bandwidth is the effective bandwidth: the minimum number of bytes the kernel has to read and write, divided by the time.

### Vector addition

n = 2^24 + 3 floats, 201 MB moved.

| Kernel | Time | Bandwidth |
|---|---|---|
| basic                    | 0.922 ms | 218 GB/s |
| grid-stride (160 blocks) | 0.889 ms | 227 GB/s |
| `float4`                 | 0.808 ms | 249 GB/s |

Vector addition is memory-bound. With 16-byte loads and stores the kernel reaches 249 GB/s, 78 % of the T4's theoretical peak of 320 GB/s.

### Matrix multiplication

1000 × 1000 floats, 12 MB moved.

| Kernel | Time | Bandwidth |
|---|---|---|
| naive           | 4.318 ms | 2.8 GB/s |
| tiled (16 × 16) | 3.267 ms | 3.7 GB/s |

Tiling reduces global memory loads by the tile width and gives a 1.3x speedup. The effective bandwidth is low by design: every input value is used 1000 times, so the minimum traffic says little about how busy the GPU is.

### Reduction

n = 2^21 floats (8.4 MB). Stages 1-3 run in a single block and are only checked for correctness, not timed.

| Kernel | Time | Bandwidth |
|---|---|---|
| segmented + `atomicAdd` | 0.109 ms | 77 GB/s |
| thread coarsening       | 0.073 ms | 115 GB/s |
| warp shuffle            | 0.043 ms | 195 GB/s |

Coarsening lets each thread add elements in registers before the tree starts, and warp shuffles replace the last shared-memory steps and their barriers. Together they make the kernel 2.5x faster.

### 3D stencil

512^3 grid, 7-point stencil, 1.07 GB moved.

| Kernel | Time | Bandwidth |
|---|---|---|
| basic                  | 9.114 ms | 118 GB/s |
| shared memory tiles    | 19.198 ms | 56 GB/s |
| thread coarsening in z | 9.793 ms | 110 GB/s |
| register tiling        | 7.483 ms | 144 GB/s |

The shared-memory version is slower than the basic one. Its 8×8×8 input tile only has 6×6×6 inner points, so 58 % of the threads only load halo cells and never compute, while the basic kernel already gets most of its neighbour reads from the L1 cache. Coarsening in z fixes this with a 2D tile that moves through the grid, and register tiling, which keeps the z neighbours in registers, is the fastest stage.

### Histogram

n = 2^24 + 3 random lowercase letters and spaces, 7 bins of 4 letters each.

| Kernel | Time | Bandwidth |
|---|---|---|
| global atomics                | 8.173 ms | 2.1 GB/s |
| privatization (shared memory) | 0.855 ms | 19.6 GB/s |
| contiguous coarsening         | 0.774 ms | 21.7 GB/s |
| interleaved coarsening        | 0.459 ms | 36.6 GB/s |
| aggregation                   | 0.499 ms | 33.6 GB/s |

With only 7 bins, all threads compete for the same bins, so the basic kernel is limited by atomic contention. A private histogram per block in shared memory is 9.6x faster. Interleaved coarsening beats contiguous coarsening because neighbouring threads read neighbouring bytes, so loads are coalesced. Aggregation (counting runs of the same bin before one atomic) does not pay off on random text, where runs are short.

## Running

Requirements: an NVIDIA GPU and the CUDA toolkit (`nvcc`). Each file is a standalone program:

```
mkdir -p build
nvcc -arch=sm_75 -o build/reduction reduction.cu && ./build/reduction
```

`-arch=sm_75` matches the T4.

