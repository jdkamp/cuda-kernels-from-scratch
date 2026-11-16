#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "cuda_utils.cuh"

constexpr int kThreadsPerBlock = 256;
constexpr int kElementsPerThread = 8;    // output elements per thread
constexpr int kElementsPerBlock = kThreadsPerBlock * kElementsPerThread;

// Merge sorted a[0..m) and sorted b[0..n) into c[0..m+n).
// On equal values the element from a comes first.
__host__ __device__ void merge_sequential(const int* a, int m, const int* b, int n, int* c) {
    int i = 0;  // index for a
    int j = 0;  // index for b
    int k = 0;  // index for c
    
    while ((i < m) && (j < n)) {
        if (a[i] <= b[j]) {     // <= keeps the merge stable
            c[k++] = a[i++];
        } else {
            c[k++] = b[j++];
        }
    }
    while (i < m) {     // copy the rest of a
        c[k++] = a[i++];
    }
    while (j < n) {     // copy the rest of b
        c[k++] = b[j++];
    }
}

// Return i such that c[0..k) is the merge of a[0..i) and b[0..k-1).
// Binary seach taht keeps the same tie-breaking as merge_sequentual.
__host__ __device__ int co_rank(int k, const int* a, int m, const int* b, int n) {
    int i = k < m ? k : m;              // start, as many from as possible: (min k, m)
    int j = k - i;
    int i_low = 0 > (k-n) ? 0 : k - n;  // smallest possible i: max(0, k - n)
    int j_low = 0 > (k-m) ? 0 : k - m;  // smallest possible j: max(0, k - m)

    while (true) {
        if (i > 0 && j < n && a[i-1] > b[j]) {
            int delta = (i - i_low + 1) / 2;
            j_low = j;
            i -= delta;
            j += delta;
        } else if (j > 0 && i < m && b[j-1] >= a[i]){
            int delta = (j - j_low + 1) / 2;
            i_low = i;
            i += delta;
            j -= delta;
        } else {
            return i;
        }
    }

}

// Stage 1: each thread merges kElementsPerThread output elements. It finds its
// parts of a and b with co_rank and merges them sequentially. a and b are read
// directly from global memory.
__global__ void merge_basic_kernel(const int* a, int m, const int* b, int n, int* c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;    // global thread index

    // This thread's part of c: [k_curr, k_next), not beyond m + n
    int k_curr = min(tid * kElementsPerThread, m + n);
    int k_next = min((tid+1) * kElementsPerThread, m + n);

    // Matching parts of a and b
    int i_curr = co_rank(k_curr, a, m, b, n);
    int i_next = co_rank(k_next, a, m, b, n);
    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;

    // Merge them sequentially into c, starting at k_curr
    merge_sequential(&a[i_curr], i_next-i_curr, &b[j_curr], j_next-j_curr, &c[k_curr]);
}

float merge_basic(const int* h_a, int m, const int* h_b, int n, int* h_c) {
    size_t bytes_a = m * sizeof(int);
    size_t bytes_b = n * sizeof(int);
    size_t bytes_c = bytes_a + bytes_b;
    int num_blocks = (m + n + kElementsPerBlock - 1) / kElementsPerBlock;
    int *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, bytes_a);
    cudaMalloc(&d_b, bytes_b);
    cudaMalloc(&d_c, bytes_c);

    cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, bytes_c, cudaMemcpyHostToDevice);

    float ms = time_kernel([&] {
        merge_basic_kernel<<<num_blocks, kThreadsPerBlock>>>(d_a, m, d_b, n, d_c);
    });

    cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return ms;
}

int main() {
    int m = 1000;
    int n = 1500;
    std::vector<int> h_a(m);
    std::vector<int> h_b(n);

    srand(0);   // reproducibility
    for (int i = 0; i < m; i++) {
        h_a[i] =  rand() % 10;
    }
    for (int i = 0; i < n; i++) {
        h_b[i] =  rand() % 10;
    }
    std::sort(h_a.begin(), h_a.end());
    std::sort(h_b.begin(), h_b.end());

    std::vector<int> h_expected(m+n);
    merge_sequential(h_a.data(), m, h_b.data(), n, h_expected.data());

    std::vector<int> h_reference(m + n);
    std::merge(h_a.begin(), h_a.end(), h_b.begin(), h_b.end(), h_reference.begin());
    assert(count_mismatches(h_expected, h_reference) == 0);
    printf("merged %d + %d elements\n", m, n);

    for (int k = 0; k <= m + n; k++) {
        int i = co_rank(k, h_a.data(), m, h_b.data(), n);
        int j = k - i;
        assert(i >= 0 && i <= m && j >= 0 && j <= n);
        assert(i == 0 || j == n || h_a[i - 1] <= h_b[j]);    // i not too large
        assert(j == 0 || i == m || h_b[j - 1] < h_a[i]);     // j not too large
    }
    printf("co_rank ok for all %d positions\n", m + n + 1);

    std::vector<int> h_c_basic(m + n, -1);
    float ms = merge_basic(h_a.data(), m, h_b.data(), n, h_c_basic.data());
    printf("basic:  %.3f ms\n", ms);
    assert(count_mismatches(h_c_basic, h_expected) == 0);
}
