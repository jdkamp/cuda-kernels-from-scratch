#ifndef MERGE_CUH
#define MERGE_CUH

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

// Returns i such that c[0..k) is the merge of a[0..i) and b[0..k-i).
// Binary search that keeps the same tie-breaking as merge_sequential.
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

#endif // MERGE _CUH