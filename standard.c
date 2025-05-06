#include "standard.h"
#include <omp.h>

void standard_multiply(const int * restrict A, const int * restrict B, int * restrict C, int n, int threadcount) {
    #pragma omp parallel for collapse(2) \
        num_threads(threadcount) \
        schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int sum = 0;       
            for (int k = 0; k < n; k++) {
                sum += A[i*n + k] * B[k*n + j];
            }
            C[i*n + j] = sum;
        }
    }
}