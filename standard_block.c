#include "standard_block.h"
#include <omp.h>

void blocked_multiply(const int * restrict A, const int * restrict B, int * restrict C, int n, int BS, int threadcount) {
    #pragma omp parallel for collapse(2) \
        num_threads(threadcount) \
        schedule(static)
    for (int ii = 0; ii < n; ii += BS) {
        for (int jj = 0; jj < n; jj += BS) {
            for (int kk = 0; kk < n; kk += BS) {
                for (int i = ii; i < ii + BS && i < n; i++) {
                    for (int j = jj; j < jj + BS && j < n; j++) {
                        int sum = 0;
                        for (int k = kk; k < kk + BS && k < n; k++) {
                            sum += A[i*n + k] * B[k*n + j];
                        }
                        C[i*n + j] += sum;
                    }
                }
            }
        }
    }
}