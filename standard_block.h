#ifndef STANDARD_BLOCK_H
#define STANDARD_BLOCK_H

// multiply C = A·B with the standard triple loop (O(n^3))
void blocked_multiply(const int *A, const int *B, int *C, int n, int BS, int threadcount);

#endif // STANDARD_H