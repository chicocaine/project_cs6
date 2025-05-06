#ifndef STANDARD_H
#define STANDARD_H

// multiply C = A·B with the standard triple loop (O(n^3))
void standard_multiply(const int *A, const int *B, int *C, int n, int threadcount);

#endif // STANDARD_H