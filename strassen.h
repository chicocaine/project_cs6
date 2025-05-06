#ifndef STRASSEN_H
#define STRASSEN_H

// multiply C = A·B using Strassen’s algorithm (thresholded to standard for small n)
void strassen_rec(const int * restrict A, const int * restrict B, int * restrict C, int n, int threshold, int threadcount);

#endif // STRASSEN_H
