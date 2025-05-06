#include "strassen.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

static void add_matrix(const int* a, const int* b, int* c, int n) {
    for (int i = 0, size = n*n; i < size; i++) c[i] = a[i] + b[i];
}
static void sub_matrix(const int* a, const int* b, int* c, int n) {
    for (int i = 0, size = n*n; i < size; i++) c[i] = a[i] - b[i];
}
static void multiply_matrices(const int* A, const int* B, int* C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            int sum = 0;
            for (int k = 0; k < n; k++)
                sum += A[i*n + k] * B[k*n + j];
            C[i*n + j] = sum;
        }
}
static int* allocate_matrix(int n) {
    int* m = malloc((size_t)n*n*sizeof *m);
    if (!m) { perror("malloc"); exit(EXIT_FAILURE); }
    return m;
}


// --- Recursive Strassen ---
void strassen_rec(const int * restrict a, const int * restrict b, int * restrict c, int n, int threshold, int threadcount) {
    // set threshold for Strassen's algorithm
    if (n <= threshold) {
        multiply_matrices(a, b, c, n);
        return;
    }
    // base case
    if (n == 1) {
        c[0] = a[0] * b[0];
        return;
    }
    int m = n/2, sz = m*m;
    int *a11=allocate_matrix(m), *a12=allocate_matrix(m),
        *a21=allocate_matrix(m), *a22=allocate_matrix(m),
        *b11=allocate_matrix(m), *b12=allocate_matrix(m),
        *b21=allocate_matrix(m), *b22=allocate_matrix(m);
    // split A,B into quadrants …
    for(int i=0;i<m;i++)for(int j=0;j<m;j++){
        int idx = i*n + j, idx2 = i*m + j;
        a11[idx2]=a[idx];               a12[idx2]=a[idx+m];
        a21[idx2]=a[idx+n*m];           a22[idx2]=a[idx+n*m+m];
        b11[idx2]=b[idx];               b12[idx2]=b[idx+m];
        b21[idx2]=b[idx+n*m];           b22[idx2]=b[idx+n*m+m];
    }
    // allocate temporaries
    int *m1=allocate_matrix(m),*m2=allocate_matrix(m),*m3=allocate_matrix(m),
        *m4=allocate_matrix(m),*m5=allocate_matrix(m),*m6=allocate_matrix(m),
        *m7=allocate_matrix(m);
    // M1 = (A11+A22)*(B11+B22)
    #pragma omp task firstprivate(a11, a22, b11, b22) shared(m1)
    {
        int *t1_loc=allocate_matrix(m); 
        int *t2_loc=allocate_matrix(m);
        add_matrix(a11,a22,t1_loc,m); 
        add_matrix(b11,b22,t2_loc,m);
        strassen_rec(t1_loc,t2_loc,m1,m,threshold,threadcount);
        free(t1_loc); free(t2_loc);
    }
    // M2 = (A21+A22)*B11
    #pragma omp task firstprivate(a21,a22,b11) shared(m2)
    {
        int *t1_loc=allocate_matrix(m);
        add_matrix(a21,a22,t1_loc,m);
        strassen_rec(t1_loc,b11,m2,m,threshold,threadcount);
        free(t1_loc);
    }
    // M3 = A11*(B12−B22)
    #pragma omp task firstprivate(a11,b12,b22) shared(m3)
    {
        int *t2_loc=allocate_matrix(m);
        sub_matrix(b12,b22,t2_loc,m);
        strassen_rec(a11,t2_loc,m3,m,threshold,threadcount);
        free(t2_loc);
    }
    // M4 = A22*(B21−B11)
    #pragma omp task firstprivate(a22,b21,b11) shared(m4)
    {
        int *t2_loc=allocate_matrix(m);
        sub_matrix(b21,b11,t2_loc,m);
        strassen_rec(a22,t2_loc,m4,m,threshold,threadcount);
        free(t2_loc);
    }
    // M5 = (A11+A12)*B22
    #pragma omp task firstprivate(a11,a12,b22) shared(m5)
    {
        int *t1_loc=allocate_matrix(m);
        add_matrix(a11,a12,t1_loc,m);
        strassen_rec(t1_loc,b22,m5,m,threshold,threadcount);
        free(t1_loc);
    }
    // M6 = (A21−A11)*(B11+B12)
    #pragma omp task firstprivate(a21,a11,b11,b12) shared(m6)
    {
        int *t1_loc=allocate_matrix(m);
        int  *t2_loc=allocate_matrix(m);
        sub_matrix(a21,a11,t1_loc,m);
        add_matrix(b11,b12,t2_loc,m);
        strassen_rec(t1_loc,t2_loc,m6,m,threshold,threadcount);
        free(t1_loc); free(t2_loc);
    }
    // M7 = (A12−A22)*(B21+B22)
    #pragma omp task firstprivate(a12,a22,b21,b22) shared(m7)
    {
        int *t1_loc=allocate_matrix(m);
        int *t2_loc=allocate_matrix(m);
        sub_matrix(a12,a22,t1_loc,m);
        add_matrix(b21,b22,t2_loc,m);
        strassen_rec(t1_loc,t2_loc,m7,m,threshold,threadcount);
        free(t1_loc); free(t2_loc);
    }

    #pragma omp taskwait

    // combine results into C quadrants
    int *c11=allocate_matrix(m),*c12=allocate_matrix(m),
        *c21=allocate_matrix(m),*c22=allocate_matrix(m);
    for(int i=0;i<sz;i++){
        c11[i]=m1[i]+m4[i]-m5[i]+m7[i];
        c12[i]=m3[i]+m5[i];
        c21[i]=m2[i]+m4[i];
        c22[i]=m1[i]-m2[i]+m3[i]+m6[i];
    }
    for(int i=0;i<m;i++)for(int j=0;j<m;j++){
        int idx=i*n+j, idx2=i*m+j;
        c[idx]=c11[idx2]; c[idx+m]=c12[idx2];
        c[idx+n*m]=c21[idx2]; c[idx+n*m+m]=c22[idx2];
    }
    // free all temporaries
    free(a11); free(a12); free(a21); free(a22);
    free(b11); free(b12); free(b21); free(b22);
    free(m1); free(m2); free(m3); free(m4);
    free(m5); free(m6); free(m7);
    free(c11); free(c12); free(c21); free(c22);
}