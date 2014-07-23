#ifndef _HIK_IPA_TPR_DAE_MATH_H_
#define _HIK_IPA_TPR_DAE_MATH_H_

#include "common.h"

// vector & matrix memory
void*   vecalloc(size_t size);      /* cudaMallocManaged and cudaMemset 0 */
void*   vecmalloc(size_t size);     /* cudaMallocManaged */
void*   vecxalloc(size_t size);     /* cudaMalloc */
void    vecfree(void *memblock);    /* cudaFree */

// vector & matrix copy
void    veccpy(real *y, const real *x, const int n);    /* cudaMemcpy */
void    veccpy2(real *y, const real *x, const int n);   /* cublas copy */
void    vecncpy(real *y, const real *x, const int n);   /* negative copy, y = -x */

// set values for vector & matrix
void    vecset(real *x, const real c, const int n);     /* x = c, n-by-1 vector */
void    identity(real *x, const int n);                 /* x = n-by-n identity matrix */
void    ones(real *x, const int n);                     /* x = 1.0, n-by-1 vector */
void    zeros(real *x, const int n);                    /* x = 0.0, n-by-1 vector */

// vector normalization
void    vec2norm(real *s, const real *x, const int n);      /* F-norm of matrix, using cublas norm */
void    vec2norm2(real *s, const real *x, const int n);     /* L2-norm of vector, using dot and sqrt*/    
void    vec2norminv(real *s, const real *x, const int n);   /* 1.0 divided by vec2norm  */

// vector manipulations
void    vecadd(real *z, const real *x, const real *y, const real a, const real b, const int n); /* z = a * x + b * y */
void    vecmul(real *z, const real *x, const real *y, const real a, const int n);               /* z = a * x .* y    */
void    vecadd(real *y, const real *x, const real c, const int n);              /* y = y + cx               */
void    vecmul(real *y, const real *x, const int n);                            /* y = x .* y               */
void    vecadd(real *z, const real *x, const real *y, const int n);             /* z = x + y                */
void    vecdiff(real *z, const real *x, const real *y, const int n);            /* z = x - y                */
void    vecinc(real *x, const real c, const int n);                             /* x = x + c                */
void    vecscale(real *x, const real c, const int n);                           /* x = cx                   */
void    vecproj(real *x, const real a, const real b, const int n);              /* x = (b - a) * x + a      */
void    vecdiv(real *z, const real *x, const real *y, const int n);             /* z = x ./ y               */
void    vecinv(real *x, const real c, const int n);                             /* x = c ./ x               */
void    vecdot(real *s, const real *x, const real *y, const int n);             /* s = x' * y               */
void    vecexp(real *x, const real a, const real b, const real c, const int n); /* x = a * exp(b * x + c)   */

// vector reductions
void    vecmax(real *s, const real *x, const int n);        /* s = max(x) */
void    vecmin(real *s, const real *x, const int n);        /* s = min(x) */
real    vecsum(real *s, const real *x, const int n);        /* s = sum(x) */
real    vecmean(real *s, const real *x, const int n);       /* s = mean(x) */
void    vecsum_s(real *s, const real *x, const int n);      /* efficient for vectors with length < blocksize */
void    vecmean_s(real *s, const real *x, const int n);     /* efficient for vectors with length < blocksize */

// random matrix and shuffle
int     cuda_randu(real *x, int n);
int     cuda_randn(real *x, int n, real mean, real stddev);
int     cuda_shuffle(real *X, int *perm, int m, int n);     /* extra memory needed for m+n values */

// matrix normalization
int     cuda_norm(const real *A, int num_data, real *result);
int     cuda_matsqsum(const real *A, int num_data, real *result);
real    cuda_matsqsum(const real *A, int num_data);

// row sum and mean computation, and repeat add one col to a matrix
int     cuda_rowsum(const real *X, real *result, real alpha, int m, int n); /* extra memory needed for m values */
int     cuda_repaddcol(real *A, const real *x, real alpha, int m, int n);
int     cuda_repaddrow(real *A, const real *x, real alpha, int m, int n);
int     cuda_repaddcol(real *A, const real *x, int m, int n);
int     cuda_repaddrow(real *A, const real *x, int m, int n);
int     cuda_getmaxid(real *probx, real *idx, int m, int n);

// matrix multiplication, column-major
int     cuda_matmul(const real *X, const real *Y, real *Z, real alpha, real beta, int m, int n, int k);
int     cuda_matmult(const real *X, const real *Y, real *Z, real alpha, real beta, int m, int n, int k);
int     cuda_matmulp(const real *X, const real *Y, real *Z, real alpha, real beta, int m, int n, int k);
int     cuda_matmulz(const real *X, const real *Y, real *Z, real alpha, real beta, int m, int n, int k);
int     cuda_matmul(const real *X, const real *Y, real *Z, int m, int n, int k);    /* Z = X  * Y   */
int     cuda_matmult(const real *X, const real *Y, real *Z, int m, int n, int k);   /* Z = X' * Y   */
int     cuda_matmulp(const real *X, const real *Y, real *Z, int m, int n, int k);   /* Z = X  * Y'  */
int     cuda_matmulz(const real *X, const real *Y, real *Z, int m, int n, int k);   /* Z = X' * Y'  */

int     nn_neuron(real *Z, const real *W, const real *X, const real *bias, int m, int n, int k);
int     nn_sigm(real *F, const real *X, int n);
int     nn_sigm_grad(real *G, const real *X, int n);
int     nn_tanh(real *F, const real *X, int n);
int     nn_tanh_grad(real *G, const real *X, int n);
int     nn_relu(real *F, const real *X, int n);
int     nn_relu_grad(real *G, const real *X, int n);
int     nn_softplus(real *F, const real *X, int n);
int     nn_softplus_grad(real *G, const real *X, int n);
int     nn_linear(real *F, const real *X, int n);
int     nn_linear_grad(real *G, const real *X, int n);
int     nn_active(real *F, const real *X, int n, ACT_FUN fun_type);
int     nn_grad(real *G, const real *X, int n, ACT_FUN fun_type);

int     nn_kldiv(real *kld, const real *rho, real sparsity, int n);
int     nn_kldiv_grad(real *grad, const real *rho, real sparsity, int n);

int     nn_softmax_fit(real *probx, real *labels, real *J_fit, const int m, const int n);
int     nn_softmax_err(real *probx, real *labels, int m, int n);

#endif //_HIK_IPA_TPR_DAE_MATH_H_