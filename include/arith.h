
#ifndef _HIK_IPA_TPR_DAE_ARITH_H_
#define _HIK_IPA_TPR_DAE_ARITH_H_

#include <memory.h>
#include "generics.h"
#include "cuda_generics.h"
#include "kernel.h"

typedef unsigned int uint32_t;

#if LBFGS_FLOAT == 32 && LBFGS_IEEE_FLOAT
#define fsigndiff(x, y) (((*(uint32_t*)(x)) ^ (*(uint32_t*)(y))) & 0x80000000U)
#else
#define fsigndiff(x, y) (*(x) * (*(y) / fabs(*(y))) < 0.)
#endif/*LBFGS_IEEE_FLOAT*/

void* vecalloc(size_t size);
void* vecmalloc(size_t size);
void* valloc(size_t size);
void* vmalloc(size_t size);
void veccpy(real *y, const real *x, const int n);
void veccpy2(real *y, const real *x, const int n);

void identity(real *x, const int n);
void ones(real *x, const int n);
void vec2norm(real *s, const real *x, const int n);
void vec2norminv(real *s, const real *x, const int n);
void vecadd(real *y, const real *x, const real c, const int n);
void vecdiff(real *z, const real *x, const real *y, const int n);
void vecdot(real *s, const real *x, const real *y, const int n);
void vecexp(real *x, const real a, const real b, const real c, const int n);
void vecfree(void *memblock);
void vecinc(real *x, const real c, const int n);
void vecmax(real *s, const real *x, const int n);
void vecmean_small(real *s, const real *x, const int n);
void vecmin(real *s, const real *x, const int n);
void vecmul(real *y, const real *x, const int n);
void vecncpy(real *y, const real *x, const int n);
void vecscale(real *y, const real c, const int n);
void vecset(real *x, const real c, const int n);
void vecsum_small(real *s, const real *x, const int n);
void zeros(real *x, const int n);

real vecmean(real *s, const real *x, const int n);
real vecsum(real *s, const real *x, const int n);

#endif // _HIK_IPA_TPR_DAE_ARITH_H_