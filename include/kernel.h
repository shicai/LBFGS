#ifndef _HIK_IPA_TPR_DAE_KERNEL_H_
#define _HIK_IPA_TPR_DAE_KERNEL_H_

#ifndef _HIK_IPA_TPR_DAE_GENERICS_H_
#include "cuda_generics.h"
#endif

#ifndef _HIK_IPA_TPR_DAE_KERNEL_H_
#include "kernel.h"
#endif

#ifndef _HIK_IPA_TPR_DAE_UTILS_H_
#include "utils.h"
#endif


/***********************************************************************************************************************
* ������:   double����ԭ�Ӽӷ�
* �Ρ���:   address     - I/O   double����ָ��
* ������    val         - I     double����
* ����ֵ:               - O     ��
***********************************************************************************************************************/
__device__ void atomicAdd(double *address, double val);

/***********************************************************************************************************************
* ������:   float����ԭ�������
* �Ρ���:   address     - I/O   float����ָ��
* ������    val         - I     float����
* ����ֵ:               - O     ��
***********************************************************************************************************************/
__device__ void atomicMax(float *address, float val);

/***********************************************************************************************************************
* ������:   double����ԭ�������
* �Ρ���:   address     - I/O   double����ָ��
* ������    val         - I     double����
* ����ֵ:               - O     ��
***********************************************************************************************************************/
__device__ void atomicMax(double *address, double val);

/***********************************************************************************************************************
* ������:   float����ԭ������С
* �Ρ���:   address     - I/O   float����ָ��
* ������    val         - I     float����
* ����ֵ:               - O     ��
***********************************************************************************************************************/
__device__ void atomicMin(float *address, float val);

/***********************************************************************************************************************
* ������:   double����ԭ������С
* �Ρ���:   address     - I/O   double����ָ��
* ������    val         - I     double����
* ����ֵ:               - O     ��
***********************************************************************************************************************/
__device__ void atomicMin(double *address, double val);

/***********************************************************************************************************************
* ������:   ���÷���Խ�Ԫ��Ϊ1
* �Ρ���:   x           - I     ���ݷ���, n * n
* ������    n           - I     ����������
* ����ֵ:               - O     ��
***********************************************************************************************************************/
__global__ void cuda_identity(real *x, int n);

/***********************************************************************************************************************
* ������:   ��������Ϊĳ����ֵ
* �Ρ���:   x           - I     ����������, n * 1
* ������    n           - I     ����
* ����ֵ:               - O     ��
***********************************************************************************************************************/
__global__ void cuda_setval(real *x, real val, int n);

/***********************************************************************************************************************
* ������:   ��������Ϊ����һ��������ĳ������
* �Ρ���:   x           - I     ����������, n * 1
* ������    y           - I     ����������, n * 1
* ������    n           - I     ����
* ����ֵ:               - O     ��
***********************************************************************************************************************/
__global__ void cuda_setvec(real *x, const real *y, real alpha, int n);

/***********************************************************************************************************************
* ������:   ��������Ϊ������������һ��������˻���ĳ������
* �Ρ���:   x           - I     ����������, n * 1
* ������    y           - I     ����������, n * 1
* ������    n           - I     ����
* ����ֵ:               - O     ��
***********************************************************************************************************************/
__global__ void cuda_setprod(real *x, const real *y, real alpha, int n);

/***********************************************************************************************************************
* ������:   [a, b]���ȷֲ����������cuda_randu���ɵľ��ȷֲ����������ӳ�䵽[a, b]����
* �Ρ���:   x           - I     ����������, n * 1
* ������    n           - I     ����
* ������    a           - I     ���ȷֲ��±߽�
* ������    b           - I     ���ȷֲ��ϱ߽�
* ����ֵ:               - O     ��
***********************************************************************************************************************/
__global__ void cuda_randuab(real *x, int n, real a, real b);

/***********************************************************************************************************************
* ������:   Block�������ۼ�ֵ, sx = sum(data)
* �Ρ���:   data        - I     ��������
* ������    sx          - O     �ۼ�ֵ
* ������    n           - I     �����С
* ����ֵ:               - O     ��
***********************************************************************************************************************/
__global__ void cuda_blocksum(const real *data, real *sx, int n);

/***********************************************************************************************************************
* ������:   Block��������ֵ, mx = mean(x)
* �Ρ���:   data        - I     ��������
* ������    sx          - O     ��ֵ
* ������    n           - I     �����С
* ����ֵ:               - O     ��
***********************************************************************************************************************/
__global__ void cuda_blockmean(const real *data, real *sx, int n);

/***********************************************************************************************************************
* ������:   ������Сֵ, val = min(data)
* �Ρ���:   data        - I     ��������
* ������    val         - O     ��Сֵ
* ������    n           - I     �����С
* ����ֵ:               - O     ��
***********************************************************************************************************************/
__global__ void cuda_vecmin(const real *data, real *val, int n);

/***********************************************************************************************************************
* ������:   �������ֵ, val = max(data)
* �Ρ���:   data        - I     ��������
* ������    val         - O     ���ֵ
* ������    n           - I     �����С
* ����ֵ:               - O     ��
***********************************************************************************************************************/
__global__ void cuda_vecmax(const real *data, real *val, int n);

/***********************************************************************************************************************
* ������:   �����о�ֵ, mx = mean(X, 1)
* �Ρ���:   X           - I     ��������
* ������    mx          - O     ��ֵ��������
* ������    m           - I     ��������
* ������    n           - I     ��������
* ����ֵ:               - O     ��
***********************************************************************************************************************/
__global__ void cuda_matmeanc(real *X, real *mx, int m, int n);

/***********************************************************************************************************************
* ������:   �����о�ֵ, mx = mean(X, 2)
* �Ρ���:   X           - I     ��������
* ������    mx          - O     ��ֵ��������
* ������    m           - I     ��������
* ������    n           - I     ��������
* ����ֵ:               - O     ��
***********************************************************************************************************************/
__global__ void cuda_matmeanr(real *X, real *mx, int m, int n);

/***********************************************************************************************************************
* ������:   ����ÿ�����ֵ������
* �Ρ���:   X           - I     ��������
* ������    idx         - O     ��ֵ��������
* ������    m           - I     ��������
* ������    n           - I     ��������
* ����ֵ:               - O     ��
***********************************************************************************************************************/
__global__ void cuda_matmaxidc(real *X, real *idx, int m, int n);

/***********************************************************************************************************************
* ������:   ����ÿ��Ԫ������һ������ֵ, X = X + val
* �Ρ���:   X           - I     ��������
* ������    val         - I     �ۼ�ֵ
* ������    n           - I     ��������*��������
* ����ֵ:               - O     ��
***********************************************************************************************************************/
__global__ void cuda_matinc(real *X, real val, int n);

/***********************************************************************************************************************
* ������:   �������ÿ��Ԫ�ص�ָ����, X = alpha * exp(beta * X + gamma)
* �Ρ���:   X           - I     ��������
* ������    alpha       - I     ϵ��
* ������    beta        - I     ϵ��
* ������    gamma       - I     ϵ��
* ������    n           - I     ��������*��������
* ����ֵ:               - O     ��
***********************************************************************************************************************/
__global__ void cuda_matexp(real *X, real alpha, real beta, real gamma, int n);

/***********************************************************************************************************************
* ������:   �������ۼ�, sx = sum(X, 1)
* �Ρ���:   X           - I     ��������
* ������    sx          - O     �ۼ�ֵ��������
* ������    m           - I     ��������
* ������    n           - I     ��������
* ����ֵ:               - O     ��
***********************************************************************************************************************/
__global__ void cuda_matsumc(real *X, real *sx, int m, int n);

/***********************************************************************************************************************
* ������:   �������ۼ�, mx = sum(X, 2)
* �Ρ���:   X           - I     ��������
* ������    sx          - O     �ۼ�ֵ��������
* ������    m           - I     ��������
* ������    n           - I     ��������
* ����ֵ:               - O     ��
***********************************************************************************************************************/
__global__ void cuda_matsumr(real *X, real *sx, int m, int n);

/***********************************************************************************************************************
* ������:   ������ƽ�������ۼ�, sx = sum(X.^2, 1)
* �Ρ���:   X           - I     ��������
* ������    sx          - O     �ۼ�ֵ��������
* ������    m           - I     ��������
* ������    n           - I     ��������
* ����ֵ:               - O     ��
***********************************************************************************************************************/
__global__ void cuda_matsqsumc(real *X, real *sx, int m, int n);

/***********************************************************************************************************************
* ������:   CUBLAS������Ȩ���, z = alpha * x + beta * y
* �Ρ���:   x           - I     ��������, n * 1
* ������    y           - I     ��������, n * 1
* ������    z           - I     �������, n * 1
* ������    alpha       - I     x����ϵ��
* ������    beta        - I     y����ϵ��
* ������    n           - I     ����ά��
* ����ֵ:               - O     ��
***********************************************************************************************************************/
__global__ void cuda_vecadd(const real *x, const real *y, real *z, real alpha, real beta, int n);

/***********************************************************************************************************************
* ������:   CUBLAS����ÿ�м�ƫ������, A = A + repmat(x, 1, n) = A + x * ones(1, n)
* �Ρ���:   A           - I     ���ݾ���, m * n
* ������    x           - I     ƫ����������m * 1
* ������    m           - I     X��������
* ������    n           - I     X��������
* ����ֵ:               - O     ��
***********************************************************************************************************************/
__global__ void cuda_reppluscol(real *A, const real *x, real alpha, int m, int n);
__global__ void cuda_repplusrow(real *A, const real *x, real alpha, int m, int n);

/***********************************************************************************************************************
* ������:   �������, Z = X - Y
* �Ρ���:   X           - I     ��������
* ������    Y           - I     ��������
* ������    Z           - O     �������
* ������    n           - I     ��������*��������
* ����ֵ:               - O     ��
***********************************************************************************************************************/
__global__ void cuda_matminus(real *X, real *Y, real *Z, int n);

/***********************************************************************************************************************
* ������:   ����Ԫ�س˷�(Hadamard/Entrywise Product), Z = X .* Y
* �Ρ���:   X           - I     ��������
* ������    Y           - I     ��������
* ������    Z           - I     �������
* ������    m           - I     ��������
* ������    n           - I     ��������
* ����ֵ:               - O     ��
***********************************************************************************************************************/
__global__ void cuda_matprod(real *X, real *Y, real *Z, int n);

/***********************************************************************************************************************
* ������:   CUDA KL Divergence����
* �Ρ���:   r           - I     ����
* ������    rh          - I     ����
* ������    rho         - I     rh��ɵ�����
* ������    grad        - I     KL Gradient
* ������    n           - I     ����ά��
* ����ֵ:               - O     ��
***********************************************************************************************************************/
__device__ real kl_div(real r, real rh);
__global__ void cuda_kldiv(real *rho, real r, int n);
__global__ void cuda_klgrad(real *rho, real *grad, real r, int n);

/***********************************************************************************************************************
* ������:   CUDA SIGMOID����
* �Ρ���:   x           - I     ����������, n * 1
* ������    y           - I     ����������, n * 1
* ������    n           - I     ����
* ����ֵ:               - O     ��
***********************************************************************************************************************/
__device__ real sigmoid(real x);
__device__ real sigmoid_grad(real x);
__global__ void cuda_sigmoid(real *x, real *y, int n);
__global__ void cuda_sigmoid_grad(real *x, real *y, int n);

/***********************************************************************************************************************
* ������:   CUDA TANH����
* �Ρ���:   x           - I     ����������, n * 1
* ������    y           - I     ����������, n * 1
* ������    n           - I     ����
* ����ֵ:               - O    ��
***********************************************************************************************************************/
__device__ real tanh_grad(real x);
__global__ void cuda_tanh(real *x, real *y, int n);
__global__ void cuda_tanh_grad(real *x, real *y, int n);

/***********************************************************************************************************************
* ������:   CUDA SOFTPLUS����
* �Ρ���:   x           - I     ����������, n * 1
* ������    y           - I     ����������, n * 1
* ������    n           - I     ����
* ����ֵ:               - O     ��
***********************************************************************************************************************/
__device__ real softplus(real x);
__global__ void cuda_softplus(real *x, real *y, int n);
__global__ void cuda_softplus_grad(real *x, real *y, int n);

/***********************************************************************************************************************
* ������:   CUDA LINEAR����
* �Ρ���:   x           - I     ����������, n * 1
* ������    y           - I     ����������, n * 1
* ������    n           - I     ����
* ����ֵ:               - O     ��
***********************************************************************************************************************/
__device__ real linear_act(real x);
__device__ real linear_grad(real x);
__global__ void cuda_linear(real *x, real *y, int n);
__global__ void cuda_linear_grad(real *x, real *y, int n);

/***********************************************************************************************************************
* ������:   CUDA RELU����
* �Ρ���:   x           - I     ����������, n * 1
* ������    y           - I     ����������, n * 1
* ������    n           - I     ����
* ����ֵ:               - O     ��
***********************************************************************************************************************/
__device__ real relu(real x);
__device__ real relu_grad(real x);
__global__ void cuda_relu(real *x, real *y, int n);
__global__ void cuda_relu_grad(real *x, real *y, int n);

/***********************************************************************************************************************
* ������:   ����sotmax����������
* �Ρ���:   probx       - I     ���ݾ���, m x n
* ������    labels      - I     ���ݱ�ǩ, 1 x n
* ������    J_fit       - I     ���, ����ֵ
* ������    m           - I     ����
* ������    n           - I     ����
* ����ֵ:               - O     ��
***********************************************************************************************************************/
__global__ void cuda_softmax_fit(real *probx, real *labels, real *J_fit, int m, int n);

/***********************************************************************************************************************
* ������:   ����sotmax��ǩ�������
* �Ρ���:   probx       - I     ���ʾ���, m x n
* ������    labels      - I     ���ݱ�ǩ, 1 x n
* ������    m           - I     ����
* ������    n           - I     ����
* ����ֵ:               - O     ��
***********************************************************************************************************************/
__global__ void cuda_softmax_err(real *probx, real *labels, int m, int n);

/***********************************************************************************************************************
* ������:   CUDA��sum reduction
* �Ρ���:   size        - I     ����ά��
* ������    threads     - I     Thread����
* ������    blocks      - I     Block����
* ������    data        - I     ����
* ������    sum_data    - O     ����partial sumֵ
* ����ֵ:               - O     ��
***********************************************************************************************************************/
void reduce(int size, int threads, int blocks, const real *data, real *sum_data);

#endif // _HIK_IPA_TPR_DAE_KERNEL_H_