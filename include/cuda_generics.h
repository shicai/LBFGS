#ifndef _HIK_IPA_TPR_DAE_CUDAGENERICS_H_
#define _HIK_IPA_TPR_DAE_CUDAGENERICS_H_

#ifdef __cplusplus
//extern "C" {
#endif

#ifndef _HIK_IPA_TPR_DAE_GENERICS_H_
#include "generics.h"
#endif

// CUDA and CUBLAS functions
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "curand.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

// CUDA Helper functions
#include "helper_functions.h"
#include "helper_cuda.h" 

#define CUDA_CC             350
//#define CUDA_MALLOC         cudaMallocManaged
//#define CUDA_MALLOCM        cudaMallocManaged
//#define CUDA_SYNC           cudaDeviceSynchronize()

// set cuda block size
#define BLOCK_SIZE           512

// set cuda safe function calls
#ifdef __CUDA_RUNTIME_H__
#define CUDA_CALL(x) do {\
    CHECK_ERR(((x) != cudaSuccess), CUDA_ERR);\
} while (0)
#endif // __CUDA_RUNTIME_H__

#ifdef CUBLAS_API_H_
#define CUBLAS_CALL(x) do {\
    CHECK_ERR(((x) != CUBLAS_STATUS_SUCCESS), CUBLAS_ERR);\
} while (0)
#endif // CUBLAS_API_H_

#ifdef CURAND_H_
#define CURAND_CALL(x) do {\
    CHECK_ERR(((x) != CURAND_STATUS_SUCCESS), CURAND_ERR);\
} while (0)
#endif // CURAND_H_

// set cuda function alias
#ifdef __CUDA_RUNTIME_H__ | CUBLAS_API_H_ | CURAND_H_
#if HIK_REAL == 32
#define EXP             expf
#define LOG             logf
#define HYPERTAN        tanhf
#define SQRT            sqrtf
#define CUBLAS_GEMV     cublasSgemv
#define CUBLAS_COPY     cublasScopy
#define CUBLAS_DOT      cublasSdot
#define CUBLAS_GEMM     cublasSgemm
#define CUBLAS_GER      cublasSger
#define CUBLAS_NRM2     cublasSnrm2
#define CURANDU         curandGenerateUniform
#define CURANDN         curandGenerateNormal
#else
#define EXP             exp
#define LOG             log
#define HYPERTAN        tanh
#define SQRT            sqrt
#define CUBLAS_GEMV     cublasDgemv
#define CUBLAS_COPY     cublasDcopy
#define CUBLAS_DOT      cublasDdot
#define CUBLAS_GEMM     cublasDgemm
#define CUBLAS_GER      cublasDger
#define CUBLAS_NRM2     cublasDnrm2
#define CURANDU         curandGenerateUniformDouble
#define CURANDN         curandGenerateNormalDouble
#endif
#endif

extern int dev_id;
//static time_t rawtime;
//static struct tm *t;

#ifdef __cplusplus
//}
#endif
#endif // _HIK_IPA_TPR_DAE_CUDAGENERICS_H_