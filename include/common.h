#ifndef _HIK_IPA_TPR_DAE_COMMON_H_
#define _HIK_IPA_TPR_DAE_COMMON_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <Windows.h>
#include <time.h>

#ifndef HIK_REAL
#define HIK_REAL            32
#endif

#if HIK_REAL == 32
typedef     float           real;
#else
typedef     double          real;
#endif
typedef     unsigned int    size;
typedef     unsigned int    index;

#ifndef TRUE
#define TRUE                1
#define FALSE               0
#endif

#ifndef HIK_DEBUG_MODE
#define  HIK_DEBUG_MODE     0
#define  HIK_SHOW_MSG       0
#define  GRAD_CHECK         0
#define  HIK_VERBOSE        0 //(HIK_DEBUG_MODE | HIK_SHOW_MSG)
#endif

#define HIK_E               2.718281828459045
#define HIK_LOG_OF_2        0.693147180559945
#define HIK_PI              3.141592653589793
#define HIK_EPS_FLT         1.19209290E-07F
#define HIK_EPS_DBL         2.220446049250313E-16
#define HIK_DIFF_EPS        1.0E-7F

#define HIK_MAX_INT         0x7FFFFFFFL
#define HIK_MIN_INT         (- HIK_MAX_INT - 1)
#define HIK_MAX_UINT        0xFFFFFFFFUL

// some macro functions
#define HIK_STRINGIFY(x)    # x
#define HIK_XSTRINGIFY(x)   HIK_STRINGIFY(x)
#define HIK_CAT(x,y)        x ## y
#define HIK_XCAT(x,y)       HIK_CAT(x, y)
#define HIK_XCAT3(x,y,z)    HIK_XCAT(HIK_XCAT(x,y),z)
#define HIK_XCAT4(x,y,z,u)  HIK_XCAT(HIK_XCAT3(x,y,z),u)
#define HIK_MIN(x,y)        (((x)<(y))?(x):(y))
#define HIK_MAX(x,y)        (((x)>(y))?(x):(y))
#define HIK_SHIFT_LEFT(x,n) (((n)>=0)?((x)<<(n)):((x)>>-(n)))   // signed left shift operation*/

// time calculation
typedef struct _HIK_TIME_STATE
{
    LARGE_INTEGER start;
    LARGE_INTEGER freq;
} HIK_TIME_STATE;

static HIK_TIME_STATE time_state;
static void   hik_tic();
static double hik_toc();

static void hik_tic()
{
    HIK_TIME_STATE *state;
    memset(&time_state, 0, sizeof(HIK_TIME_STATE));
    state = &time_state;

    QueryPerformanceFrequency(&state->freq);
    QueryPerformanceCounter(&state->start);
}

static double hik_toc()
{
    HIK_TIME_STATE *state;
    state = &time_state;

    LARGE_INTEGER stop;
    QueryPerformanceCounter(&stop);
    return (double) (stop.QuadPart - state->start.QuadPart) / state->freq.QuadPart;
}

#define TIC (hik_tic())
#define TOC (hik_toc())

static void get_timestr(char t_str[16])
{
    struct tm   tim;
    time_t      now;

    now = time(NULL);
    tim = *(localtime(&now));
    strftime(t_str, 16, "%Y%m%dT%H%M%S", &tim);
    return;
}


#ifdef __cplusplus
}
#endif

// CUDA and CUBLAS functions
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "curand.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

// CUDA Helper functions
#include "helper_functions.h"
#include "helper_cuda.h" 


// set cuda block size
#define BLOCK_SIZE          512

#define CUDA_CC             350
#define CUDA_MALLOC         cudaMallocManaged
#define CUDA_XALLOC         cudaMalloc
#define CUDA_SYNC           cudaDeviceSynchronize()

#define FORCE_SYNC          1

typedef enum _ACT_FUN
{
    NONE        = 0,
    SIGMOID     = 1,
    TANH        = 2,
    RELU        = 3,
    SOFTPLUS    = 4,
    LINEAR      = 5,
} ACT_FUN;

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

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(tid, n) \
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; \
    tid < (n); \
    tid += blockDim.x * gridDim.x)

// status or error code
enum
{
    DAE_OK                  = 0x00000000,
    DAE_SUCCESS             = 0x00000000,

    DAE_ERR_BASE            = 0x82040000,

    MEM_ALLOC_ERR           = 0x1000,

    GPU_MEM_ALL_ERR         = 0x1100,
    CUDA_ERR,
    GPU_MEM_CPY_ERR,
    GPU_CALC_ERR,
    GPU_NO_MEM,

    CUBLAS_HND_ERR          = 0x1140,
    CUBLAS_ERR,

    CURAND_ERR              = 0x1181,

    FILE_OPEN_ERROR         = 0x1200,
    FILE_SIZE_ERROR,
    FILE_SEEK_ERROR,
    MAGIC_NUM_ERROR,

    DATA_READ_ERROR         = 0x1300,
    DATA_WRITE_ERROR,
    DATA_TYPE_ERROR,
    DATA_NULL_ERROR,
    DATA_BOUNDS_ERROR,
    DATA_SIZE_ERROR,

    LOAD_CFG_ERR            = 0x1400,
    CMD_CFG_ERR,
    ACT_FUN_ERR,

    COST_FUN_ERR            = 0x1500,
    GRAD_CHK_ERR,
};

static const char* cublasGetErrorString(cublasStatus_t error) {
    switch (error) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
    }
    return "Unknown cublas status";
}

static const char* curandGetErrorString(curandStatus_t error) {
    switch (error) {
    case CURAND_STATUS_SUCCESS:
        return "CURAND_STATUS_SUCCESS";
    case CURAND_STATUS_VERSION_MISMATCH:
        return "CURAND_STATUS_VERSION_MISMATCH";
    case CURAND_STATUS_NOT_INITIALIZED:
        return "CURAND_STATUS_NOT_INITIALIZED";
    case CURAND_STATUS_ALLOCATION_FAILED:
        return "CURAND_STATUS_ALLOCATION_FAILED";
    case CURAND_STATUS_TYPE_ERROR:
        return "CURAND_STATUS_TYPE_ERROR";
    case CURAND_STATUS_OUT_OF_RANGE:
        return "CURAND_STATUS_OUT_OF_RANGE";
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
        return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
        return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
    case CURAND_STATUS_LAUNCH_FAILURE:
        return "CURAND_STATUS_LAUNCH_FAILURE";
    case CURAND_STATUS_PREEXISTING_FAILURE:
        return "CURAND_STATUS_PREEXISTING_FAILURE";
    case CURAND_STATUS_INITIALIZATION_FAILED:
        return "CURAND_STATUS_INITIALIZATION_FAILED";
    case CURAND_STATUS_ARCH_MISMATCH:
        return "CURAND_STATUS_ARCH_MISMATCH";
    case CURAND_STATUS_INTERNAL_ERROR:
        return "CURAND_STATUS_INTERNAL_ERROR";
    }
    return "Unknown curand status";
}

// print error code
#ifndef PRINT_ERR
#define PRINT_ERR(err_code, err_info){\
    printf("Error ocurred in Line %d of File [%s].\n", __LINE__, __FILE__);\
    printf("Code: 0x%08x, Info: %s.\n", (err_code), (err_info));\
}
#endif  // PRINT_ERR

// check state, if true, return error code
#ifndef CHECK_ERR
#define CHECK_ERR(state, err_code, err_info){\
    if (state)\
{\
    PRINT_ERR(DAE_ERR_BASE | err_code, err_info);\
    return (DAE_ERR_BASE | err_code);\
}\
}
#endif  // CHECK_ERR

// set cuda safe function calls
#ifdef __CUDA_RUNTIME_H__
// code block avoids redefinition of cudaError_t error
#define CUDA_CALL(x) do {\
    cudaError_t error = x; \
    CHECK_ERR((error != cudaSuccess), CUDA_ERR, cudaGetErrorString(error));\
} while (0)
#endif // __CUDA_RUNTIME_H__

#ifdef CUBLAS_API_H_
#define CUBLAS_CALL(x) do {\
    cublasStatus_t status = x; \
    CHECK_ERR((status != CUBLAS_STATUS_SUCCESS), CUBLAS_ERR, cublasGetErrorString(status));\
} while (0)
#endif // CUBLAS_API_H_

#ifdef CURAND_H_
#define CURAND_CALL(x) do {\
    curandStatus_t status = x; \
    CHECK_ERR((status != CURAND_STATUS_SUCCESS), CURAND_ERR, curandGetErrorString(status));\
} while (0)
#endif // CURAND_H_


#endif // _HIK_IPA_TPR_DAE_COMMON_H_