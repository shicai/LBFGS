/***********************************************************************************************************************
* 版权信息: 版权所有(c) 2013-2018, 杭州海康威视数字技术股份有限公司, 保留所有权利
*
* 文件名称: generics.h
* 摘　　要: AutoEncoder算法的类型、参数等基本设定头文件
*
* 当前版本: 0.1.0
* 作　　者: 杨世才
* 日　　期: 2014/04/15
* 备　　注: 初始版本
***********************************************************************************************************************/

#ifndef _HIK_IPA_TPR_DAE_GENERICS_H_
#define _HIK_IPA_TPR_DAE_GENERICS_H_

#ifdef __cplusplus
//extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <Windows.h>
#include <time.h>

#ifndef HIK_REAL
#define HIK_REAL            32
#endif

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


#if HIK_REAL == 32
typedef     float           real;
#else
typedef     double          real;
#endif
typedef     unsigned int    size;
typedef     unsigned int    index;

// print error code
#ifndef PRINT_ERR
#define PRINT_ERR(err_code){\
    printf("Error ocurred: 0x%x in line %d of file <%s>.\n", (err_code), __LINE__, __FILE__);\
}
#endif  // PRINT_ERR

// check state, if true, return error code
#ifndef CHECK_ERR
#define CHECK_ERR(state, err_code){\
if (state)\
{\
    PRINT_ERR(DAE_ERR_BASE | err_code);\
    return (DAE_ERR_BASE | err_code);\
}\
}
#endif  // CHECK_ERR


// random number
typedef struct _HIK_RAND
{
    unsigned int mt[624];
    unsigned int mti;
} HIK_RAND;

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

#ifdef __cplusplus
//}
#endif
#endif // _HIK_IPA_TPR_DAE_GENERICS_H_