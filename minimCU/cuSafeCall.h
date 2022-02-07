#pragma once

#include <stdio.h>
#include <assert.h>

#include <cusparse.h>
#include <cusolverSp.h>
#include <cuda_runtime_api.h>

/********************/
/* CUDA ERROR CHECK */
/********************/
void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) { exit(code); }
    }
}

extern "C" void gpuErrchk(cudaError_t ans) { gpuAssert((ans), __FILE__, __LINE__); }

/**************************/
/* CUSOLVE ERROR CHECKING */
/**************************/
static const char* _cusolverGetErrorEnum(cusolverStatus_t error)
{
    switch (error)
    {
    case CUSOLVER_STATUS_SUCCESS:
        return "CUSOLVER_SUCCESS";

    case CUSOLVER_STATUS_NOT_INITIALIZED:
        return "CUSOLVER_STATUS_NOT_INITIALIZED";

    case CUSOLVER_STATUS_ALLOC_FAILED:
        return "CUSOLVER_STATUS_ALLOC_FAILED";

    case CUSOLVER_STATUS_INVALID_VALUE:
        return "CUSOLVER_STATUS_INVALID_VALUE";

    case CUSOLVER_STATUS_ARCH_MISMATCH:
        return "CUSOLVER_STATUS_ARCH_MISMATCH";

    case CUSOLVER_STATUS_EXECUTION_FAILED:
        return "CUSOLVER_STATUS_EXECUTION_FAILED";

    case CUSOLVER_STATUS_INTERNAL_ERROR:
        return "CUSOLVER_STATUS_INTERNAL_ERROR";

    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    }

    return "<unknown>";
}

inline void __cusolveSafeCall(cusolverStatus_t err, const char *file, const int line)
{
    if (CUSOLVER_STATUS_SUCCESS != err) {
        fprintf(stderr, "CUSOLVE error in file '%s', line %d, error: %s\nterminating!\n", file, line, _cusolverGetErrorEnum(err));
        assert(0);
    }
}

extern "C" void cusolveSafeCall(cusolverStatus_t err) { __cusolveSafeCall(err, __FILE__, __LINE__); }

/***************************/
/* CUSPARSE ERROR CHECKING */
/***************************/
static const char* _cusparseGetErrorEnum(cusparseStatus_t error)
{
    switch (error)
    {
    case CUSPARSE_STATUS_SUCCESS:
        return "CUSPARSE_STATUS_SUCCESS";

    case CUSPARSE_STATUS_NOT_INITIALIZED:
        return "CUSPARSE_STATUS_NOT_INITIALIZED";

    case CUSPARSE_STATUS_ALLOC_FAILED:
        return "CUSPARSE_STATUS_ALLOC_FAILED";

    case CUSPARSE_STATUS_INVALID_VALUE:
        return "CUSPARSE_STATUS_INVALID_VALUE";

    case CUSPARSE_STATUS_ARCH_MISMATCH:
        return "CUSPARSE_STATUS_ARCH_MISMATCH";

    case CUSPARSE_STATUS_MAPPING_ERROR:
        return "CUSPARSE_STATUS_MAPPING_ERROR";

    case CUSPARSE_STATUS_EXECUTION_FAILED:
        return "CUSPARSE_STATUS_EXECUTION_FAILED";

    case CUSPARSE_STATUS_INTERNAL_ERROR:
        return "CUSPARSE_STATUS_INTERNAL_ERROR";

    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";

    case CUSPARSE_STATUS_ZERO_PIVOT:
        return "CUSPARSE_STATUS_ZERO_PIVOT";
    }

    return "<unknown>";
}

inline void __cusparseSafeCall(cusparseStatus_t err, const char *file, const int line)
{
    if (CUSPARSE_STATUS_SUCCESS != err) {
        fprintf(stderr, "CUSPARSE error in file '%s', line %d, error %s\nterminating!", file, line, _cusparseGetErrorEnum(err));
        cudaDeviceReset(); assert(0);
    }
}

extern "C" void cusparseSafeCall(cusparseStatus_t err) { __cusparseSafeCall(err, __FILE__, __LINE__); }
