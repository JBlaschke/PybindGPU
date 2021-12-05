#ifndef CUDA_HIP_WRAPPER_H
#define CUDA_HIP_WRAPPER_H

#ifdef USE_HIP

#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <hipfft.h>

// cuda.h adapters
#define cudaDeviceReset hipDeviceReset
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaError_t hipError_t
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventDestroy hipEventDestroy
#define cudaEvent_t hipEvent_t
#define cudaFree hipFree
#define cudaFreeHost hipHostFree // hipFreeHost is deprecated
#define cudaGetDevice hipGetDevice
#define cudaGetErrorName hipGetErrorName
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaMalloc hipMalloc
#define cudaMallocHost hipHostMalloc // hipMallocHost is deprecated
#define cudaMemcpy hipMemcpy
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemset hipMemset
#define cudaSetDevice hipSetDevice
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess

// cuComplex.h adapters
#define cuDoubleComplex hipDoubleComplex
#define cuFloatComplex hipFloatComplex

#else

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>

#endif

#endif