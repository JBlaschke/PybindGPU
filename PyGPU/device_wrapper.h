#ifndef DEVICE_WRAPPER_H
#define DEVICE_WRAPPER_H

#include <cuda_hip_wrapper.h>


class CudaEvent {
    public:
        CudaEvent();
        CudaEvent(unsigned int flags);
        ~CudaEvent();

        cudaEvent_t * get() { return & event; } ;
        cudaError_t last_status() const { return status; };
    private:
        cudaEvent_t event;
        cudaError_t status;
};


class CudaStream {
    public:
        CudaStream();
        CudaStream(unsigned int flags);
#ifndef USE_HIP
        CudaStream(unsigned int flags, int priority);
#endif
        ~CudaStream();

        cudaStream_t * get() { return & stream; } ;
        cudaError_t last_status() const { return status; };
    private:
        cudaStream_t stream;
        cudaError_t status;
};

#endif