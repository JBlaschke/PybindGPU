#ifndef DEVICE_WRAPPER_H
#define DEVICE_WRAPPER_H

#include <cuda_hip_wrapper.h>


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