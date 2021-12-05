#include <device_wrapper.h>


CudaStream::CudaStream() {
    status = cudaStreamCreate(& stream);
}


CudaStream::CudaStream(unsigned int flags) {
    status = cudaStreamCreateWithFlags(& stream, flags);
}


#ifndef USE_HIP
CudaStream::CudaStream(unsigned int flags, int priority) {
    status = cudaStreamCreateWithPriority(& stream, flags, priority);
}
#endif


CudaStream::~CudaStream() {
    status = cudaStreamDestroy(stream);
}