#include <device_wrapper.h>



CudaEvent::CudaEvent() {
    status = cudaEventCreate(& event);
}


CudaEvent::CudaEvent(unsigned int flags) {
    status = cudaEventCreateWithFlags(& event, flags);
}


CudaEvent::~CudaEvent() {
    status = cudaEventDestroy(event);
}


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