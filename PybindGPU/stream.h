#ifndef STREAM_H
#define STREAM_H

#include <ptr_wrapper.h>
#include <cuda_hip_wrapper.h>
#include <pybind11/pybind11.h>

#include <error.h>


namespace py = pybind11;

class CudaStream {
    public:
        CudaStream();
        CudaStream(unsigned int flags);
#ifndef USE_HIP
        CudaStream(unsigned int flags, int priority);
#endif
        ~CudaStream();

        cudaStream_t & operator* () { return stream; }
        cudaStream_t * get() { return & stream; }
        cudaError_t last_status() const { return status; }
    private:
        cudaStream_t stream;
        cudaError_t status;
};

namespace py = pybind11;
void generate_cuda_stream(py::module & m);

#endif