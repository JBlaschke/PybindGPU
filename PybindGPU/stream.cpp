#include <stream.h>


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


void generate_cuda_stream(py::module & m){
    py::class_<CudaStream>(m, "cudaStream_t")
        .def(py::init<>())
        .def(py::init<int>())
        .def("get",
            [](CudaStream & a) {
                return ptr_wrapper<cudaStream_t>(a.get());
            }
        )
        .def("last_status",
            [](const CudaStream & a) {
                return CudaError(a.last_status());
            }
        );
}