#include <event.h>

CudaEvent::CudaEvent() {
    status = cudaEventCreate(& event);
}


CudaEvent::CudaEvent(unsigned int flags) {
    status = cudaEventCreateWithFlags(& event, flags);
}


CudaEvent::~CudaEvent() {
    status = cudaEventDestroy(event);
}


void generate_cuda_event(py::module & m) {
    // This needs to be defined so that the ptr_wrapper has something to return
    py::class_<ptr_wrapper<cudaEvent_t>>(m, "_CUevent_st__ptr");

    py::class_<CudaEvent>(m, "cudaEvent_t")
        .def(py::init<>())
        .def(py::init<int>())
        .def("get",
            [](CudaEvent & a) {
                return ptr_wrapper<cudaEvent_t>(a.get());
            }
        )
        .def("last_status",
            [](const CudaEvent & a) {
                return CudaError(a.last_status());
            }
        );
}