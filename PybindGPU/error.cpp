#include <error.h>


int CudaError::as_int() const {
    return static_cast<int>(_obj);
}


int NvmlReturn::as_int() const {
    return static_cast<int>(_obj);
}


void generate_cuda_error(py::module & m){

    py::class_<CudaError>(m, "cudaError_t")
        .def(py::init<int>())
        .def("as_int", & CudaError::as_int)
        .def("__repr__",
            [](const CudaError & a) {
                return "<CudaError: 'code=" + std::to_string(a.as_int()) + "'>";
            }
        );

    py::class_<NvmlReturn>(m, "nvmlReturn_t")
        .def(py::init<int>())
        .def("as_int", & NvmlReturn::as_int)
        .def("__repr__",
            [](const NvmlReturn & a) {
                return "<NvmlReturn: 'code=" + std::to_string(a.as_int()) + "'>";
            }
        );
}
