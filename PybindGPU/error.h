#ifndef ERROR_H
#define ERROR_H

#include <ptr_wrapper.h>
#include <cuda_hip_wrapper.h>
#include <pybind11/pybind11.h>


struct CudaError : public obj_wrapper<cudaError_t> {

    CudaError(int a_error) : obj_wrapper(static_cast<cudaError_t>(a_error)) {};

    int as_int() const;
};

namespace py = pybind11;
void generate_cuda_error(py::module & m);

#endif