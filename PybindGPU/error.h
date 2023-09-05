#ifndef ERROR_H
#define ERROR_H

#include <ptr_wrapper.h>
#include <cuda_hip_wrapper.h>
#include <pybind11/pybind11.h>


struct CudaError : public obj_wrapper<cudaError_t> {

    CudaError(int a_error) : obj_wrapper(static_cast<cudaError_t>(a_error)) {};

    int as_int() const;
};

struct NvmlReturn : public obj_wrapper<nvmlReturn_t> {

    NvmlReturn(int a_return) : obj_wrapper(static_cast<nvmlReturn_t>(a_return)) {};

    int as_int() const;
};

namespace py = pybind11;
void generate_cuda_error(py::module & m);

#endif