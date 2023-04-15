#ifndef EVENT_H
#define EVENT_H

#include <ptr_wrapper.h>
#include <cuda_hip_wrapper.h>
#include <pybind11/pybind11.h>

#include <error.h>


class CudaEvent {
    public:
        CudaEvent();
        CudaEvent(unsigned int flags);
        ~CudaEvent();

        cudaEvent_t & operator* () { return event; }
        cudaEvent_t * get() { return & event; }
        cudaError_t last_status() const { return status; }
    private:
        cudaEvent_t event;
        cudaError_t status;
};

namespace py = pybind11;
void generate_cuda_event(py::module & m);

#endif