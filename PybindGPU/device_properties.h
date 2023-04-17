#ifndef DEVICE_PPROPERTIES_H
#define DEVICE_PPROPERTIES_H

#include <string>
#include <cuda_hip_wrapper.h>
#include <pybind11/pybind11.h>


std::string mem_to_string(const void * address, size_t size);


class DeviceProperties {
    public:
        DeviceProperties(int i);
        ~DeviceProperties();

        cudaDeviceProp & operator* () { return prop; }
        cudaDeviceProp * get() { return & prop; }
        cudaError_t last_status() const { return status; }
    private:
        cudaDeviceProp prop;
        cudaError_t status;
};


namespace py = pybind11;
void generate_device_properties(py::module & m);


#endif