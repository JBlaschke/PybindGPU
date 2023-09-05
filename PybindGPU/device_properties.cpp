#include <cstdint>  // used by intptr_t
#include <stdio.h>

#include <ptr_wrapper.h>
#include <cuda_hip_wrapper.h>

#include <error.h>
#include <device_properties.h>


std::string mem_to_string(const void * address, size_t size) {
    const unsigned char * p = (const unsigned char *) address;
    char buffer[size];
    for (size_t i = 0; i < size; i++) {
        snprintf(buffer + i, sizeof(buffer), "%02hhx", p[i]);
    }

    std::string s(buffer);
    return s;
}


DeviceProperties::DeviceProperties(int i) {
    cudaGetDeviceProperties(& prop, i);
}


DeviceProperties::~DeviceProperties() {}


DeviceHandle::DeviceHandle(int i) {
    status = nvmlDeviceGetHandleByIndex_v2 (i, & handle);
}


DeviceHandle::~DeviceHandle() {}


void generate_device_properties(py::module & m){
    // This needs to be defined so that the ptr_wrapper has something to return
    py::class_<ptr_wrapper<cudaDeviceProp>>(m, "_CudaDeviceProp__ptr");

    py::class_<DeviceProperties>(m, "cudaDeviceProp")
        .def(py::init<int>())
        .def("get",
            [](DeviceProperties & a) {
                return ptr_wrapper<cudaDeviceProp>(a.get());
            }
        )
        .def("name",
            [](DeviceProperties & a) {
                std::string s(a.get()->name);
                return s;
            }
        )
#ifndef USE_HIP
        .def("uuid",
            [](DeviceProperties & a) {
                std::string s = mem_to_string(
                    reinterpret_cast<void *>(& a.get()->uuid), 16
                );
                return s;
            }
        )
#endif
        .def("pciBusID",
            [](DeviceProperties & a) {
                return a.get()->pciBusID;
            }
        )
        .def("pciDeviceID",
            [](DeviceProperties & a) {
                return a.get()->pciBusID;
            }
        )
        .def("pciDomainID",
            [](DeviceProperties & a) {
                return a.get()->pciBusID;
            }
        )
        .def("last_status",
            [](const DeviceProperties & a) {
                return CudaError(a.last_status());
            }
        );
}