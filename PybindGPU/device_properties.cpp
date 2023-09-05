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


NvmlDevice::NvmlDevice(int i) {
    status = nvmlDeviceGetHandleByIndex_v2 (i, & handle);
}


NvmlDevice::~NvmlDevice() {}


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

    py::class_<NvmlDevice>(m, "nvmlDevice")
        .def(py::init<int>())
        .def("get",
            [](NvmlDevice & a) {
                return ptr_wrapper<nvmlDevice_t>(a.get());
            }
        )
        .def("utilization_rates",
            [](NvmlDevice & a) {
                nvmlUtilization_t util;
                a.status = nvmlDeviceGetUtilizationRates(* a, & util);
                return py::make_tuple(util.gpu, util.memory);
            }
        )
        .def("memory_info",
            [](NvmlDevice & a) {
                nvmlMemory_t mem;
                a.status = nvmlDeviceGetMemoryInfo(* a, & mem);
                return py::make_tuple(mem.free, mem.total, mem.used);
            }
        )
        .def("last_status",
            [](const NvmlDevice & a) {
                return NvmlReturn(a.status);
            }
        );

    m.def(
        "nvmlInit",
        [](){
            return NvmlReturn(nvmlInit_v2());
        }
    );

    m.def(
        "nvmlShutdown",
        [](){
            return NvmlReturn(nvmlShutdown());
        }
    );

}