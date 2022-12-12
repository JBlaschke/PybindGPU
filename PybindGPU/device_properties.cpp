#include <cstdint>  // used by intptr_t
#include <stdio.h>

#include <device_properties.h>
#include <cuda_hip_wrapper.h>


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