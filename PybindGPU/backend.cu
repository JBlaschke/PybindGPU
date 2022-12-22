#include <data_type.h>
#include <ptr_wrapper.h>
#include <device_wrapper.h>
#include <device_properties.h>
#include <cuda_hip_wrapper.h>
#include <pybind11/pybind11.h>


namespace py = pybind11;


PYBIND11_MODULE(backend, m) {

    // Build all enumerations used internally by cuda bindings
    generate_enumeration(m);

    // Build all datatype wrapper bindings
    generate_datatype(m);

    py::class_<CudaError>(m, "cudaError_t")
        .def(py::init<int>())
        .def("as_int", & CudaError::as_int)
        .def("__repr__",
            [](const CudaError & a) {
                return "<CudaError: 'code=" + std::to_string(a.as_int()) + "'>";
            }
        );

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

    m.def(
        "cudaDeviceReset",
        []() {
            return CudaError(cudaDeviceReset());
        }
    );

    m.def(
        "cudaDeviceSynchronize",
        []() {
            return CudaError(cudaDeviceSynchronize());
        }
    );

    m.def(
        "cudaEventElapsedTime",
        [](CudaEvent & start, CudaEvent & end) {
            float ms;
            cudaError_t err = cudaEventElapsedTime(& ms, * start, * end);
            return std::make_tuple(ms, CudaError(err));
        }
    );

    m.def(
        "cudaEventRecord",
        [](CudaEvent & event) {
            return CudaError(cudaEventRecord(* event, 0));
        }
    );

    m.def(
        "cudaEventRecord",
        [](CudaEvent & event, CudaStream & end) {
            return CudaError(cudaEventRecord(* event, * end));
        }
    );

    m.def(
        "cudaEventSynchronize",
        [](CudaEvent & event) {
            return CudaError(cudaEventSynchronize(* event));
        }
    );

    m.def(
        "cudaGetDevice",
        []() {
            int device;
            cudaError_t err = cudaGetDevice(& device);
            return std::make_tuple(device, CudaError(err));
        }
    );

    m.def(
        "cudaSetDevice",
        [](int device) {
            return CudaError(cudaSetDevice(device));
        }
    );

    m.def(
        "cudaGetErrorName",
        [](CudaError & error) {
            return std::string(cudaGetErrorName(* error));
        }
    );

    m.def(
        "cudaGetErrorString",
        [](CudaError & error) {
            return std::string(cudaGetErrorString(* error));
        }
    );

    m.def(
        "cudaGetLastError",
        []() {
            return CudaError(cudaGetLastError());
        }
    );

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

        m.def(
            "cudaGetDeviceCount",
            []() {
                int device;
                cudaError_t err = cudaGetDeviceCount(& device);
                return std::make_tuple(device, CudaError(err));
            }
        );

    m.attr("major_version")   = py::int_(0);
    m.attr("minor_version")   = py::int_(2);
    m.attr("release_version") = py::int_(1);

    // Let the user know if PybindGPU has been built in HIP mode instead of CUDA
#ifdef USE_HIP
    m.attr("use_hip") = py::bool_(true);
#else
    m.attr("use_hip") = py::bool_(false);
#endif

    // Let the user know that this backend has been compiled _with_ CUDA support
    m.attr("cuda_enabled") = py::bool_(true);
}
