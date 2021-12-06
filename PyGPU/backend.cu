#include <data_type.h>
#include <device_wrapper.h>
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
    py::class_<ptr_wrapper<CUevent_st * >>(m, "_CUevent_st__ptr");

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



    // TODO: this is a clumsy way to define data types -- clean this up a wee
    // bit in the future.

    py::class_<ptr_wrapper<int *>>(m, "IntPtr_t");

    m.def(
        "NewIntPtr_t",
        []() {return ptr_wrapper<int *>(new int *); }
    );

    py::class_<ptr_wrapper<double *>>(m, "DoublePtr_t");

    m.def(
        "NewDoublePtr_t",
        []() {return ptr_wrapper<double *>(new double *); }
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
        "cudaFree",
        [](void * dev_ptr) {
            return CudaError(cudaFree(dev_ptr));
        }
    );


    m.def(
        "cudaFreeHost",
        [](void * ptr) {
            return CudaError(cudaFreeHost(ptr));
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


    // TODO: Template the argument data type
    m.def(
        "cudaMalloc",
        [](ptr_wrapper<int *> dev_ptr, uint64_t size) {
            return CudaError(cudaMalloc(dev_ptr.get(), size*sizeof(int)));
        }
    );

    m.def(
        "cudaMalloc",
        [](ptr_wrapper<double *> dev_ptr, uint64_t size) {
            return CudaError(cudaMalloc(dev_ptr.get(), size*sizeof(double)));
        }
    );


    // TODO: Template the argument data type
    m.def(
        "cudaMallocHost",
        [](ptr_wrapper<int *> dev_ptr, uint64_t size) {
            return CudaError(cudaMallocHost(dev_ptr.get(), size*sizeof(int)));
        }
    );

    m.def(
        "cudaMallocHost",
        [](ptr_wrapper<double *> dev_ptr, uint64_t size) {
            // TODO: use custom type for cudaError_t
            return CudaError(cudaMallocHost(dev_ptr.get(), size*sizeof(double)));
        }
    );

// //  __host__ ​cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind ) 
// cudaMemcpyDeviceToHost
// cudaMemcpyHostToDevice
// 
//     // TODO: Template the argument data type to direct data
//     // using custom argument type
//     m.def(
//         "cudaMemcpyDeviceToHost",
//         [](ptr_wrapper<int> dst, ptr_wrapper<int> src, uint64_t count) {
//             // TODO: use custom type for cudaError_t
//             return (int64_t) cudaMemcpy();
//         }
//     );
// 
//     m.def(
//         "cudaMallocHost",
//         [](ptr_wrapper<double *> dev_ptr, uint64_t size) {
//             // TODO: use custom type for cudaError_t
//             return (int64_t) cudaMallocHost(dev_ptr.get(), size*sizeof(double));
//         }
//     );




    m.attr("major_version")   = py::int_(0);
    m.attr("minor_version")   = py::int_(1);
    m.attr("release_version") = py::int_(0);

    // Let the user know that this backend has been compiled _with_ CUDA support
    m.attr("cuda_enabled")            = py::bool_(true);
}
