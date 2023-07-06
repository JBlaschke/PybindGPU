#ifndef DEVICE_ARRAY_H
#define DEVICE_ARRAY_H

// #include <iostream>
#include <vector>
#include <numeric>
#include <ptr_wrapper.h>
#include <cuda_hip_wrapper.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <error.h>
#include <event.h>
#include <data_type.h>


#include <iostream>


namespace py = pybind11;


template<class T>
class DeviceArray {
    public:

        DeviceArray() = delete;
        DeviceArray(const DeviceArray &) = delete;
        DeviceArray(DeviceArray && o)
        : m_size(o.m_size),
          m_shape(o.m_shape),
          m_ndim(o.m_ndim),
          m_strides(o.m_strides),
          host_ptr(o.host_ptr),
          device_ptr(o.device_ptr),
          device_allocated(o.device_allocated),
          host_allocated(o.host_allocated)
        {
            // Stop destructor (on other object) from freeing pointed memory
            // that has just been moved to this instance
            o.host_allocated = false;
            o.device_allocated = false;
        }

        DeviceArray(ssize_t size)
        : m_size(size), m_shape{size}, device_allocated(false) {
            // define array strides
            m_ndim = 1;
            m_strides = std::vector<ssize_t>(1);
            m_strides[0] = sizeof(T);
            // allocate data
            host_ptr = new T[size];
            // allocation status
            host_allocated = true;
            device_allocated = false;
        };

        DeviceArray(T * data_ptr, ssize_t size)
        : m_size(size), m_shape{size}, device_allocated(false) {
            // define array strides
            m_ndim = 1;
            m_strides = std::vector<ssize_t>(1);
            m_strides[0] = sizeof(T);
            // transfer data
            host_ptr = data_ptr;
            // allocation status
            host_allocated = false;
            device_allocated = false;
        };

        void set_strides(std::vector<ssize_t> & shape, int flag_c_contiguous){
            // define array strides, eiter c- or F-order
            m_ndim = shape.size();
            m_strides = std::vector<ssize_t>(m_ndim);
            ssize_t stride = sizeof(T);
            if (flag_c_contiguous == 1) {
                for (int i = m_ndim - 1; i >= 0; i--) {
                    m_strides[i] = stride;
                    stride = stride * shape[i];
                }
            } else {
                for (int i = 0; i < m_ndim; i++) {
                    m_strides[i] = stride;
                    stride = stride * shape[i];
                }
            }
        }

        DeviceArray(std::vector<ssize_t> & shape, int flag_c_contiguous)
        : m_shape{shape}, device_allocated(false) {
            // total size
            m_size = std::accumulate(
                shape.begin(), shape.end(), 1,
                std::multiplies<ssize_t>()
            );
            set_strides(shape, flag_c_contiguous);
            // allocate data
            host_ptr = new T[m_size];
            // allocation status
            host_allocated = true;
            device_allocated = false;
        };

        DeviceArray(T * data_ptr, std::vector<ssize_t> & shape, int flag_c_contiguous)
        : m_shape{shape}, device_allocated(false) {
            // total size
            m_size = std::accumulate(
                shape.begin(), shape.end(), 1,
                std::multiplies<ssize_t>()
            );
            set_strides(shape, flag_c_contiguous);
            // transfer data
            host_ptr = data_ptr;
            // allocation status
            host_allocated = false;
            device_allocated = false;
        };
        
        DeviceArray(T * data_ptr, std::vector<ssize_t> & shape, std::vector<ssize_t> & strides)
        : m_shape{shape}, device_allocated(false) {
            // total size
            m_size = std::accumulate(
                shape.begin(), shape.end(), 1,
                std::multiplies<ssize_t>()
            );
            m_ndim = shape.size();
            m_strides = strides;
            // transfer data
            host_ptr = data_ptr;
            // allocation status
            host_allocated = false;
            device_allocated = false;
        };
        
        DeviceArray(size_t host_addr, size_t device_addr, std::vector<ssize_t> & shape, int flag_c_contiguous)
        : m_shape{shape}, device_allocated(false) {
            // total size
            m_size = std::accumulate(
                shape.begin(), shape.end(), 1,
                std::multiplies<ssize_t>()
            );
            set_strides(shape, flag_c_contiguous);
            // Just point to the given addresses
            host_ptr = reinterpret_cast<T *>(host_addr);
            device_ptr = reinterpret_cast<T *>(device_addr);
            // allocation status
            host_allocated = false;
            device_allocated = false;
        };

        ~DeviceArray() {
            if (host_allocated)
                delete host_ptr;

            if (device_allocated)
                status = cudaFree(device_ptr);
        }

        void allocate() {
            if (device_allocated) return;

            status = cudaMalloc(& device_ptr, m_size*sizeof(T));
            device_allocated = true;
        }

        void to_device() {
            // FIXME: Below line is commented out until future fix.
            // CAUSE: With allocator, divice memory is not owned by this class
            // but we still need to be able to copy to and from this memory.   
            //if (!device_allocated) return;

            status = cudaMemcpy(
                device_ptr, host_ptr, m_size*sizeof(T), cudaMemcpyHostToDevice
            );
        }

        void to_host() {
            // FIXME: see to_device() 
            //if (!device_allocated) return;

            status = cudaMemcpy(
                host_ptr, device_ptr, m_size*sizeof(T), cudaMemcpyDeviceToHost
            );
        }
        
        void set(size_t external_host_addr) {
            // Provides compatibilty with PyCUDA .set(). 
            // The input argument is the address of an external array.

            // FIXME: see to_device() 
            //if (!device_allocated) return;
            
            T* external_host_ptr = reinterpret_cast<T *>(external_host_addr);

            status = cudaMemcpy(
                device_ptr, external_host_ptr, m_size*sizeof(T), cudaMemcpyHostToDevice
            );
        }

        void set_val(ssize_t idx, T val) {
            device_ptr[idx] = val;
        }

        T * host_data() { return host_ptr; }
        T * device_data() { return device_ptr; }
        ssize_t size() const { return m_size; }
        ssize_t nbytes() const { return m_size * sizeof(T); }
        const std::vector<ssize_t> & shape() const { return m_shape; };
        const std::vector<ssize_t> & strides() const { return m_strides; };
        cudaError_t last_status() const { return status; }
        bool allocated() const { return device_allocated; }

        py::buffer_info buffer_info() {
            return py::buffer_info(
                /* Pointer to buffer */
                host_ptr,
                /* Size of one scalar */
                sizeof(T),
                /* Python struct-style format descriptor */
                py::format_descriptor<T>::format(),
                /* Number of dimensions */
                m_ndim,
                /* Buffer dimensions */
                m_shape,
                /* Strides (in bytes) for each index */
                m_strides
            );
        }

    private:
        ssize_t m_size;
        std::vector<ssize_t> m_shape;
        int m_ndim;
        std::vector<ssize_t> m_strides;

        T * host_ptr;
        T * device_ptr;

        bool device_allocated;
        bool host_allocated;

        cudaError_t status;
};


template <template <size_t> class SpecT, size_t ... DataIdx>
void generate_device_array(py::module & _mod, std::index_sequence<DataIdx ...>) {
    FOLD_EXPRESSION(
        py::class_<DeviceArray<typename SpecT<DataIdx>::type>>(
            _mod, ("DeviceArray_" + SpecT<DataIdx>::label()).c_str(),
            py::buffer_protocol()
        )
        .def(py::init<size_t>())
        .def(py::init(
            [](py::list l, int flag_c_contiguous) {
                using dtype = typename SpecT<DataIdx>::type;
                std::vector<ssize_t> shape(py::len(l));
                for (size_t i = 0; i < shape.size(); i++) {
                    shape[i] = l[i].cast<ssize_t>();
                }
                return DeviceArray<dtype>(shape, flag_c_contiguous);
            }
        ), py::arg("l"), py::arg("flag_c_contiguous")=1, py::return_value_policy::reference)
        .def(py::init(
            [](ptr_wrapper<typename SpecT<DataIdx>::type> & a, py::list l, int flag_c_contiguous) {
                using dtype = typename SpecT<DataIdx>::type;
                std::vector<ssize_t> shape(py::len(l));
                for (size_t i = 0; i < shape.size(); i++) {
                    shape[i] = l[i].cast<ssize_t>();
                }
                return DeviceArray<dtype>(a.get(), shape, flag_c_contiguous);
            }
        ), py::arg("a"), py::arg("l"), py::arg("flag_c_contiguous")=1, py::return_value_policy::reference)
        .def(py::init(
            [](size_t host_addr, size_t device_addr, py::list l, int flag_c_contiguous) {
                using dtype = typename SpecT<DataIdx>::type;
                std::vector<ssize_t> shape(py::len(l));
                for (size_t i = 0; i < shape.size(); i++) {
                    shape[i] = l[i].cast<ssize_t>();
                }
                return DeviceArray<dtype>(host_addr, device_addr, shape, flag_c_contiguous);
            }
        ), py::arg("host_addr"), py::arg("device_addr"), py::arg("l"), py::arg("flag_c_contiguous")=1, py::return_value_policy::reference)
        .def(py::init(
            [](py::buffer b) {
                py::buffer_info info = b.request();
                using dtype = typename SpecT<DataIdx>::type;
                return DeviceArray<dtype>(
                    static_cast<dtype *>(info.ptr), info.shape, info.strides
                );
            }
        ), py::return_value_policy::reference)
        .def_buffer(
            [](DeviceArray<typename SpecT<DataIdx>::type> & m) {
                return m.buffer_info();
        })
        .def("size",
            & DeviceArray<typename SpecT<DataIdx>::type>::size
        )
        .def("nbytes",
            & DeviceArray<typename SpecT<DataIdx>::type>::nbytes
        )
        .def("strides",
            [](DeviceArray<typename SpecT<DataIdx>::type> & a) {
                py::object strides = py::cast(a.strides());
                return strides;
            }
        )
        .def("last_status",
            [](const DeviceArray<typename SpecT<DataIdx>::type> & a) {
                return CudaError(a.last_status());
            }
        )
        .def("allocate",
            & DeviceArray<typename SpecT<DataIdx>::type>::allocate
        )
        .def("to_host",
            & DeviceArray<typename SpecT<DataIdx>::type>::to_host
        )
        .def("to_device",
            & DeviceArray<typename SpecT<DataIdx>::type>::to_device
        )
        .def("host_data", 
            [](DeviceArray<typename SpecT<DataIdx>::type> & a) {
                using dtype = typename SpecT<DataIdx>::type;
                return ptr_wrapper<dtype>(a.host_data(), true);
            }
        )
        .def("device_data",
            [](DeviceArray<typename SpecT<DataIdx>::type> & a) {
                using dtype = typename SpecT<DataIdx>::type;
                return ptr_wrapper<dtype>(a.device_data(), false);
            }
        )
        .def("allocated",
            & DeviceArray<typename SpecT<DataIdx>::type>::allocated
        )
        .def("set_val",
            [](DeviceArray<typename SpecT<DataIdx>::type> & a, ssize_t idx, typename SpecT<DataIdx>::type val) {
                if (idx >= 0 && idx < a.size()) {
                    a.set_val(idx, val);
                }else{
                    printf("Idx must be between 0 and %ld (Idx: %ld)\n", a.size(), idx);
                }
            }
        )
        .def("set",
            & DeviceArray<typename SpecT<DataIdx>::type>::set
        )
        .def("shape",
            [](DeviceArray<typename SpecT<DataIdx>::type> & a) {
                py::object shape = py::cast(a.shape());
                return shape;
            }
        )
    );
}


void generate_device_array(py::module & _mod);

#endif
