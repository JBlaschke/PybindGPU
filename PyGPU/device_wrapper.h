#ifndef DEVICE_WRAPPER_H
#define DEVICE_WRAPPER_H

#include <iostream>
#include <vector>
#include <numeric>
#include <cuda_hip_wrapper.h>
#include <pybind11/pybind11.h>


namespace py = pybind11;


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


class CudaStream {
    public:
        CudaStream();
        CudaStream(unsigned int flags);
#ifndef USE_HIP
        CudaStream(unsigned int flags, int priority);
#endif
        ~CudaStream();

        cudaStream_t & operator* () { return stream; }
        cudaStream_t * get() { return & stream; }
        cudaError_t last_status() const { return status; }
    private:
        cudaStream_t stream;
        cudaError_t status;
};


template<class T>
class DeviceArray {
    public:
        DeviceArray(ssize_t size)
        : m_size(size), m_shape{size}, device_allocated(false) {
            // define array strides
            m_ndim = 1;
            m_strides = std::vector<ssize_t>(1);
            m_strides[0] = sizeof(T);
            // allocate data
            host_ptr = new T[size];
            host_allocated = true;
        };

        DeviceArray(T * data_ptr, ssize_t size)
        : m_size(size), m_shape{size}, device_allocated(false) {
            // define array strides
            m_ndim = 1;
            m_strides = std::vector<ssize_t>(1);
            m_strides[0] = sizeof(T);
            // transfer data
            host_ptr = data_ptr;
            host_allocated = false;
        };

        DeviceArray(std::vector<ssize_t> & shape)
        : m_shape{shape}, device_allocated(false) {
            // total size
            m_size = std::accumulate(
                shape.begin(), shape.end(), 1,
                std::multiplies<ssize_t>()
            );
            // define array strides, assuming c-order
            m_ndim = shape.size();
            m_strides = std::vector<ssize_t>(m_ndim);
            ssize_t stride = sizeof(T);
            for (int i = m_ndim - 1; i >= 0; i--) {
                m_strides[i] = stride;
                stride = stride * shape[i];
            }
            // allocate data
            host_ptr = new T[m_size];
            host_allocated = true;
        };

        DeviceArray(T * data_ptr, std::vector<ssize_t> & shape)
        : m_shape{shape}, device_allocated(false) {
            // total size
            m_size = std::accumulate(
                shape.begin(), shape.end(), 1,
                std::multiplies<ssize_t>()
            );
            // define array strides, assuming c-order
            m_ndim = shape.size();
            m_strides = std::vector<ssize_t>(m_ndim);
            ssize_t stride = sizeof(T);
            for (int i = m_ndim - 1; i >= 0; i--) {
                m_strides[i] = stride;
                stride = stride * shape[i];
            }
            // transfer data
            host_ptr = data_ptr;
            host_allocated = false;
        };

        ~DeviceArray() {
            if (host_allocated) delete host_ptr;
            if (device_allocated) status = cudaFree(device_ptr);
        }

        void allocate() {
            if (device_allocated) return;

            status = cudaMalloc(& device_ptr, m_size*sizeof(T));
            device_allocated = true;
        }

        void to_device() {
            if (!device_allocated) return;

            status = cudaMemcpy(
                device_ptr, host_ptr, m_size, cudaMemcpyHostToDevice
            );
        }

        void to_host() {
            if (!device_allocated) return;

            status = cudaMemcpy(
                host_ptr, device_ptr, m_size, cudaMemcpyDeviceToHost
            );
        }

        T * host_data() { return host_ptr; }
        T * device_data() { return device_ptr; }
        ssize_t size() const { return m_size; }
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
                // 1,
                m_ndim,
                /* Buffer dimensions */
                // { m_size },
                m_shape,
                /* Strides (in bytes) for each index */
                // { sizeof(T) }
                m_strides
            );
        }

    private:
        bool host_allocated;
        bool device_allocated;

        ssize_t m_size;
        int m_ndim;
        std::vector<ssize_t> m_shape;
        std::vector<ssize_t> m_strides;

        T * host_ptr;
        T * device_ptr;

        cudaError_t status;
};

#endif