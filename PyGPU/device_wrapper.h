#ifndef DEVICE_WRAPPER_H
#define DEVICE_WRAPPER_H

#include <cuda_hip_wrapper.h>


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
        DeviceArray(size_t size)
        : m_size(size), device_allocated(false) {
            host_ptr = new T[size];
            host_allocated = true;
        };

        DeviceArray(T * data_ptr, size_t size)
        : m_size(size), device_allocated(false) {
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

        T * data() { return host_ptr; }
        size_t size() const { return m_size; }
        cudaError_t last_status() const { return status; };
        bool allocated() const { return device_allocated; };

    private:
        bool host_allocated;
        bool device_allocated;

        size_t m_size;
        T * host_ptr;
        T * device_ptr;

        cudaError_t status;
};

#endif