#ifndef HOST_WRAPPER_H
#define HOST_WRAPPER_H

#include <iostream>


template<class T, class E>
class Allocator {
    public:
        Allocator() : _ptr(NULL), _bytes(0), _owner(false) {
            std::cout << "Using default constructor" << std::endl;
        }
        Allocator(const Allocator & o) : _ptr(o._ptr), _bytes(o._bytes) {
            std::cout << "Using copy constructor" << std::endl;
            _owner = false;
        }
        Allocator(Allocator && o) : _ptr(o._ptr), _bytes(o._bytes) {
            std::cout << "Using move constructor" << std::endl;
            if (o._owner) {
                _owner   = true;
                o._owner = false;
            } else {
                _owner = false;
            }
        }
        ~Allocator() {
            std::cout << "Destructor: " << _owner << std::endl;
            if (_owner) this->deallocate();
        }
        virtual E allocate(size_t elts) {
            std::cout << "this is wrong" << std::endl;
        }
        virtual E deallocate() {
            std::cout << "this is wrong" << std::endl;
        }
        T * ptr() {return _ptr;}
    protected:
        T * _ptr;
        size_t _bytes;
        bool _owner;
};


template<class T>
class HostAllocator : public Allocator<T, void> {
    public:
        void allocate(size_t elts) override {
            if (_owner) deallocate();
            std::cout << "Allocating on host" << std::endl;
            _bytes = sizeof(T)*elts;
            _ptr = (T*) malloc(_bytes);
            _owner = true;
        }

        void deallocate() override {
            std::cout << "Dellocating on host" << std::endl;
            free(_ptr);
            _owner = false;
        }

        ~HostAllocator() {
            std::cout << "HostDestructor: " << _owner << std::endl;
            if (_owner) this->deallocate();
        }
    protected:
        using  Allocator<T, void>::_ptr;
        using  Allocator<T, void>::_bytes;
        using  Allocator<T, void>::_owner;
};


template<class T>
class DeviceAllocator : public Allocator<T, cudaError_t> {
    public:
        cudaError_t allocate(size_t elts) override {
            if (_owner) deallocate();
            std::cout << "Allocating on device" << std::endl;
            _bytes = sizeof(T)*elts;
            _owner = true;
            return cudaMalloc((void**) & _ptr, _bytes);
        }

        cudaError_t deallocate() override {
            std::cout << "DeviceDeallocating on device" << std::endl;
            _owner = false;
            return cudaFree(_ptr);
        }

        ~DeviceAllocator() {
            std::cout << "Destructor: " << _owner << std::endl;
            if (_owner) this->deallocate();
        }
    protected:
        using  Allocator<T, cudaError_t>::_ptr;
        using  Allocator<T, cudaError_t>::_bytes;
        using  Allocator<T, cudaError_t>::_owner;
};


template<class T>
class PagelockedAllocator : public Allocator<T, cudaError_t> {
    public:
        cudaError_t allocate(size_t elts) override {
            if (_owner) deallocate();
            std::cout << "Allocating on pagelocked host" << std::endl;
            _bytes = sizeof(T)*elts;
            _owner = true;
            return cudaMallocHost((void**) & _ptr, _bytes);
        }

        cudaError_t deallocate() override {
            std::cout << "Deallocating on pagelocked host" << std::endl;
            _owner = false;
            return cudaFreeHost(_ptr);
        }

        ~PagelockedAllocator() {
            std::cout << "PagelockedDestructor: " << _owner << std::endl;
            if (_owner) this->deallocate();
        }
    protected:
        using  Allocator<T, cudaError_t>::_ptr;
        using  Allocator<T, cudaError_t>::_bytes;
        using  Allocator<T, cudaError_t>::_owner;
};


// template<class T>
// class HostArray {
//     public:
// 
//         HostArray() = delete;
//         HostArray(const HostArray & ) = delete;
// 
//         HostArray(std::vector<ssize_t> & shape)
//         : m_shape{shape}, device_allocated(false) {
//             // total size
//             m_size = std::accumulate(
//                 shape.begin(), shape.end(), 1,
//                 std::multiplies<ssize_t>()
//             );
//             // define array strides, assuming c-order
//             m_ndim = shape.size();
//             m_strides = std::vector<ssize_t>(m_ndim);
//             ssize_t stride = sizeof(T);
//             for (int i = m_ndim - 1; i >= 0; i--) {
//                 m_strides[i] = stride;
//                 stride = stride * shape[i];
//             }
//             // allocate data
//             host_ptr = new T[m_size];
//             // allocation status
//             host_allocated = true;
//             device_allocated = false;
//         };
// 
// 
//         T * host_data() { return host_ptr; }
//         ssize_t size() const { return m_size; }
//         const std::vector<ssize_t> & shape() const { return m_shape; };
//         const std::vector<ssize_t> & strides() const { return m_strides; };
// 
//         py::buffer_info buffer_info() {
//             return py::buffer_info(
//                 /* Pointer to buffer */
//                 host_ptr,
//                 /* Size of one scalar */
//                 sizeof(T),
//                 /* Python struct-style format descriptor */
//                 py::format_descriptor<T>::format(),
//                 /* Number of dimensions */
//                 m_ndim,
//                 /* Buffer dimensions */
//                 m_shape,
//                 /* Strides (in bytes) for each index */
//                 m_strides
//             );
//         }
// 
//     private:
//         ssize_t m_size;
//         std::vector<ssize_t> m_shape;
//         int m_ndim;
//         std::vector<ssize_t> m_strides;
// 
//         T * host_ptr;
// };

#endif