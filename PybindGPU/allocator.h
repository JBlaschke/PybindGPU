#ifndef ALLOCATOR_H
#define ALLOCATOR_H

// #include <iostream>
#include <ptr_wrapper.h>
#include <cuda_hip_wrapper.h>
#include <pybind11/pybind11.h>

#include <data_type.h>


namespace py = pybind11;


template<class T, class E>
class Allocator {
    public:
        Allocator() : _ptr(NULL), _bytes(0), _owner(false) {
            // std::cout << "Using default constructor" << std::endl;
        }
        Allocator(const Allocator & o) : _ptr(o._ptr), _bytes(o._bytes) {
            // std::cout << "Using copy constructor" << std::endl;
            _owner = false;
        }
        Allocator(Allocator && o) : _ptr(o._ptr), _bytes(o._bytes) {
            // std::cout << "Using move constructor" << std::endl;
            if (o._owner) {
                _owner   = true;
                o._owner = false;
            } else {
                _owner = false;
            }
        }
        ~Allocator() {
            // std::cout << "Destructor: " << _owner << std::endl;
            if (_owner) this->deallocate();
        }
        virtual E allocate(size_t elts) {
            // std::cout << "this is wrong" << std::endl;
            return E();
        }
        virtual E deallocate() {
            // std::cout << "this is wrong" << std::endl;
            return E();
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
            // std::cout << "Allocating on host" << std::endl;
            _bytes = sizeof(T)*elts;
            _ptr = (T*) malloc(_bytes);
            _owner = true;
        }

        void deallocate() override {
            // std::cout << "Dellocating on host" << std::endl;
            free(_ptr);
            _owner = false;
        }

        ~HostAllocator() {
            // std::cout << "HostDestructor: " << _owner << std::endl;
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
            // std::cout << "Allocating on device" << std::endl;
            _bytes = sizeof(T)*elts;
            _owner = true;
            return cudaMalloc((void**) & _ptr, _bytes);
        }

        cudaError_t deallocate() override {
            // std::cout << "DeviceDeallocating on device" << std::endl;
            _owner = false;
            return cudaFree(_ptr);
        }

        ~DeviceAllocator() {
            // std::cout << "Destructor: " << _owner << std::endl;
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
            // std::cout << "Allocating on pagelocked host" << std::endl;
            _bytes = sizeof(T)*elts;
            _owner = true;
            return cudaMallocHost((void**) & _ptr, _bytes);
        }

        cudaError_t deallocate() override {
            // std::cout << "Deallocating on pagelocked host" << std::endl;
            _owner = false;
            return cudaFreeHost(_ptr);
        }

        ~PagelockedAllocator() {
            // std::cout << "PagelockedDestructor: " << _owner << std::endl;
            if (_owner) this->deallocate();
        }
    protected:
        using  Allocator<T, cudaError_t>::_ptr;
        using  Allocator<T, cudaError_t>::_bytes;
        using  Allocator<T, cudaError_t>::_owner;
};


template <template <size_t> class SpecT, size_t ... DataIdx>
void generate_allocator(py::module & _mod, std::index_sequence<DataIdx ...>) {
    FOLD_EXPRESSION(
        py::class_<HostAllocator<typename SpecT<DataIdx>::type>>(
            _mod, ("HostAllocator_" + SpecT<DataIdx>::label()).c_str()
        )
        .def(py::init<>())
        .def("allocate",
            & HostAllocator<typename SpecT<DataIdx>::type>::allocate
        )
        .def("ptr",
            [](HostAllocator<typename SpecT<DataIdx>::type> & a) {
                using dtype = typename SpecT<DataIdx>::type;
                return ptr_wrapper<dtype>(a.ptr(), true);
            }
        )
    );
    FOLD_EXPRESSION(
        py::class_<DeviceAllocator<typename SpecT<DataIdx>::type>>(
            _mod, ("DeviceAllocator_" + SpecT<DataIdx>::label()).c_str()
        )
        .def(py::init<>())
        .def("allocate",
            [](DeviceAllocator<typename SpecT<DataIdx>::type> & a, size_t n) {
                return CudaError(a.allocate(n));
            }
        )
        .def("ptr",
            [](DeviceAllocator<typename SpecT<DataIdx>::type> & a) {
                using dtype = typename SpecT<DataIdx>::type;
                return ptr_wrapper<dtype>(a.ptr(), true);
            }
        )
    );
    FOLD_EXPRESSION(
        py::class_<PagelockedAllocator<typename SpecT<DataIdx>::type>>(
            _mod, ("PagelockedAllocator_" + SpecT<DataIdx>::label()).c_str()
        )
        .def(py::init<>())
        .def("allocate",
            [](PagelockedAllocator<typename SpecT<DataIdx>::type> & a, size_t n) {
                return CudaError(a.allocate(n));
            }
        )
        .def("ptr",
            [](PagelockedAllocator<typename SpecT<DataIdx>::type> & a) {
                using dtype = typename SpecT<DataIdx>::type;
                return ptr_wrapper<dtype>(a.ptr(), true);
            }
        )
    );
}


void generate_allocator(py::module & _mod);

#endif