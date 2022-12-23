#ifndef DATA_TYPE_H
#define DATA_TYPE_H

#include <set>
#include <ptr_wrapper.h>
#include <device_wrapper.h>
#include <host_wrapper.h>
#include <cuda_hip_wrapper.h>
#include <pybind11/pybind11.h>

#include <complex>
#include <pybind11/complex.h>


// TODO: these are from the PyKokkos source code -- and they need to be
// documented

#define FOLD_EXPRESSION(...)                                                   \
    ::consume_parameters(::std::initializer_list<int>{(__VA_ARGS__, 0)...})


#define GET_FIRST_STRING(...)                                                  \
    static std::string _value = []() {                                         \
        return std::get<0>(std::make_tuple(__VA_ARGS__));                      \
    }();                                                                       \
    return _value


#define GET_STRING_SET(...)                                                    \
    static auto _value = []() {                                                \
        auto _ret = std::set<std::string>{};                                   \
        for (auto itr : std::set<std::string>{__VA_ARGS__}) {                  \
            if (!itr.empty()) {                                                \
            _ret.insert(itr);                                                  \
            }                                                                  \
        }                                                                      \
        return _ret;                                                           \
    }();                                                                       \
    return _value


#define DATA_TYPE(TYPE, ENUM_ID, ...)                                          \
    template <>                                                                \
    struct DataTypeSpecialization<ENUM_ID> {                                   \
        using type = TYPE;                                                     \
        static std::string label() { GET_FIRST_STRING(__VA_ARGS__); }          \
        static const auto & labels() { GET_STRING_SET(__VA_ARGS__); }          \
    };


template <typename... Args>
void consume_parameters(Args && ...) {}


template <size_t data_type>
struct DataTypeSpecialization;


namespace py = pybind11;


enum DataType {
    Int16 = 0,
    Int32,
    Int64,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    Complex64,
    Complex128,
    DataTypesEnd
};


template <template <size_t> class SpecT, typename Tp, size_t ... Idx>
void generate_enumeration(
        py::enum_<Tp> & _enum, std::index_sequence<Idx ...>
    ) {
        auto _generate = [& _enum](const auto & _labels, Tp _idx) {
            for (const auto & itr : _labels) {
                assert(!itr.empty());
                _enum.value(itr.c_str(), _idx);
            }
        };

        FOLD_EXPRESSION(_generate(SpecT<Idx>::labels(), static_cast<Tp>(Idx)));
        _enum.value("__size__", DataTypesEnd);
}


template <template <size_t> class SpecT, size_t ... DataIdx>
void generate_datatype(py::module & _mod, std::index_sequence<DataIdx ...>) {
    FOLD_EXPRESSION(
        py::class_<ptr_wrapper<typename SpecT<DataIdx>::type>>(
            _mod, ("ptr_wrapper_" + SpecT<DataIdx>::label()).c_str()
        )
        .def(py::init<>())
        .def("create", & ptr_wrapper<typename SpecT<DataIdx>::type>::create)
        .def("destroy", & ptr_wrapper<typename SpecT<DataIdx>::type>::destroy)
        .def("is_safe", & ptr_wrapper<typename SpecT<DataIdx>::type>::is_safe)
        .def("__int__",
            [](const ptr_wrapper<typename SpecT<DataIdx>::type> & a) {
                intptr_t a_ptr = a;  // calls custom conversion operator
                return a_ptr;
            }
        )
        .def("print_address",
            & ptr_wrapper<typename SpecT<DataIdx>::type>::print_address
        )
        .def("__repr__",
            [](const ptr_wrapper<typename SpecT<DataIdx>::type> & a) {
                intptr_t a_ptr = a;  // calls custom conversion operator
                return "<ptr_wrapper<"
                    + SpecT<DataIdx>::label() + ">, "
                    + "is_safe=" + std::to_string(a.is_safe()) + ", "
                    + "addr=" + std::to_string(a_ptr)
                    + ">";
            }
        )
    );
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
    FOLD_EXPRESSION(
        py::class_<DeviceArray<typename SpecT<DataIdx>::type>>(
            _mod, ("DeviceArray_" + SpecT<DataIdx>::label()).c_str(),
            py::buffer_protocol()
        )
        .def(py::init<size_t>())
        .def(py::init(
            [](py::list l) {
                using dtype = typename SpecT<DataIdx>::type;
                std::vector<ssize_t> shape(py::len(l));
                for (size_t i = 0; i < shape.size(); i++) {
                    shape[i] = l[i].cast<ssize_t>();
                }
                return DeviceArray<dtype>(shape);
            }
        ), py::return_value_policy::reference)
        .def(py::init(
            [](ptr_wrapper<typename SpecT<DataIdx>::type> & a, py::list l) {
                using dtype = typename SpecT<DataIdx>::type;
                std::vector<ssize_t> shape(py::len(l));
                for (size_t i = 0; i < shape.size(); i++) {
                    shape[i] = l[i].cast<ssize_t>();
                }
                return DeviceArray<dtype>(a.get(), shape);
            }
        ), py::return_value_policy::reference)
        .def(py::init(
            [](py::buffer b) {
                py::buffer_info info = b.request();
                using dtype = typename SpecT<DataIdx>::type;
                return DeviceArray<dtype>(
                    static_cast<dtype *>(info.ptr), info.shape
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
        .def("shape",
            & DeviceArray<typename SpecT<DataIdx>::type>::shape
        )
        .def("strides",
            & DeviceArray<typename SpecT<DataIdx>::type>::strides
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
    );
}


void generate_enumeration(py::module & _mod);
void generate_datatype(py::module & _mod);

#endif