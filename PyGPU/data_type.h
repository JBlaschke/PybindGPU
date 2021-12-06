#ifndef DATA_TYPE_H
#define DATA_TYPE_H

#include <set>
#include <device_wrapper.h>
#include <cuda_hip_wrapper.h>
#include <pybind11/pybind11.h>


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


template <class T>
class ptr_wrapper {
    public:
        ptr_wrapper() : ptr(nullptr) {}
        ptr_wrapper(T * ptr) : ptr(ptr) {}
        ptr_wrapper(const ptr_wrapper & other) : ptr(other.ptr) {}
        void create(size_t N) { ptr = new T[N]; }
        T & operator* () const { return * ptr; }
        T * operator->() const { return   ptr; }
        T * get() const { return ptr; }
        void destroy() { delete ptr; }
        T & operator[](std::size_t idx) const { return ptr[idx]; }
    private:
        T * ptr;
};


template <class T>
struct obj_wrapper {
    T _obj;

    obj_wrapper(T & a_obj) : _obj(a_obj) {}
    obj_wrapper(T   a_obj) : _obj(a_obj) {}
    T & operator* () const { return _obj; }
    T & operator* ()       { return _obj; }
};


struct CudaError : public obj_wrapper<cudaError_t> {

    CudaError(int a_error) : obj_wrapper(static_cast<cudaError_t>(a_error)) {};

    int as_int() const;
};


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
        .def("__repr__",
            [](const ptr_wrapper<typename SpecT<DataIdx>::type> & a) {
                return "<ptr_wrapper<" +  SpecT<DataIdx>::label() + ">";
            }
        )
    );
    FOLD_EXPRESSION(
        py::class_<DeviceArray<typename SpecT<DataIdx>::type>>(
            _mod, ("DeviceArray_" + SpecT<DataIdx>::label()).c_str(),
            py::buffer_protocol()
        )
        .def(py::init<int>())
        .def_buffer(
            [](DeviceArray<typename SpecT<DataIdx>::type> & m) -> py::buffer_info {
                return py::buffer_info(
                    m.data(),
                    sizeof(typename SpecT<DataIdx>::type),
                    py::format_descriptor<typename SpecT<DataIdx>::type>::format(),
                    1,                                /* Number of dimensions */
                    { m.size() },                     /* Buffer dimensions */
                    { sizeof(typename SpecT<DataIdx>::type) }  /* Strides (in bytes) for each index */
                );
        })
    );
}


void generate_enumeration(py::module & _mod);
void generate_datatype(py::module & _mod);

#endif