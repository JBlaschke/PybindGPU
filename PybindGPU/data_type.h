#ifndef DATA_TYPE_H
#define DATA_TYPE_H

#include <set>
#include <ptr_wrapper.h>
#include <cuda_hip_wrapper.h>
#include <pybind11/pybind11.h>

#include <complex>
#include <pybind11/complex.h>

#include <error.h>
#include <event.h>


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
                _ret.insert(itr);                                              \
            }                                                                  \
        }                                                                      \
        return _ret;                                                           \
    }();                                                                       \
    return _value


#define GET_ALIAS_LIST(...)                                                    \
    static auto _value = []() {                                                \
        auto _ret = std::vector<std::string>{};                                \
        size_t count = 0;                                                      \
        for (auto itr : std::vector<std::string>{__VA_ARGS__}) {               \
            if (!itr.empty() && count > 0) {                                   \
                _ret.push_back(itr);                                           \
            }                                                                  \
            count ++;                                                          \
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
        static const auto & aliases() { GET_ALIAS_LIST(__VA_ARGS__); }         \
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


#include <data_type_impl.h>


template <template <size_t> class SpecT, typename Tp, size_t ... Idx>
void generate_enumeration(
        py::enum_<Tp> & _enum, std::index_sequence<Idx ...>
    ) {
        auto _generate = [& _enum](const auto & _labels, Tp _idx) {
            for (const auto & itr : _labels) {
                assert(!itr.empty());
                _enum.value((itr + "_alias").c_str(), _idx);
            }
        };

        FOLD_EXPRESSION(
            _enum.value(SpecT<Idx>::label().c_str(), static_cast<Tp>(Idx))
        );
        FOLD_EXPRESSION(_generate(SpecT<Idx>::aliases(), static_cast<Tp>(Idx)));
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
}


void generate_enumeration(py::module & _mod);
void generate_datatype(py::module & _mod);

#endif