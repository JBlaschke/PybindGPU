#ifndef DATA_TYPE_H
#define DATA_TYPE_H

#include <set>
#include <cuda_hip_wrapper.h>

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
        ~ptr_wrapper() { delete ptr; }
        T & operator[](std::size_t idx) const { return ptr[idx]; }
    private:
        T * ptr;
};

struct CudaError {

    cudaError_t error_code;

    CudaError(cudaError_t a_error) : error_code(a_error) {}
    CudaError(int a_error) : error_code(static_cast<cudaError_t>(a_error)) {}
    int as_int() const;
};

#endif