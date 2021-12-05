#ifndef DATA_TYPE_H
#define DATA_TYPE_H

#include <set>

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

#endif