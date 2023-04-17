#include <data_type.h>


void generate_enumeration(py::module & _mod) {
    py::enum_<DataType> _dtype(_mod, "dtype", "Raw data types");
    _dtype.export_values();
    generate_enumeration<DataTypeSpecialization>(
        _dtype,
        std::make_index_sequence<DataTypesEnd>{}
    );
}


void generate_datatype(py::module & _mod) {
    generate_datatype<DataTypeSpecialization>(
        _mod,
        std::make_index_sequence<DataTypesEnd>{}
    );
}