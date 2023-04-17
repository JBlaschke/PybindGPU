#include <device_array.h>


void generate_device_array(py::module & _mod){
    generate_device_array<DataTypeSpecialization>(
        _mod,
        std::make_index_sequence<DataTypesEnd>{}
    );
}