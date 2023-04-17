#include <allocator.h>


void generate_allocator(py::module & _mod){
    generate_allocator<DataTypeSpecialization>(
        _mod,
        std::make_index_sequence<DataTypesEnd>{}
    );
}