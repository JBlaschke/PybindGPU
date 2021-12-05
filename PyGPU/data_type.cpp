#include <data_type.h>


int CudaError::as_int() const {
    return static_cast<int>(error_code);
}