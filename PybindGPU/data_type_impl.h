//----------------------------------------------------------------------------//
// <data-type> <enum> <string identifiers>
//  the first string identifier is the "canonical name" (i.e. what gets encoded)
//  and the remaining string entries are used to generate aliases
//
DATA_TYPE(int16_t, Int16, "int16", "short")
DATA_TYPE(int32_t, Int32, "int32", "int")
DATA_TYPE(int64_t, Int64, "int64", "long")
DATA_TYPE(uint16_t, UInt16, "uint16", "unsigned_short")
DATA_TYPE(uint32_t, UInt32, "uint32", "unsigned", "unsigned_int")
DATA_TYPE(uint64_t, UInt64, "uint64", "unsigned_long")
DATA_TYPE(float, Float32, "float32", "float")
DATA_TYPE(double, Float64, "float64", "double")
DATA_TYPE(std::complex<float>, Complex64, "complex64", "complex64")
DATA_TYPE(std::complex<double>, Complex128, "complex128", "complex128")