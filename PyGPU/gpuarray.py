import numpy as np


from .backend import dtype
from .backend import \
    DeviceArray_int16,    DeviceArray_int32,  DeviceArray_int64,\
    DeviceArray_uint16,   DeviceArray_uint32, DeviceArray_uint64,\
    DeviceArray_float32,  DeviceArray_float64


SUPPORTED_DTYPES = tuple(dtype(i).name for i in range(dtype.__size__.value))


class UnsupportedDataType(Exception):
    def __init__(self, message, errors):
        super().__init__(message)


class GPUArray(object):

    def __init__(self, a):
        if not isinstance(a, np.ndarray):
            raise UnsupportedDataType(
                "input must either a numpy array, or a tuple"
            )

        if a.dtype.name == "int16":
            self._device_array = DeviceArray_int16(a)
        elif a.dtype.name == "int32":
            self._device_array = DeviceArray_int32(a)
        elif a.dtype.name == "int64":
            self._device_array = DeviceArray_int64(a)
        elif a.dtype.name == "uint16":
            self._device_array = DeviceArray_uint16(a)
        elif a.dtype.name == "uint32":
            self._device_array = DeviceArray_uint32(a)
        elif a.dtype.name == "uint64":
            self._device_array = DeviceArray_uint64(a)
        elif a.dtype.name == "float32":
            self._device_array = DeviceArray_float32(a)
        elif a.dtype.name == "float64":
            self._device_array = DeviceArray_float64(a)
        else:
            raise UnsupportedDataType(
                f"Data type: {a.dtype.name} is not supported!"
            )





def to_gpu(cpu_data):
    pass