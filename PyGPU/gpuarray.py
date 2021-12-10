import numpy as np


from .backend import dtype
from .backend import \
    DeviceArray_int16,    DeviceArray_int32,   DeviceArray_int64,\
    DeviceArray_uint16,   DeviceArray_uint32,  DeviceArray_uint64,\
    DeviceArray_float32,  DeviceArray_float64, DeviceArray_complex64


SUPPORTED_DTYPES = tuple(dtype(i).name for i in range(dtype.__size__.value))


class UnsupportedDataType(Exception):
    def __init__(self, message):
        super().__init__(message)


class GPUArray(object):

    def __init__(self, *args, **kwargs):

        a = args[0]

        if isinstance(a, np.ndarray):

            if kwargs.get("copy", False):
                self._hold = a.copy()
            else:
                self._hold = a

            if self._hold.dtype.name == "int16":
                self._device_array = DeviceArray_int16(self._hold)
                self._dtypestr = "int16"
            elif self._hold.dtype.name == "int32":
                self._device_array = DeviceArray_int32(self._hold)
                self._dtypestr = "int32"
            elif self._hold.dtype.name == "int64":
                self._device_array = DeviceArray_int64(self._hold)
                self._dtypestr = "int64"
            elif self._hold.dtype.name == "uint16":
                self._device_array = DeviceArray_uint16(self._hold)
                self._dtypestr = "uint16"
            elif self._hold.dtype.name == "uint32":
                self._device_array = DeviceArray_uint32(self._hold)
                self._dtypestr = "uint32"
            elif self._hold.dtype.name == "uint64":
                self._device_array = DeviceArray_uint64(self._hold)
                self._dtypestr = "uint64"
            elif self._hold.dtype.name == "float32":
                self._device_array = DeviceArray_float32(self._hold)
                self._dtypestr = "float32"
            elif self._hold.dtype.name == "float64":
                self._device_array = DeviceArray_float64(self._hold)
                self._dtypestr = "float64"
            elif self._hold.dtype.name == "complex64":
                self._device_array = DeviceArray_complex64(self._hold)
                self._dtypestr = "complex64"
            else:
                raise UnsupportedDataType(
                    f"Data type: {self._hold.dtype.name} is not supported!"
                )

        elif isinstance(a, tuple) or isinstance(a, list):
            a = list(a)  # make sure that a is a list (and not a tuple)
            dtype = kwargs["dtype"]

            if dtype in ("int16", np.int16):
                self._device_array = DeviceArray_int16(a)
                self._dtypestr = "int16"
            elif dtype in ("int32", np.int32):
                self._device_array = DeviceArray_int32(a)
                self._dtypestr = "int32"
            elif dtype in ("int64", np.int64):
                self._device_array = DeviceArray_int64(a)
                self._dtypestr = "int64"
            elif dtype in ("uint16", ):
                self._device_array = DeviceArray_uint16(a)
                self._dtypestr = "uint16"
            elif dtype in ("uint32", ):
                self._device_array = DeviceArray_uint32(a)
                self._dtypestr = "uint32"
            elif dtype in ("uint64", ):
                self._device_array = DeviceArray_uint64(a)
                self._dtypestr = "uint64"
            elif dtype in ("float32", np.float32):
                self._device_array = DeviceArray_float32(a)
                self._dtypestr = "float32"
            elif dtype in ("float64", np.float64):
                self._device_array = DeviceArray_float64(a)
                self._dtypestr = "float64"
            elif dtype in ("complex64", np.complex64):
                self._device_array = DeviceArray_complex64(a)
                self._dtypestr = "complex64"
            else:
                raise UnsupportedDataType(
                    f"Data type: {dtype} is not supported!"
                )

        else:
            raise UnsupportedDataType(
                "input must either a numpy array -- or a list, or a tuple"
            )

        self._device_array.allocate()


    @property
    def size(self):
        return self._device_array.size()


    @property
    def shape(self):
        return self._device_array.shape()


    @property
    def strides(self):
        return self._device_array.strides()


    @property
    def host_data(self):
        return self._device_array.host_data()


    @property
    def device_data(self):
        return self._device_array.device_data()


    @property
    def ptr(self):
        """
        PyCDUA compatibility: .ptr returns and integer representation of the
        device pointer.
        """
        # print("PTR PTR PTR:", self.device_data.__int__(), flush=True)
        # print("allocated:", self._device_array.allocated(), flush=True)
        return self.device_data.__int__()


    def get(self):
        """
        PyCUDA compatibility: .get() transfers data back to host and generates
        a numpy array out of the host buffer.
        """
        # print("HI THERE!", flush=True)
        self._device_array.to_host()
        # print("HO THERE!", flush=True)
        a = np.array(self._device_array)
        # print("HO HO!", flush=True)
        return a


    def __cuda_array_interface__(self):
        """
        Returns a CUDA Array Interface dictionary describing this array's data.
        """
        # print("__cuda_array_interface__", flush=True)

        return {
            "shape": self.shape,
            "strides": self.strides,
            # data is a tuple: (ptr, readonly) - always export GPUArray
            # instances as read-write
            "data": (self.ptr, False),
            "typestr": self._dtypestr,
            "stream": None,
            "version": 3
        }


    @property
    def dtype(self):
        return np.dtype(self._dtypestr)  # TODO: this seems inefficient 


    @property
    def last_status(self):
        return self._device_array.last_status()


    def allocate(self):
        self._device_array.allocate()


    def to_host(self):
        self._device_array.to_host()


    def to_device(self):
        self._device_array.to_device()


    def __getitem__(self, index):
        # TODO: Implement a proper index method -- this one just borrow's the
        # __getitme__ function from numpy, which make unnecessary copies, so
        # this is very much just a HACK!

        self.to_host()  # Send data to host

        host_cpy = np.array(self._device_array).copy()
        host_idx = host_cpy[index]  # Borrow __getitem__ from numpy
        device_new = GPUArray(host_idx)

        device_new.to_device()  # Send data back to device

        return device_new



def to_gpu(cpu_data):
    gpu_array = GPUArray(cpu_data)
    gpu_array.allocate()
    gpu_array.to_device()
    return gpu_array