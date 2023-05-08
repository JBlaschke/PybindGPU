from math import prod
import numpy as np

from . import backend

from .backend import dtype

SUPPORTED_DTYPES = tuple(dtype(i).name for i in range(dtype.__size__.value))


class UnsupportedDataType(Exception):
    def __init__(self, message):
        super().__init__(message)

class Allocator:
    def __init__(self, data, host_data=None):
        if isinstance(data, GPUArray):
            self.data = data
            self._shape = data.shape
            self._dtype = data.dtype.name
        elif isinstance(data, PagelockedAllocator):
            self.data = data
            self._shape = data._shape
            self._dtype = data._dtype
        else:
            # cupy array
            self.data = data.data
            self._shape = data.shape
            self._dtype = data.dtype.name
        if host_data is not None:
            self.host_data = host_data
        else:
            # Creates a numpy array corresponds to this gpuarray (for host_ptr)
            self.host_data = np.empty(self._shape, dtype=getattr(np, self._dtype))
        self._host_ptr = self.host_data.ctypes.data

    def ptr(self):
        if isinstance(self.data, PagelockedAllocator):
            return self.data.ptr()
        return self.data.ptr
    def host_ptr(self):
        return self._host_ptr


class GPUArray(object):
    def __init__(self, *args, **kwargs):
        if "allocator" in kwargs:
            self._allocator = kwargs["allocator"]
            self._has_allocator = True
            a = self._allocator._shape
            if ("shape" in kwargs) or ("dtype" in kwargs) or (len(args) > 0):
                raise RuntimeError(
                    "When specifying the allocator kwarg, it must be the only input argument."
                )
        else:
            self._has_allocator = False
            if "shape" in kwargs:
                a = kwargs["shape"]
            else:
                a = args[0]

        if isinstance(a, np.ndarray):
            if kwargs.get("copy", False):
                self._hold = a.copy()
            else:
                self._hold = a

            self._dtypestr = self._hold.dtype.name
            if self._dtypestr not in SUPPORTED_DTYPES:
                raise UnsupportedDataType(
                    f"Data type: {self._dtypestr} is not supported!"
                )

            array_constructor = getattr(
                backend, "DeviceArray_" + self._dtypestr
            )
            self._device_array = array_constructor(self._hold)

        elif isinstance(a, tuple) or isinstance(a, list):
            a = list(a)  # make sure that a is a list (and not a tuple)

            if self._has_allocator:
                dtype = self._allocator._dtype
            else:
                dtype = kwargs.get("dtype", "float64")

            if isinstance(dtype, type):
                dtype = dtype.__name__

            self._dtypestr = dtype 
            if self._dtypestr not in SUPPORTED_DTYPES:
                raise UnsupportedDataType(
                    f"Data type: {self._dtypestr} is not supported!"
                )

            array_constructor = getattr(
                backend, "DeviceArray_" + self._dtypestr
            )
            if self._has_allocator:
                self._device_array = array_constructor(self._allocator.host_ptr(), self._allocator.ptr(), a)
            else:
                self._device_array = array_constructor(a)

        else:
            raise UnsupportedDataType(
                "input must either a numpy array -- or a list, or a tuple"
            )

        if not self._has_allocator:
            self._device_array.allocate()


    @property
    def size(self):
        return self._device_array.size()


    @property
    def nbytes(self):
        return self._device_array.nbytes()


    @property
    def shape(self):
        # Pybind11 casts std::vector to a list. We convert
        # this to uple for compatibility with other libs. 
        return tuple(self._device_array.shape())


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
        Numpy/ctypes compatibility: .ptr returns and integer representation of
        the device pointer.
        """
        return self.device_data.__int__()


    def get(self, copy=False):
        """
        PyCUDA compatibility: .get() transfers data back to host and generates
        a numpy array out of the host buffer.
        """
        self._device_array.to_host()
        a = np.array(self._device_array, copy=copy)
        return a

    
    def __cuda_array_interface__(self):
        """
        Returns a CUDA Array Interface dictionary describing this array's data.
        """
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

    
    def set_val(self, idx, val):
        self._device_array.set_val(idx, val)


    def set(self, host_data):
        """
        PyCUDA compatibility: .set() transfers external numpy array data to gpu
        """
        assert isinstance(host_data, np.ndarray)
        assert np.array_equal(host_data.shape, self.shape)
        assert host_data.dtype == self.dtype
        self._device_array.set(host_data.ctypes.data)


    def __getitem__(self, index):
        # TODO: Implement a proper index method -- this one just borrow's the
        # __getitme__ function from numpy, which make unnecessary copies, so
        # this is very much just a HACK!

        self.to_host()  # Send data to host

        host_cpy = np.array(self._device_array, copy=False)
        host_idx = host_cpy[index]  # Borrow __getitem__ from numpy

        device_new = GPUArray(host_idx, copy=True) # TODO: this copy might be unneccessary
        device_new.to_device()  # Send data back to device

        return device_new


def to_gpu(cpu_data):
    gpu_array = GPUArray(cpu_data)
    # gpu_array.allocate()
    gpu_array.to_device()
    return gpu_array


class HostAllocator:
    def __init__(self, shape, dtype):
        if isinstance(dtype, type):
            dtype = dtype.__name__

        if dtype not in SUPPORTED_DTYPES:
            raise UnsupportedDataType(
                f"Data type: {dtype} is not supported!"
            )

        constructor = getattr(backend, "HostAllocator_" + dtype)
 
        self._shape = list(shape)
        self._dtype = dtype
        self._size  = prod(shape)
        self._alloc = constructor()
        self._alloc.allocate(self._size)


    def get(self):
        return np.array(self._alloc, copy=False)

    def ptr(self):
        return self._alloc.ptr()


class PagelockedAllocator:
    def __init__(self, shape, dtype):
        if isinstance(dtype, type):
            dtype = dtype.__name__

        if dtype not in SUPPORTED_DTYPES:
            raise UnsupportedDataType(
                f"Data type: {dtype} is not supported!"
            )

        constructor = getattr(backend, "PagelockedAllocator_" + dtype)
 
        self._shape = list(shape)
        self._dtype = dtype
        self._size  = prod(shape)
        self._alloc = constructor()
        self._alloc.allocate(self._size)


    def get(self):
        return np.array(self._alloc, copy=False)

    def ptr(self):
        return self._alloc.ptr()
