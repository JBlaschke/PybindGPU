from os.path import abspath, dirname, join

import ctypes
from ctypes import c_double
from ctypes import c_int
from ctypes import c_float
from ctypes import c_void_p

import numpy as np
import PyGPU as gpuarray

# Create ctypes wrappers of the ctypes test functions

c_int_p = ctypes.POINTER(c_int)
c_float_p = ctypes.POINTER(c_float)
c_double_p = ctypes.POINTER(c_double)

lib = ctypes.cdll.LoadLibrary(
    join(dirname(abspath(__file__)), "test_ctypes.so")
)

c_print_address = lib.print_address
c_print_address.argtypes = [ c_void_p ]
c_test_ptr_int = lib.test_ptr_int
c_test_ptr_int.argtypes = [ c_void_p, c_int]
c_test_ptr_int_cuda = lib.test_ptr_int_cuda
c_test_ptr_int_cuda.argtypes = [ c_void_p, c_int]


# Test Backend

k = np.array([1, 2, 3])
dk = gpuarray.DeviceArray_int64(k)
dk.allocate()
dk.to_device()

dk.device_data().print_address()
print(dk.device_data().__int__())
c_print_address(dk.device_data().__int__())
c_test_ptr_int_cuda(dk.device_data().__int__(), 3)


# Test Frontend

k_gpu = gpuarray.to_gpu(k)

k_gpu.device_data.print_address()
print(k_gpu.ptr)
c_print_address(k_gpu.ptr)
c_test_ptr_int_cuda(k_gpu.ptr, 3)