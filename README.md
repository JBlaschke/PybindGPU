# PyGPU

Python bindings to control a GPU. This is intended to be a very lightweight
alternative to `PyCUDA`, or `CuPy`, as well as working on AMD GPUs. This module
allows all the usual device controls (setting/getting the device ID, stream, and
events), as well as controlling the data flow to and fromt he device. This
module does not let you launch device code -- that's up to you dear user.

What is this good for? If you have device code already, and you want to control
it from Python. This module has three ingredients:
1. control the device context
2. move data
3. abstract device and host pointers, as well as error codes

## Control the Device Context

Unlike `PyCUDA`, this module is intended to work with the CUDA API, and so it
uses the primary context. Currently we have: `cudaSetDevice` and `cudaGetDevice`
to control the device. We also have implemented python objects for `cudaEvent_t`
and `cudaStream_t`. Events/Streams are created when the Python constructor is
called. In the case of `cudaEvent_t` remember to call `cudaEventRecord` to
record the event onto the device.

## Move Data

We actually abstract devicde arrays using the `DeviceArray_<dtype>` object,
where `<dtype>` can be any of `int16`, `int32`, `int64`, `unit16`, `unit32`,
`uint64`, `float32`, `float64`. Device arrays have two sides: the `host_data()`
and the `device_data()`. When first created, `device_data()` is unallocated (you
can allocate it with the `allocate()` method). One way to create device data is
to pass a buffer (such as a `numpy` array), which can then be sent to the device
using the `to_device()` function. Finally, to get data back to the host, use the
`to_host()` function. This demonstrates a complete round-trip:

```python
A = numpy.array([1, 2, 3, 4, 5, 6])

da = DeviceArray_int64(A) # Point host_data() to the data pointer in A
da.allocate()             # Allocate memory on the device
da.to_device()            # Copy data to device
fn(da.device_data())      # Apply function to data on device
da.to_host()              # Copy data back into the data pointer in A
```

Note: the data type suffix (`_int64` in the example above) needs to match the
data type of the buffer.

## Abstract Host and Device Pointers and Error Codes

We use the design patter that every device call returns a `cudaError_t`. In
python this error code is represented by an object (containing an integer
representation of the error code -- so you can look it up online). `DeviceArray`
objects have a `last_status()` function which lets you check the error code of
the last device function call. If a device function does not return anything,
then the `cudaError_t` for that call is returned. For exaple `cudaSetDevice(2)`
will return `<cudaError: code=0>` if successful. If the device function has a
return value (e.g. in the CUDA API where we mihgt pass a pointer to an `int` or
a `float`),  then we return a tuple containing the returned value, and the error
code. For example `cudaGetDevice()` might return `(0, '<cudaError: code=0>')` if
successful.

In order to pass aroudn pointers, we use a `pointer_wrapper_<dtype>` class. This
encapsulates the pointers and allows them to be treated as Python objects.
Pointers controlled by the python process (host pointers that have been
allocated) are considered as "safe", and can be accessed using the `get()`
function. As Python doesn't have a concept of raw pointers, we follow the lead
of `PyCUDA` and allow raw pointers to be passed as integers (yea, I know:
shudder) using the `__int__` function.

Note: the `prt_wrapper` template is available here:
`PyGPU/include/ptr_wrapper.h`

## Installation

WIP. Currenly only avaialble via makefile. Be sure to point the `sys.path` to
where you compiled this module.