from math import prod

import numpy, PybindGPU

a = numpy.array([1., 2., 3., 4., 5.])
print(a.dtype)

# Alias a DeviceArray to the numpy array. Note that the suffix (_float64) needs
# to match the numpy array's dtype. Basic wrapper types (like
# DeviceArray_<dtype>) DON'T check the data type of buffers they are aliased to
da = PybindGPU.DeviceArray_float64(a)

# Allocate memory on device (and check result)
da.allocate()
print(da.last_status())

# Send data to device (and check result)
da.to_device()
print(da.last_status())

# Send data to host (and check result)
da.to_host()
print(da.last_status())


print("")
print("TESTING PAGELOCKED ALLOCATOR")

alloc_shape = [5, 2]
alloc_size = prod(alloc_shape)
alloc = PybindGPU.PagelockedAllocator_float64()
alloc.allocate(alloc_size)

da2 = PybindGPU.DeviceArray_float64(alloc.ptr(), alloc_shape)
# and create a clone (for testing purposes)
da2_clone = PybindGPU.DeviceArray_float64(alloc.ptr(), alloc_shape)

da2_alias = numpy.array(da2, copy=False)
# Copy data into page-locked memory
for i in range(alloc_shape[0]):
    for j in range(alloc_shape[1]):
        da2_alias[i, j] = (i+1)*(j+1)

# Check that the page-locked memory was actually written to by ckecking the
# data stored in the clone:
da2_clone_alias = numpy.array(da2, copy=False)
for i in range(alloc_shape[0]):
    for j in range(alloc_shape[1]):
        print(da2_clone_alias[i, j])

# allocate device memory
da2.allocate()
print(da2.last_status())

da2.to_device()
print(da2.last_status())

da2.to_host()
print(da2.last_status())