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