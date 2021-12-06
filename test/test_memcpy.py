from sys import path

path.append("../PyGPU")

import numpy, backend

a = numpy.array([1., 2., 3., 4., 5.])

da = backend.DeviceArray_float64(a)
da.allocate()
print(da.last_status())
da.to_device()
print(da.last_status())
da.to_host()
print(da.last_status())