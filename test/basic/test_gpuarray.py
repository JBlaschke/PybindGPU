from PybindGPU.gpuarray import GPUArray, HostAllocator, PagelockedAllocator, \
                               SUPPORTED_DTYPES
import numpy

print(SUPPORTED_DTYPES)

print("")
print("Testing Default GPUArray constructor")
print("==============")

shape = (3, 5)
a1 = GPUArray(shape, dtype="float64")

a1_alias = a1.get()
# Copy data into page-locked memory
for i in range(shape[0]):
    for j in range(shape[1]):
        a1_alias[i, j] = (i+1)*(j+1)

a1.to_device()

# Check that the page-locked memory was actually written to by ckecking the
# data stored in the clone:
a1_clone_alias = a1.get()
for i in range(shape[0]):
    for j in range(shape[1]):
        print(a1_clone_alias[i, j])


print("")
print("Testing Un-Pinned GPUArray constructor")
print("==============")


a1_alloc = HostAllocator(shape, "float64")
a1 = GPUArray(shape, allocator=a1_alloc, dtype="float64")

a1_alias = a1.get()
# Copy data into page-locked memory
for i in range(shape[0]):
    for j in range(shape[1]):
        a1_alias[i, j] = (i+1)*(j+1)

a1.to_device()

# Check that the page-locked memory was actually written to by ckecking the
# data stored in the clone:
a1_clone_alias = a1.get()
for i in range(shape[0]):
    for j in range(shape[1]):
        print(a1_clone_alias[i, j])


print("")
print("Testing Pinned GPUArray constructor")
print("==============")


a1_alloc = PagelockedAllocator(shape, "float64")
a1 = GPUArray(shape, allocator=a1_alloc, dtype="float64")

a1_alias = a1.get()
# Copy data into page-locked memory
for i in range(shape[0]):
    for j in range(shape[1]):
        a1_alias[i, j] = (i+1)*(j+1)

a1.to_device()

# Check that the page-locked memory was actually written to by ckecking the
# data stored in the clone:
a1_clone_alias = a1.get()
for i in range(shape[0]):
    for j in range(shape[1]):
        print(a1_clone_alias[i, j])


print("")
print("Testing GPUArray aliased to numpy array")
print("==============")

a = numpy.array([1., 2., 3., 4., 5.])
a1 = GPUArray(a)

a1.to_device()

# Check that the page-locked memory was actually written to by ckecking the
# data stored in the clone:
a1_alias = a1.get()
for i in range(len(a)):
    print(a1_alias[i])

