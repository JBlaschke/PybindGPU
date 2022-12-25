from PybindGPU.gpuarray import GPUArray, SUPPORTED_DTYPES

print(SUPPORTED_DTYPES)

shape = (3, 5)
a1 = GPUArray(shape, dtype="double")

a1_alias = a1.get()
# Copy data into page-locked memory
for i in range(alloc_shape[0]):
    for j in range(alloc_shape[1]):
        a1_alias[i, j] = (i+1)*(j+1)

# Check that the page-locked memory was actually written to by ckecking the
# data stored in the clone:
a1_clone_alias = a1.get()
for i in range(alloc_shape[0]):
    for j in range(alloc_shape[1]):
        print(a1_clone_alias[i, j])

