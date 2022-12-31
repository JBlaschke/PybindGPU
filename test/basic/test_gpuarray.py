from PybindGPU.gpuarray import GPUArray, SUPPORTED_DTYPES

print(SUPPORTED_DTYPES)

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

