from PybindGPU.gpuarray import dtype


c_dtypes = [None]*dtype.__size__.value

for k in dtype.__members__.keys():
    if k.endswith("_alias"):
        continue
    if k == "__size__":
        continue
    y = dtype.__members__[k]
    v = y.value
    c_dtypes[v] = k

print(c_dtypes)
print(tuple(dtype(i).name for i in range(dtype.__size__.value)))