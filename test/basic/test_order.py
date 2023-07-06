from PybindGPU.gpuarray import GPUArray
import numpy as np


shape = (2,3,4)
dtype = np.complex64


def fill_array(arr):
    n=0
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                arr[i,j,k] = n
                n+=1


print("")
print("Testing shape constructor with C order")
print("==============")
order = "C"
a0_np = np.zeros(shape, dtype=dtype, order=order)
fill_array(a0_np)
print(f'{a0_np=}')
a1_ga = GPUArray(shape=shape, dtype=dtype, order=order)
print(f'{a1_ga.strides=}')
a1_np = a1_ga.get()
fill_array(a1_np)
a1_ga.to_device()
a2_np = a1_ga.get()
print(f'{a2_np=}')
assert np.allclose(a0_np, a2_np)


print("")
print("Testing shape constructor with F order")
print("==============")
order = "F"
a0_np = np.zeros(shape, dtype=dtype, order=order)
fill_array(a0_np)
print(f'{a0_np=}')
a1_ga = GPUArray(shape=shape, dtype=dtype, order=order)
a1_np = a1_ga.get()
fill_array(a1_np)
a1_ga.to_device()
a2_np = a1_ga.get()
print(f'{a2_np=}')
assert np.allclose(a0_np, a2_np)

print("")
print("Testing numpy array constructor with C order")
print("==============")
a1_np = np.zeros(shape, dtype=dtype, order="C")
fill_array(a1_np)
print(f'{a1_np=}')
a1_ga = GPUArray(a1_np, copy=False)
a1_np[0] = 1.0
print(f'{a1_np=}')
a1_ga.to_device()
a2_np = a1_ga.get()
print(f'{a2_np=}')
assert np.allclose(a1_np, a2_np)


print("")
print("Testing numpy array constructor with F order")
print("==============")
a1_np = np.zeros(shape, dtype=dtype, order="F")
fill_array(a1_np)
print(f'{a1_np=}')
a1_ga = GPUArray(a1_np, copy=False)
a1_np[0] = 1.0
print(f'{a1_np=}')
a1_ga.to_device()
a2_np = a1_ga.get()
print(f'{a2_np=}')
assert np.allclose(a1_np, a2_np)
