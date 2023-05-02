import PybindGPU  as gpuarray
import numpy as np


print("")
print("Testing GPUArray constructor with PybindGPU GPUArray")
print("==============")

# Send a numpy array to the device
k = np.arange(6).reshape(3,2)
k_gpu = gpuarray.to_gpu(k)
print(f'{k=} {type(k)=} {k.ctypes.data=}')
print(f'{k_gpu=} {type(k_gpu)=} {k_gpu.ptr=} {k_gpu.shape=}')
# Create another gpuarray pointing to the first gpuarray
j_gpu = gpuarray.GPUArray(allocator=gpuarray.Allocator(k_gpu))
# Set j_gpu to something else
j_gpu.set(0, 43)
j_gpu.set(1, 44)
j_gpu.set(8, 45)
# j is another numpy array allocated automatically by gpuarray.Allocator
j = j_gpu.get()
print(f'{j[0,:]=} {type(j)=} {j.ctypes.data=}')
print(f'{k=} {type(k)=} {k.ctypes.data=}')
# Check answers (k shouldn't change)
known_k = np.array([0,1,2,3,4,5], dtype=np.int64).reshape(3,2)
known_j = np.array([43,44], dtype=np.int64)
assert np.array_equal(k, known_k)
assert np.array_equal(j[0,:], known_j)

try:
    import cupy
    cupy_available = True
except Exception:
    cupy_available = False

if cupy_available:
    print("")
    print("Testing GPUArray constructor with cupy array")
    print("==============")
    k=cupy.arange(6).reshape(3,2)
    k_gpu=gpuarray.GPUArray(allocator=gpuarray.Allocator(k))
    print(f'{k=} {type(k)=} {k.data.ptr=}')
    print(f'{k_gpu=} {type(k_gpu)=} {k_gpu.ptr=} {k_gpu.shape=}')
    k_gpu.set(0, 43)
    k_gpu.set(1, 44)
    j = k_gpu.get()
    print(f'{j=} {type(j)=} {j.ctypes.data=}')
    print(f'{k=} {type(k)=} {k.data.ptr=}')
    known_j = np.array([43,44,2,3,4,5], dtype=np.int64).reshape(3,2)
    assert np.array_equal(k.get(), known_j)
    assert np.array_equal(j, known_j)
else:
    print(f'Skip Testing GPUArray constructor with cupy array: cupy is not available')

