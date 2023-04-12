import PybindGPU

print("HOST ALLOCATOR:")
print("")

print("1  -----")
a = PybindGPU.HostAllocator_float32()
print(a.ptr())

print("2  -----")
a = PybindGPU.HostAllocator_float32()
print(a.ptr())

print("3  -----")
print(a.allocate(100))
print(a.ptr())

print("4  -----")
print(a.allocate(100))
print(a.ptr())

print("5  -----")
a = PybindGPU.HostAllocator_float32()
print(a.ptr())

print("")
print("DEVICE ALLOCATOR:")
print("")

print("6  -----")
a = PybindGPU.DeviceAllocator_float32()
print(a.ptr())

print("7  -----")
a = PybindGPU.DeviceAllocator_float32()
print(a.ptr())

print("8  -----")
print(a.allocate(100))
print(a.ptr())

print("9  -----")
print(a.allocate(100))
print(a.ptr())

print("10 -----")
a = PybindGPU.DeviceAllocator_float32()
print(a.ptr())

print("")
print("PAGELOCKED ALLOCATOR:")
print("")

print("11 -----")
a = PybindGPU.PagelockedAllocator_float32()
print(a.ptr())

print("12 -----")
a = PybindGPU.PagelockedAllocator_float32()
print(a.ptr())

print("13 -----")
print(a.allocate(100))
print(a.ptr())

print("14 -----")
print(a.allocate(100))
print(a.ptr())

print("15 -----")
a = PybindGPU.PagelockedAllocator_float32()
print(a.ptr())