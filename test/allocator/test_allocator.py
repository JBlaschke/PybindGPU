import PybindGPU

print("HOST ALLOCATOR:")
print("")

print("-----")
a = PybindGPU.HostAllocator_float32()
print(a.ptr())

print("-----")
a = PybindGPU.HostAllocator_float32()
print(a.ptr())

print("-----")
print(a.allocate(100))
print(a.ptr())

print("-----")
print(a.allocate(100))
print(a.ptr())

print("-----")
a = PybindGPU.HostAllocator_float32()
print(a.ptr())

print("")
print("DEVICE ALLOCATOR:")
print("")

print("-----")
a = PybindGPU.DeviceAllocator_float32()
print(a.ptr())

print("-----")
a = PybindGPU.DeviceAllocator_float32()
print(a.ptr())

print("-----")
print(a.allocate(100))
print(a.ptr())

print("-----")
print(a.allocate(100))
print(a.ptr())

print("-----")
a = PybindGPU.DeviceAllocator_float32()
print(a.ptr())

print("")
print("PAGELOCKED ALLOCATOR:")
print("")

print("-----")
a = PybindGPU.PagelockedAllocator_float32()
print(a.ptr())

print("-----")
a = PybindGPU.PagelockedAllocator_float32()
print(a.ptr())

print("-----")
print(a.allocate(100))
print(a.ptr())

print("-----")
print(a.allocate(100))
print(a.ptr())

print("-----")
a = PybindGPU.PagelockedAllocator_float32()
print(a.ptr())

