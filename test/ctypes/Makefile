CXX      := g++
NVCXX    := hipcc
CXXSTD   := -std=c++14
CXXFLAGS := $(CXXSTD) -DUSE_HIP -O3 --amdgpu-target=gfx90a -I.
NVOPTS   := -fPIC -fgpu-rdc
LDFLAGS  := -lhip
INCLUDES := -I../../PybindGPU -I../../PybindGPU/include

all: test_ctypes.so


%.so: %.cu
	$(NVCXX) -shared $(CXXFLAGS) $(NVOPTS) $(INCLUDES) $< -o $@

clean:
	rm *.so

