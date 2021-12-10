CXX      := g++
NVCXX    := nvcc
CXXSTD   := -std=c++14
CXXFLAGS := $(CXXSTD) -O3 -arch=sm_70 -I.
NVOPTS   := --compiler-options -fPIC 
INCLUDES := -IPyGPU -IPyGPU/include

all: test_ctypes.so

%.so: %.cu
	$(NVCXX) -shared $(CXXFLAGS) $(NVOPTS) $(INCLUDES) $< -o $@

clean:
	rm *.so