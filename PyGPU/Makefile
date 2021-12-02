CXX      := g++
NVCXX    := nvcc
CXXSTD   := -std=c++14
CXXFLAGS := $(CXXSTD) -O3 -shared -arch=sm_70 -I.
NVOPTS   := --compiler-options -fPIC 
INCLUDES := $(shell python -m pybind11 --includes)
PYSUFFIX := $(shell python3-config --extension-suffix)

all: backend$(PYSUFFIX)


%$(PYSUFFIX): %.cu
	$(NVCXX) $(CXXFLAGS) $(NVOPTS) $(INCLUDES) $< -o $@