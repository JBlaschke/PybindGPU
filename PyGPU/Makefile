CXX      := g++
NVCXX    := nvcc
CXXSTD   := -std=c++14
CXXFLAGS := $(CXXSTD) -O3 -arch=sm_70 -I.
NVOPTS   := --compiler-options -fPIC 
LDFLAGS  := -lcuda
INCLUDES := $(shell python -m pybind11 --includes)
PYSUFFIX := $(shell python3-config --extension-suffix)

all: backend$(PYSUFFIX)

%.o: %.cpp
	$(NVCXX) -c $(CXXFLAGS) $(NVOPTS) $(INCLUDES) $< -o $@

%.o: %.cu
	$(NVCXX) -c $(CXXFLAGS) $(NVOPTS) $(INCLUDES) $< -o $@

backend$(PYSUFFIX): backend.o data_type.o
	$(NVCXX) -shared $(CXXFLAGS) $(NVOPTS) $(LDFLAGS) $^ -o $@

clean:
	rm *.o

realclean:
	rm backend$(PYSUFFIX)