#!/bin/bash
set -x

pushd basic
python test_dtype.py  
python test_gpuarray.py  
python test_memcpy.py  
python test_order.py
popd

pushd allocator
python test_allocator.py
python test_allocator_import.py
popd

pushd ctypes
python test_ctypes.py
popd

pushd cufinufft
python test_cufinufft_pygpu.py
popd


