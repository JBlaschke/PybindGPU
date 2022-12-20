#!/bin/bash

_dir=$(readlink -f $(dirname ${BASH_SOURCE[0]}))
rm -rf ${_dir}/PybindGPU.egg-info ${_dir}/build ${_dir}/dist
rm -rf ${_dir}/PybindGPU/backend.cpython* ${_dir}/PybindGPU/__pycache__
