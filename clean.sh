#!/bin/bash

_dir=$(readlink -f $(dirname ${BASH_SOURCE[0]}))
rm -rf ${_dir}/PybindGPU.egg-info ${_dir}/build ${_dir}/dist