#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pybind11
import setuptools
import logging

from enum                        import Enum, auto
from os.path                     import join
from glob                        import glob
from setuptools                  import setup, Extension
from distutils.command.build_ext import build_ext
from pybind11.setup_helpers      import Pybind11Extension


# configure logger
logger = logging.getLogger(__name__)
FORMAT = "[%(levelname)8s | %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(format=FORMAT)


class BuildType(Enum):
    CUDA = auto()
    ROCM = auto()


def find_in_path(name, path):
    """Find first instance of a file in a search path"""
    for dir in path.split(os.pathsep):
        binpath = join(dir, name)

        if os.path.exists(binpath):
            return os.path.abspath(binpath)

    return None


def locate_cuda():
    """
    Locate the CUDA environment on the system Returns a dict with keys 'home',
    'nvcc', 'include', and 'lib64' and values giving the absolute path to each
    directory.  Starts by looking for the CUDA_HOME env variable. If not found,
    everything is based on finding 'nvcc' in the PATH.
    """
    if "CUDA_HOME" in os.environ:
        # First check if the CUDA_HOME env variable is in use
        logger.info("Found $CUDA_HOME in your environment.")

        home = os.environ["CUDA_HOME"]
        nvcc = join(home, "bin", "nvcc")
    else:
        # Otherwise, search the PATH for NVCC
        logger.info("$CUDA_HOME not found, searching $PATH.")

        nvcc = find_in_path("nvcc", os.environ["PATH"])

        if nvcc is None:
            logger.info("Could not detect `nvcc` in your $PATH -- giving up.")
            return None

        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {"home": home, "nvcc": nvcc,
                  "include": join(home, "include"),
                  "lib64": join(home, "lib64")}

    for k, v in iter(cudaconfig.items()):
        if not os.path.exists(v):
            logger.critical(
                f"The CUDA {k} path could not be located in {v} -- giving up."
            )
            return None

    logger.info("Found a `nvcc` executable.")
    return cudaconfig


def locate_rocm():
    """
    Locate the HIP environment on the system Returns a dict with keys 'home',
    'hipcc', 'include', and 'lib64' and values giving the absolute path to each
    directory.  Starts by looking for the ROCM_PATH env variable. If not found,
    everything is based on finding 'hipcc' in the PATH.
    """
    if "ROCM_PATH" in os.environ:
        # First check if the ROCM_PATH env variable is in use
        logger.info("Found $ROCM_HOME in your environment.")

        home = os.environ["ROCM_PATH"]
        hipcc = join(home, "bin", "hipcc")
    else:
        # Otherwise, search the PATH for hipcc
        logger.info("$ROCM_HOME not found, searching $PATH.")

        hipcc = find_in_path("hipcc", os.environ["PATH"])

        if hipcc is None:
            logger.info("Could not detect `hipcc` in your $PATH -- giving up.")
            return None

        home = os.path.dirname(os.path.dirname(hipcc))

    cudaconfig = {"home": home, "hipcc": hipcc,
                  "include": join(home, "include"),
                  "lib": join(home, "lib")}

    for k, v in iter(cudaconfig.items()):
        if not os.path.exists(v):
            logger.critical(
                f"The ROCM {k} path could not be located in {v} -- giving up."
            )
            return None

    logger.info("Found a `hipcc` executable.")
    return cudaconfig


def customize_compiler_for_nvcc(self):
    """
    Inject deep into distutils to customize how the dispatch to gcc/nvcc works.
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going the
    OO route, I have this. Note, it's kindof like a wierd functional subclassing
    going on.
    """

    # Tell the compiler it can processes .cu
    self.src_extensions.append(".cu")

    # Save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # Now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == ".cu":
            # use the cuda for .cu files
            self.set_executable("compiler_so", CUDA["nvcc"])
            # use only a subset of the extra_postargs, which are 1-1
            # translated from the extra_compile_args in the Extension class
            postargs = extra_postargs["nvcc"]
        else:
            postargs = extra_postargs["gcc"]

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # Reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # Inject our redefined _compile method into the class
    self._compile = _compile


def customize_compiler_for_rocm(self):
    """
    Inject deep into distutils to customize how the dispatch to gcc/hipcc works.
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going the
    OO route, I have this. Note, it's kindof like a wierd functional subclassing
    going on.
    """

    # Tell the compiler it can processes .cu
    self.src_extensions.append(".cu")

    # Save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # Now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == ".cu":
            # use the cuda for .cu files
            self.set_executable("compiler_so", ROCM["hipcc"])
            # use only a subset of the extra_postargs, which are 1-1
            # translated from the extra_compile_args in the Extension class
            postargs = extra_postargs["hipcc"]
        else:
            # postargs = extra_postargs["gcc"]
            self.set_executable("compiler_so", ROCM["hipcc"])
            postargs = extra_postargs["hipcc"]

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # Reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # Inject our redefined _compile method into the class
    self._compile = _compile

# Run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        """
        Select between nvcc and hipcc based on BACKEND
        """
        if BACKEND == BuildType.CUDA:
            customize_compiler_for_nvcc(self.compiler)
        elif BACKEND == BuildType.ROCM:
            customize_compiler_for_rocm(self.compiler)

        build_ext.build_extensions(self)


def make_extension():
    sources = sorted(glob(join("PybindGPU", "*.cpp"))) + \
              sorted(glob(join("PybindGPU", "*.cu")))

    includes = [ 
        "PybindGPU", join("PybindGPU", "include"),
        pybind11.get_include(True ), pybind11.get_include(False)
    ]

    if BACKEND == BuildType.CUDA:
        lib_dir = CUDA["lib64"]
        libraries = ["cudart", "nvToolsExt"]
        includes.append(CUDA["include"]) 
        extra_compile_args={
            "gcc": ["-std=c++14", "-O3", "-shared", "-fPIC"],
            "nvcc": ["-std=c++14", "-O3", "-shared", "--compiler-options",
                     "-fPIC", "-lnvToolsExt"]
        }
    elif BACKEND == BuildType.ROCM:
        lib_dir = ROCM["lib"]
        libraries = ["amdhip64"]
        includes.append(ROCM["include"])
        extra_compile_args={
            #"gcc": ["-std=c++14", "-O3", "-shared", "-fPIC", "-DUSE_HIP",
            #        f"--amdgpu-target={HIP_TARGET}"],
            "hipcc": ["-std=c++14", "-O3", "-fPIC", "-fgpu-rdc", "-DUSE_HIP",
                      f"--amdgpu-target={HIP_TARGET}"]
        }

    return Extension(
        "PybindGPU.backend",
        sources,
        library_dirs=[lib_dir],
        libraries=libraries,
        runtime_library_dirs=[lib_dir],
        # this syntax is specific to this build system we're only going to use
        # certain compiler args with nvcc and not with gcc the implementation of
        # this trick is in customize_compiler() below
        extra_compile_args=extra_compile_args,
        include_dirs=includes
    )

if __name__ == "__main__":

    logger.setLevel(int(os.environ.get("PYBIND_GPU_LOG_LEVEL", "20")))

    CUDA = locate_cuda()
    ROCM = locate_rocm()

    if "PYBIND_GPU_PREFERRED_BACKEND" in os.environ:
        preferred = os.environ["PYBIND_GPU_PREFERRED_BACKEND"]
        if preferred == "CUDA":
            BACKEND = BuildType.CUDA
        elif preferred == "ROCM":
            BACKEND = BuildType.ROCM
        else:
            raise RuntimeError(
                "Valid settings for PYBIND_GPU_PREFERRED_BACKEND are CUDA, ROCM"
            )
    else:
        if (CUDA is None) and (ROCM is None):
            raise RuntimeError(
                "One of: `nvcc` or 'hipcc` is required to compile PybindCUDA"
            )
        elif (CUDA is not None) and (ROCM is not None):
            raise RuntimeError(
                "Both `nvcc` and 'hipcc` found! Which one should I use?"
            )
        elif CUDA is not None:
            BACKEND = BuildType.CUDA
        elif ROCM is not None:
            BACKEND = BuildType.ROCM

    if BACKEND == BuildType.ROCM:
        if "PYBIND_GPU_TARGET" not in os.environ:
            raise RuntimeError(
                "You must specify a $PYBIND_GPU_TARGET when building for ROCM."
            )
        HIP_TARGET = os.environ["PYBIND_GPU_TARGET"]

    with open("README.md", "r") as fh:
        long_description = fh.read()

    setup(
        name="PybindGPU",
        version="0.1.1",
        author="Johannes Blaschke",
        author_email="johannes@blaschke.science",
        description="",
        ext_modules=[make_extension()],
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/JBlaschke/PybindGPU",
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        python_requires='>=3.6',
        install_requires=[
          'pybind11'
        ],
        # inject our custom trigger
        cmdclass={'build_ext': custom_build_ext},
    )
