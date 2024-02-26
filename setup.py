import sys
from os import getenv
from pathlib import Path

from setuptools import setup
from setuptools_scm import get_version
from torch import version as torch_version
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
)

script_dir = Path(__file__).parent.resolve()

do_compile = getenv("EXLLAMA_NOCOMPILE") is None
ext_debug = getenv("EXLLAMA_DEBUG") is not None
nvcc_threads = getenv("NVCC_THREADS", "4")

## Assemble C/CPP compiler flags
if sys.platform == "win32":
    print("Windows build")
    cxx_flags = ["-Ox"]
    nvcc_flags = [
        "-Xcompiler",
        "/Zc:lambda",
        "-Xcompiler",
        "/Zc:preprocessor",
    ]
    libraries = ["cublas"]
else:
    print("Linux/macOS build")
    cxx_flags = ["-O3"]
    nvcc_flags = []
    libraries = []

## Assemble CUDA compiler flags
nvcc_flags += ["--threads", nvcc_threads, "-lineinfo", "-O3"]
if torch_version.hip is not None:
    print("HIP/ROCm build")
    nvcc_flags += ["-DHIPBLAS_USE_HIP_HALF"]
if ext_debug:
    print("Debug build mode, enabling time reporting and DSA support")
    nvcc_flags += [
        "--ptxas-options=-v",
        "-ftime-report",
        "-DTORCH_USE_CUDA_DSA",
    ]

## Generate list of extension sources
ext_dir = script_dir.joinpath("exllamav2", "exllamav2_ext")
ext_sources = [
    str(x.relative_to(script_dir))
    for x in ext_dir.rglob("**/*")
    if x.is_file() and x.suffix.lower() in [".cpp", ".cu"]
]

## Actual setup process
setup(
    ext_modules=[
        CUDAExtension(
            name="exllamav2_ext",  # as it would be imported
            # may include packages/namespaces separated by `.`
            sources=ext_sources,  # all sources are compiled into a single binary file
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
            libraries=libraries,
        ),
    ]
    if do_compile
    else None,  # if not compiling, do not include the extension, but still run the setup
    cmdclass={"build_ext": BuildExtension},
    version=get_version(),
)
