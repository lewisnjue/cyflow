from pathlib import Path
import os

from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"

# --- SMART CUDA PATH DETECTION ---


def find_cuda_path():
    # 1. Check if CUDA_HOME environment variable is set
    if "CUDA_HOME" in os.environ:
        return os.environ["CUDA_HOME"]

    # 2. Check standard NVIDIA runfile path
    if os.path.exists("/usr/local/cuda/include/cuda_runtime.h"):
        return "/usr/local/cuda"

    # 3. Check system apt package path (Ubuntu / Debian default)
    if os.path.exists("/usr/include/cuda_runtime.h"):
        return "/usr"

    return "/usr/local/cuda"  # Default fallback


CUDA_HOME = find_cuda_path()

# Determine include and library directories dynamically
if CUDA_HOME == "/usr":
    INCLUDE_DIRS = [str(SRC_DIR / "include"), np.get_include(), "/usr/include"]
    LIBRARY_DIRS = ["/usr/lib/x86_64-linux-gnu"]
else:
    INCLUDE_DIRS = [
        str(SRC_DIR / "include"),
        np.get_include(),
        os.path.join(CUDA_HOME, "include"),
    ]
    LIBRARY_DIRS = [os.path.join(CUDA_HOME, "lib64")]


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").lower() in {"1", "true", "yes", "on"}


extension_sources = [
    str(SRC_DIR / "cyflow" / "tensor.pyx"),
    str(SRC_DIR / "cyflow" / "cpu" / "tensor_cpu.c"),
]

extra_compile_args = ["-O3", "-std=c99"]
# Always link cudart since tensor.pyx references cudaMemcpy
libraries = ["cudart"]

if _env_flag("CYFLOW_ENABLE_CUDA"):
    extension_sources.append(str(SRC_DIR / "cyflow" / "cuda" / "tensor_cuda.cu"))
    extra_compile_args = ["-O3", "-std=c++17"]
    libraries.extend(["cublas", "curand"])
else:
    extension_sources.append(str(SRC_DIR / "cyflow" / "cuda_stubs.c"))

ext_modules = [
    Extension(
        "cyflow.tensor",
        sources=extension_sources,
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        runtime_library_dirs=LIBRARY_DIRS,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
    )
]

setup(
    name="cyflow",
    version="0.1.0",
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={"language_level": "3"},
    ),
    zip_safe=False,
)
