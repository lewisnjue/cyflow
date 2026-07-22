from pathlib import Path
import os

from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
INCLUDE_DIRS = [str(SRC_DIR / "include"), np.get_include()]


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").lower() in {"1", "true", "yes", "on"}


extension_sources = [
    str(SRC_DIR / "cyflow" / "tensor.pyx"),
    str(SRC_DIR / "cyflow" / "cpu" / "tensor_cpu.c"),
]

extra_compile_args = ["-O3", "-std=c99"]
extra_link_args = []

if _env_flag("CYFLOW_ENABLE_CUDA"):
    extension_sources.append(str(SRC_DIR / "cyflow" / "cuda" / "tensor_cuda.cu"))
    extra_compile_args = ["-O3", "-std=c++17"]
    extra_link_args = ["-lcudart", "-lcublas", "-lcurand"]
else:
    extension_sources.append(str(SRC_DIR / "cyflow" / "cuda_stubs.c"))

ext_modules = [
    Extension(
        "cyflow.tensor",
        sources=extension_sources,
        include_dirs=INCLUDE_DIRS,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
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