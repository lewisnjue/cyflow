from pathlib import Path
import os
import shutil

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
INCLUDE_DIR = SRC_DIR / "include"
CSRC_DIR = SRC_DIR / "csrc"


# --- SMART CUDA PATH DETECTION ---
def find_cuda_path():
    if "CUDA_HOME" in os.environ:
        return os.environ["CUDA_HOME"]
    if os.path.exists("/usr/local/cuda/include/cuda_runtime.h"):
        return "/usr/local/cuda"
    if os.path.exists("/usr/include/cuda_runtime.h"):
        return "/usr"
    return "/usr/local/cuda"


CUDA_HOME = find_cuda_path()

if CUDA_HOME == "/usr":
    INCLUDE_DIRS = [str(INCLUDE_DIR), np.get_include(), "/usr/include"]
    LIBRARY_DIRS = ["/usr/lib/x86_64-linux-gnu"]
else:
    INCLUDE_DIRS = [
        str(INCLUDE_DIR),
        np.get_include(),
        os.path.join(CUDA_HOME, "include"),
    ]
    LIBRARY_DIRS = [os.path.join(CUDA_HOME, "lib64")]


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").lower() in {"1", "true", "yes", "on"}


def _cuda_toolchain_available() -> bool:
    nvcc_candidates = [
        os.path.join(CUDA_HOME, "bin", "nvcc"),
        shutil.which("nvcc"),
    ]
    for candidate in nvcc_candidates:
        if candidate and os.path.exists(candidate):
            return True
    return os.path.exists(os.path.join(CUDA_HOME, "include", "cuda_runtime.h"))


# Custom build_ext command to handle .cu files with nvcc and smart flag filtering
class CudaBuildExt(build_ext):
    def build_extensions(self):
        self.compiler.src_extensions.append(".cu")
        original_compile = self.compiler._compile

        def custom_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            if os.path.splitext(src)[1] == ".cu":
                nvcc_bin = os.path.join(CUDA_HOME, "bin", "nvcc")
                nvcc = nvcc_bin if os.path.exists(nvcc_bin) else "nvcc"

                postargs = (
                    extra_postargs.get("nvcc", ["-O3", "-std=c++17"])
                    if isinstance(extra_postargs, dict)
                    else extra_postargs
                )
                cmd = [nvcc, "-c", src, "-o", obj, "--compiler-options", "-fPIC"] + pp_opts + postargs
                self.spawn(cmd)
            else:
                postargs = (
                    extra_postargs.get("gcc", extra_postargs)
                    if isinstance(extra_postargs, dict)
                    else extra_postargs
                )

                safe_postargs = []
                is_c_file = src.endswith(".c")
                is_cpp_file = src.endswith((".cpp", ".cc", ".cxx"))

                for arg in postargs:
                    if is_c_file and "std=c++" in arg:
                        continue
                    if is_cpp_file and "std=c99" in arg:
                        continue
                    safe_postargs.append(arg)

                original_compile(obj, src, ext, cc_args, safe_postargs, pp_opts)

        self.compiler._compile = custom_compile
        super().build_extensions()


USE_CUDA = _env_flag("CYFLOW_ENABLE_CUDA") and _cuda_toolchain_available()
if _env_flag("CYFLOW_ENABLE_CUDA") and not USE_CUDA:
    print("CUDA requested but no usable toolchain was found; falling back to CPU-only build.")


tensor_sources = [
    str(SRC_DIR / "cyflow" / "tensor.pyx"),
    str(CSRC_DIR / "core" / "tensor.c"),
    str(CSRC_DIR / "core" / "utils.c"),
    str(CSRC_DIR / "cpu" / "inline_op_cpu.c"),
    str(CSRC_DIR / "cpu" / "out_op_cpu.c"),
    str(CSRC_DIR / "core" / "utils_cuda.cu")
]

tensor_libraries = []

if USE_CUDA:
    tensor_sources.extend(
        [
            str(CSRC_DIR / "cuda" / "inline_op.cu"),
            str(CSRC_DIR / "cuda" / "tensor_cuda.cu"),
            str(CSRC_DIR / "cuda" / "out_op_cuda.cu"),  # <-- Added CUDA out-of-place ops
        ]
    )
    tensor_libraries.extend(["cudart", "cublas", "curand"])
    extra_compile_args = {
        "gcc": ["-O3", "-std=c99", "-std=c++17"],
        "nvcc": ["-O3", "-std=c++17"],
    }
else:
    tensor_sources.append(str(CSRC_DIR / "cuda" / "cuda_stubs.c"))
    extra_compile_args = ["-O3", "-std=c99", "-std=c++17"]

ext_modules = [
    Extension(
        "cyflow.tensor",
        sources=tensor_sources,
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        runtime_library_dirs=LIBRARY_DIRS,
        libraries=tensor_libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=["-lstdc++"],
        language="c++",
    ),
    Extension(
        "cyflow.autograd",
        sources=[str(SRC_DIR / "cyflow" / "autograd.pyx")],
        include_dirs=INCLUDE_DIRS,
        extra_compile_args=["-O3", "-std=c++17"],
        extra_link_args=["-lstdc++"],
        language="c++",
    ),
]

setup(
    name="cyflow",
    version="0.1.0",
    cmdclass={"build_ext": CudaBuildExt},
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={"language_level": "3"},
    ),
    zip_safe=False,
)