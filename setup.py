from pathlib import Path
import os

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"

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


# Custom build_ext command to handle .cu files with nvcc and smart flag filtering
class CudaBuildExt(build_ext):
    def build_extensions(self):
        # Teach setuptools to accept .cu files
        self.compiler.src_extensions.append('.cu')
        original_compile = self.compiler._compile

        def custom_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            if os.path.splitext(src)[1] == '.cu':
                # Locate nvcc executable
                nvcc_bin = os.path.join(CUDA_HOME, "bin", "nvcc")
                nvcc = nvcc_bin if os.path.exists(nvcc_bin) else "nvcc"
                
                postargs = extra_postargs.get('nvcc', ['-O3', '-std=c++17']) if isinstance(extra_postargs, dict) else extra_postargs
                
                # pp_opts contains all -I include directories from extension setup
                cmd = [nvcc, "-c", src, "-o", obj, "--compiler-options", "-fPIC"] + pp_opts + postargs
                self.spawn(cmd)
            else:
                postargs = extra_postargs.get('gcc', extra_postargs) if isinstance(extra_postargs, dict) else extra_postargs
                
                # --- SMART FILTER ---
                # We need to filter out C++ flags (like -std=c++17) for .c files
                # and filter out C flags (like -std=c99) for .cpp/.cc files to avoid compiler errors.
                safe_postargs = []
                is_c_file = src.endswith('.c')
                is_cpp_file = src.endswith('.cpp') or src.endswith('.cc') or src.endswith('.cxx')
                
                for arg in postargs:
                    if is_c_file and 'std=c++' in arg:
                        continue  # Skip C++ flags for C files
                    if is_cpp_file and 'std=c99' in arg:
                        continue  # Skip C flags for C++ files
                    safe_postargs.append(arg)

                original_compile(obj, src, ext, cc_args, safe_postargs, pp_opts)

        self.compiler._compile = custom_compile
        super().build_extensions()


extension_sources = [
    str(SRC_DIR / "cyflow" / "tensor.pyx"),
    str(SRC_DIR / "cyflow" / "cpu" / "tensor_cpu.c"),
]

libraries = ["cudart"]

if _env_flag("CYFLOW_ENABLE_CUDA"):
    extension_sources.append(str(SRC_DIR / "cyflow" / "cuda" / "tensor_cuda.cu"))
    extra_compile_args = {
        # Added -std=c++17 here so Cython's generated .cpp file uses C++17
        'gcc': ["-O3", "-std=c99", "-std=c++17"],
        'nvcc': ["-O3", "-std=c++17"]
    }
    libraries.extend(["cublas", "curand"])
else:
    extension_sources.append(str(SRC_DIR / "cyflow" / "cuda_stubs.c"))
    # Added -std=c++17 here too for the fallback build
    extra_compile_args = ["-O3", "-std=c99", "-std=c++17"]

ext_modules = [
    Extension(
        "cyflow.tensor",
        sources=extension_sources,
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        runtime_library_dirs=LIBRARY_DIRS,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=["-lstdc++"], # <--- FIX: Forces the C++ standard lib to link
        language="c++",               # <--- FIX: Forces setuptools to output a .cpp and use the C++ Linker
    )
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