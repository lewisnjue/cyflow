from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "cyflow.tensor",
        sources=["src/cyflow/tensor.pyx", "src/cyflow/c_tensor.c"],
        include_dirs=["src/cyflow", np.get_include()],
        libraries=["openblas"],  # Links against libopenblas.so
        extra_compile_args=["-O3", "-march=native"],  # Optimization flags
    )
]

setup(
    name="cyflow",
    version="0.1.0",
    ext_modules=cythonize(ext_modules, compiler_directives={"language_level": "3"}),
    zip_safe=False,
)
