from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        name="cyflow.tensor",
        sources=[
            "src/cyflow/tensor.pyx",
            "src/cyflow/c_tensor.c",
        ],
        include_dirs=["src/cyflow"],
        extra_compile_args=["-O3"],
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
)
