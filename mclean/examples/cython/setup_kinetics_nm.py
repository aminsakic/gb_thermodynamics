from Cython.Build import cythonize
from setuptools import setup, Extension
import numpy as np

extensions = [
    Extension(
        "kinetics_cythonized_nm",
        ["./examples/cython/nelder_mead_cython.pyx"],
        include_dirs=[np.get_include()],  # Include the NumPy headers
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': "3"}  # Ensure Cython uses Python 3 syntax
    ),
)
