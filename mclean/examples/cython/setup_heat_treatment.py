from Cython.Build import cythonize
from setuptools import setup, Extension
import numpy as np

extensions = [
    Extension(
        "heat_treatment_annotated",
        ["./examples/cython/heat_treatment_annotated.pyx"],
        include_dirs=[np.get_include()],  # Include the NumPy headers
    ),
    Extension(
        "heat_treatment_cythonized",
        ["./examples/cython/heat_treatment_cythonized.pyx"],
        include_dirs=[np.get_include()],  # Include the NumPy headers
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': "3"}  # Ensure Cython uses Python 3 syntax
    ),
)
