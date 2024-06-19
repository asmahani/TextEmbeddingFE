from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='skmeans_lloyd_update_cython',
    ext_modules=cythonize("skmeans_lloyd_update_cython.pyx"),
    include_dirs=[np.get_include()]
)
