"""Cython setup.

Make sure that you have cython. In the terminal, cd to this file's folder then
run:
    python step04_setup_cython_lstsqr.py build_ext --inplace
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize("step03_cython_lstsqr.pyx"))
