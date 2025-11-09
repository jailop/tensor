"""
Alternative simpler setup using direct compilation.
For use when CMake setup is complex.
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess

class get_pybind_include:
    def __str__(self):
        import pybind11
        return pybind11.get_include()

ext_modules = [
    Extension(
        'tensor4d',
        sources=[
            'tensor_wrapper.cc',
            '../src/tensor_instantiations.cc',
        ],
        include_dirs=[
            get_pybind_include(),
            '../include',
        ],
        language='c++',
        extra_compile_args=['-std=c++17', '-O3', '-Wall'],
    ),
]

setup(
    name='tensor4d',
    version='1.0.0',
    author='Tensor4D Team',
    description='Python bindings for Tensor4D C++ library',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.6.0', 'numpy>=1.19.0'],
    zip_safe=False,
    python_requires='>=3.7',
)
