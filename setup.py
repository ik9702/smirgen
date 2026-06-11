"""Build the native extension. Project metadata lives in pyproject.toml.

setuptools merges this with pyproject.toml: metadata comes from [project],
the C++ extension is declared here (extensions are not yet expressible in
pyproject.toml). Build with `pip install .`.
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "smirgen._smir_loop",
        ["smir_loop.cpp"],
        cxx_std=14,
        extra_compile_args=["-O3"],
    ),
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
