"""Optional buffer-based C extension; NumPy headers are not needed to build."""
import os
from setuptools import Extension, setup

setup(ext_modules=[Extension(
    "uwcc._native", ["uwcc/_native.c"],
    libraries=[] if os.name == "nt" else ["m"],
    extra_compile_args=["/O2"] if os.name == "nt" else ["-O3"],
    optional=True,
)])
