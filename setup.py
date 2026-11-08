"""Optional buffer-based C extension; NumPy headers are not needed to build."""
import os
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.errors import CCompilerError


class NativeBuildExt(build_ext):
    """Keep AVX2 instructions out of the baseline/CPU-dispatch object."""

    def build_extension(self, ext):
        target = (self.plat_name or "").lower()
        x86 = any(name in target for name in ("amd64", "x86_64", "i386", "i686")) or target == "win32"
        if not x86:
            return super().build_extension(ext)
        flags = (["/O2", "/fp:strict", "/arch:AVX2", "/GL-"] if self.compiler.compiler_type == "msvc"
                 else ["-O3", "-mavx2", "-ffp-contract=off", "-fno-lto"])
        macros, objects = ext.define_macros[:], ext.extra_objects[:]
        force, compiler_force = self.force, self.compiler.force
        try:
            # Rebuild both objects together: cached baseline objects must not
            # retain a previous scalar/AVX2 macro after compiler availability
            # changes. These are small, optional native sources.
            self.force = self.compiler.force = True
            try:
                compiled = self.compiler.compile(
                    ["uwcc/_native_avx2.c"], output_dir=self.build_temp,
                    extra_postargs=flags, debug=self.debug,
                )
            except CCompilerError as exc:
                self.warn(f"AVX2 kernel unavailable; building scalar native fallback: {exc}")
            else:
                ext.define_macros = macros + [("UWCC_BUILD_AVX2", "1")]
                ext.extra_objects = objects + compiled
            super().build_extension(ext)
        finally:
            ext.define_macros, ext.extra_objects = macros, objects
            self.force, self.compiler.force = force, compiler_force

setup(cmdclass={"build_ext": NativeBuildExt}, ext_modules=[Extension(
    "uwcc._native", ["uwcc/_native.c"],
    depends=["uwcc/_native_rgb48.h", "uwcc/_native_avx2.c"],
    libraries=[] if os.name == "nt" else ["m"],
    extra_compile_args=["/O2", "/fp:strict"] if os.name == "nt" else ["-O3", "-ffp-contract=off"],
    optional=True,
)])
