"""AVX2 flags belong only to its object; other targets retain scalar builds."""
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("target,compiler,avx", [
    ("win-amd64", "msvc", True), ("linux-x86_64", "unix", True),
    ("win-arm64", "msvc", False), ("linux-aarch64", "unix", False),
])
def test_native_build_targets_avx2_object_only(monkeypatch, tmp_path, target, compiler, avx):
    setuptools = pytest.importorskip("setuptools")
    from setuptools.command.build_ext import build_ext
    captured, compiles, links = {}, [], []
    monkeypatch.setattr(setuptools, "setup", lambda **kw: captured.update(kw))
    scope = runpy.run_path(str(Path(__file__).resolve().parents[1]/"setup.py"))
    ext = captured["ext_modules"][0]
    monkeypatch.setattr(build_ext, "build_extension", lambda self, item: links.append(
        (item.define_macros[:], item.extra_objects[:], item.extra_compile_args[:])))
    cmd = scope["NativeBuildExt"](setuptools.Distribution())
    cmd.plat_name, cmd.build_temp = target, str(tmp_path)
    cmd.compiler = SimpleNamespace(compiler_type=compiler, force=False,
        compile=lambda sources, **kw: compiles.append((sources, kw)) or ["avx2.obj"])
    cmd.build_extension(ext)
    assert bool(compiles) == avx
    assert (("UWCC_BUILD_AVX2", "1") in links[0][0]) == avx
    if avx:
        assert ("/arch:AVX2" if compiler == "msvc" else "-mavx2") in compiles[0][1]["extra_postargs"]
        assert ("/GL-" if compiler == "msvc" else "-fno-lto") in compiles[0][1]["extra_postargs"]
        assert links[0][1] == ["avx2.obj"]
    assert not any("AVX2" in flag or "mavx2" in flag for flag in links[0][2])
    assert ext.define_macros == [] and ext.extra_objects == []
    assert cmd.compiler.force is False


def test_avx2_compile_failure_still_builds_scalar_native(monkeypatch, tmp_path):
    setuptools = pytest.importorskip("setuptools")
    from setuptools.command.build_ext import build_ext
    from setuptools.errors import CompileError
    captured, links = {}, []
    monkeypatch.setattr(setuptools, "setup", lambda **kw: captured.update(kw))
    scope = runpy.run_path(str(Path(__file__).resolve().parents[1]/"setup.py"))
    monkeypatch.setattr(build_ext, "build_extension", lambda self, ext: links.append(ext.define_macros[:]))
    cmd = scope["NativeBuildExt"](setuptools.Distribution())
    cmd.plat_name, cmd.build_temp = "linux-x86_64", str(tmp_path)

    def fail(*args, **kwargs):
        raise CompileError("no AVX2 compiler")

    cmd.compiler = SimpleNamespace(compiler_type="unix", force=False, compile=fail)
    cmd.build_extension(captured["ext_modules"][0])
    assert links == [[]]
