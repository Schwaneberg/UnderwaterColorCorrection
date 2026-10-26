from types import SimpleNamespace

import pytest

from uwcc import nvidia_runtime as runtime


@pytest.fixture
def private_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("UWCC_NVIDIA_ENV", str(tmp_path/"nvidia"))
    monkeypatch.setattr(runtime.os, "geteuid", lambda: 1000)
    return runtime.environment_path()


def test_default_runtime_is_versioned_and_does_not_create_data(tmp_path, monkeypatch):
    monkeypatch.delenv("UWCC_NVIDIA_ENV", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    abi = f"{runtime.sys.version_info.major}.{runtime.sys.version_info.minor}"
    assert runtime.environment_path() == tmp_path/"uwcc/runtimes"/f"nvidia-py{abi}-ort{runtime.ORT_VERSION}"
    assert not (tmp_path/"uwcc").exists()


def test_setup_decline_performs_no_install(private_environment, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "no")
    monkeypatch.setattr(runtime.venv, "EnvBuilder", lambda **_: pytest.fail("Unexpected environment creation"))
    assert runtime.setup_main([]) == 0
    assert not private_environment.exists()


def test_setup_refuses_root(private_environment, monkeypatch):
    monkeypatch.setattr(runtime.os, "geteuid", lambda: 0)
    with pytest.raises(SystemExit) as exc:
        runtime.setup_main(["--yes"])
    assert exc.value.code == 1


@pytest.mark.parametrize("validation_code", [0, 1])
def test_setup_uses_private_pinned_wheels_and_validates(private_environment, monkeypatch, validation_code):
    calls = []
    def create(path):
        assert path == private_environment
        (path/"bin").mkdir(parents=True)
        (path/"bin/python").touch()
    def builder(**options):
        assert options == {"with_pip": True, "system_site_packages": True}
        return SimpleNamespace(create=create)
    monkeypatch.setattr(runtime.venv, "EnvBuilder", builder)
    monkeypatch.setattr(runtime.subprocess, "run", lambda args, **kwargs: calls.append(args))
    monkeypatch.setattr(runtime, "validate_environment", lambda _: SimpleNamespace(
        returncode=validation_code, stdout="", stderr="failed"))
    assert runtime.setup_main(["--yes"]) == validation_code
    assert calls == [[str(private_environment/"bin/python"), "-m", "pip", "--isolated", "install",
                      "--only-binary=:all:", "--index-url", "https://pypi.org/simple", runtime.REQUIREMENT]]


def test_setup_reuses_working_runtime(private_environment, monkeypatch):
    (private_environment/"bin").mkdir(parents=True)
    (private_environment/"bin/python").touch()
    monkeypatch.setattr(runtime, "validate_environment", lambda _: SimpleNamespace(returncode=0))
    monkeypatch.setattr(runtime.venv, "EnvBuilder", lambda **_: pytest.fail("Working runtime was modified"))
    assert runtime.setup_main(["--yes"]) == 0


def test_launcher_missing_runtime_never_downloads(private_environment, monkeypatch):
    monkeypatch.setattr(runtime.subprocess, "run", lambda *_: pytest.fail("Unexpected subprocess"))
    assert runtime.launch_main(["models"]) == 1


def test_launcher_rejects_cpu_fallback(private_environment, monkeypatch):
    (private_environment/"bin").mkdir(parents=True)
    (private_environment/"bin/python").touch()
    monkeypatch.setattr(runtime, "validate_environment", lambda _: SimpleNamespace(returncode=1, stderr="No CUDA", stdout=""))
    monkeypatch.setattr(runtime.os, "execv", lambda *_: pytest.fail("Launched without CUDA"))
    assert runtime.launch_main([]) == 1


@pytest.mark.parametrize("args,module,forwarded", [(["gui", "my video.mp4"], "uwcc", ["gui", "my video.mp4"]),
    (["model-tools", "validate-gpu"], "uwcc.model_setup", ["validate-gpu"])])
def test_launcher_uses_gpu_interpreter(private_environment, monkeypatch, args, module, forwarded):
    (private_environment/"bin").mkdir(parents=True)
    python = private_environment/"bin/python"
    python.touch()
    monkeypatch.setattr(runtime, "validate_environment", lambda _: SimpleNamespace(returncode=0))
    calls = []
    monkeypatch.setattr(runtime.os, "execv", lambda executable, command: calls.append((executable, command)))
    runtime.launch_main(args)
    assert calls == [(str(python), [str(python), "-m", module, *forwarded])]
