from pathlib import Path
from contextlib import contextmanager
import sys
from types import SimpleNamespace

import pytest

from uwcc import model_setup


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DLL search paths")
@pytest.mark.parametrize("fails", [False, True])
def test_frozen_converter_isolates_bundle_paths_and_restores_dll_search(tmp_path, monkeypatch, fails):
    bundle = tmp_path / "WindowsApps" / "UWCC"
    converter = tmp_path / "user runtime" / "python.exe"
    system = tmp_path / "system"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(bundle / "uwcc.exe"))
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle / "_internal"), raising=False)
    original = {"PATH": model_setup.os.pathsep.join(map(str, [bundle, bundle / "_internal" / "PySide6", system])),
                "UWCC_INFERENCE_DEVICE": "directml"}
    events = []

    @contextmanager
    def dll_search():
        events.append("clear")
        try:
            yield
        finally:
            events.append("restore")

    def launch(args, **kwargs):
        assert events == ["clear"]
        assert kwargs["env"] == {"PATH": str(system), "UWCC_INFERENCE_DEVICE": "directml"}
        assert kwargs["cwd"] == str(converter.parent)
        if fails:
            raise OSError("launch failed")
        return "child"

    monkeypatch.setattr(model_setup, "_system_dll_search", dll_search)
    if fails:
        with pytest.raises(OSError, match="launch failed"):
            model_setup._converter_process(launch, [str(converter)], env=original)
    else:
        assert model_setup._converter_process(launch, [str(converter)], env=original) == "child"
    assert events == ["clear", "restore"]
    assert str(bundle) in original["PATH"]


def test_frozen_app_self_launch_keeps_bundle_environment(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    calls = []
    model_setup._converter_process(lambda args, **kwargs: calls.append(kwargs), [sys.executable])
    assert calls == [{}]


@pytest.mark.parametrize("command,script", [("prepare", "prepare_models.py"),
                                           ("prepare-gpu", "prepare_gpu_denoise.py"),
                                           ("validate-gpu", "validate_gpu_denoise.py"),
                                           ("prepare-orientation", "prepare_orientation_model.py")])
def test_installed_model_commands_use_selected_storage_and_interpreter(tmp_path, monkeypatch, command, script):
    (tmp_path/script).touch()
    monkeypatch.setattr(model_setup, "tool_directory", lambda: tmp_path)
    monkeypatch.setenv("UWCC_MODEL_DIR", str(tmp_path/"user models"))
    calls = []
    monkeypatch.setattr(model_setup.subprocess, "run",
                        lambda args: calls.append(args) or SimpleNamespace(returncode=0))
    assert model_setup.main([command]) == 0
    assert calls == [[model_setup.sys.executable, str(tmp_path/script),
                      "--models-dir", str(tmp_path/"user models")]]


def test_model_download_only_and_failure_propagation(tmp_path, monkeypatch):
    (tmp_path/"prepare_models.py").touch()
    monkeypatch.setattr(model_setup, "tool_directory", lambda: tmp_path)
    calls = []
    monkeypatch.setattr(model_setup.subprocess, "run",
                        lambda args: calls.append(args) or SimpleNamespace(returncode=7))
    assert model_setup.main(["prepare", "--download-only", "--models-dir", str(tmp_path)]) == 7
    assert calls[0][-1] == "--download-only"
    with pytest.raises(SystemExit) as exc:
        model_setup.main(["prepare-gpu", "--download-only"])
    assert exc.value.code == 2


def test_run_setup_executes_requested_steps_and_reports_progress(tmp_path, monkeypatch):
    for script in model_setup.SCRIPTS.values():
        (tmp_path / script).touch()
    monkeypatch.setattr(model_setup, "tool_directory", lambda: tmp_path)
    monkeypatch.setattr(model_setup, "ensure_converter_python", lambda **_: "converter-py")
    calls = []
    monkeypatch.setattr(
        model_setup,
        "_run_cancellable",
        lambda args, cancel=None, env=None: calls.append(args) or f"ran {Path(args[1]).name}",
    )
    progress_events = []
    output = model_setup.run_setup(
        "all",
        models_dir=tmp_path / "models",
        progress=lambda frac, msg: progress_events.append((frac, msg)),
    )
    assert len(calls) == 2
    assert calls[0] == ["converter-py", str(tmp_path / "prepare_models.py"), "--models-dir", str(tmp_path / "models")]
    assert calls[1] == [
        model_setup.sys.executable,
        str(tmp_path / "prepare_orientation_model.py"),
        "--models-dir",
        str(tmp_path / "models"),
    ]
    assert "prepare_models.py" in output and "prepare_orientation_model.py" in output
    assert progress_events[-1][0] == 1.0


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows-specific converter bootstrap test')
def test_frozen_windows_app_bootstraps_embedded_python_and_installs_converter(tmp_path, monkeypatch):
    env = tmp_path / "converter"
    monkeypatch.setenv("UWCC_CONVERTER_ENV", str(env))
    monkeypatch.setattr(model_setup.sys, "frozen", True, raising=False)
    monkeypatch.setattr(model_setup.os, "name", "nt")
    python = env / "python.exe"

    def bootstrap(destination, cancel=None, progress=None):
        destination.mkdir(parents=True)
        python.touch()
        return python

    monkeypatch.setattr(model_setup, "_bootstrap_embedded_python", bootstrap)
    installs = []
    monkeypatch.setattr(model_setup, "_run_cancellable", lambda args, cancel=None, env=None: installs.append(args) or "")
    monkeypatch.setattr(model_setup.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stderr=""))
    assert model_setup.ensure_converter_python() == str(python)
    assert any("torch" in args and "https://download.pytorch.org/whl/cpu" in args for args in installs)
    assert any("onnx" in args and "onnxruntime" in args for args in installs)


def test_cli_model_tools_subcommand_delegates_to_model_setup(monkeypatch):
    from uwcc import __main__ as cli

    forwarded = []
    monkeypatch.setattr(model_setup, "main", lambda argv: forwarded.append(argv) or 0)
    assert cli.main(["model-tools", "prepare-orientation", "--models-dir", "/tmp/m"]) == 0
    assert forwarded == [["prepare-orientation", "--models-dir", "/tmp/m"]]


def test_converter_repairs_missing_pip_and_uses_isolated_python(tmp_path, monkeypatch):
    monkeypatch.setattr(model_setup, "_has_converter_modules", lambda: False)
    monkeypatch.setattr(model_setup, "converter_environment_path", lambda: tmp_path)
    python = model_setup._venv_python(tmp_path)
    python.parent.mkdir(parents=True, exist_ok=True)
    python.touch()
    checks = []
    results = iter([1, 1, 0])  # missing modules, missing pip, successful validation

    def check(args, **kwargs):
        checks.append(args)
        return SimpleNamespace(returncode=next(results), stderr="")

    monkeypatch.setattr(model_setup.subprocess, "run", check)
    installs = []
    monkeypatch.setattr(model_setup, "_run_cancellable",
                        lambda args, **kwargs: installs.append(args) or "")

    assert model_setup.ensure_converter_python() == str(python)
    assert installs[0] == [str(python), "-I", "-m", "ensurepip", "--upgrade"]
    assert "torch" in installs[1]
    assert "onnx" in installs[2]
    assert all(args[:2] == [str(python), "-I"] for args in checks + installs)


def test_converter_validation_failure_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(model_setup, "_has_converter_modules", lambda: False)
    monkeypatch.setattr(model_setup, "converter_environment_path", lambda: tmp_path)
    python = model_setup._venv_python(tmp_path)
    python.parent.mkdir(parents=True, exist_ok=True)
    python.touch()
    monkeypatch.setattr(model_setup.subprocess, "run",
                        lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="missing torch"))
    monkeypatch.setattr(model_setup, "_run_cancellable", lambda *args, **kwargs: "")
    with pytest.raises(model_setup.UWCCError, match="validation failed:\\nmissing torch"):
        model_setup.ensure_converter_python()
