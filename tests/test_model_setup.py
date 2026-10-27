from types import SimpleNamespace

import pytest

from uwcc import model_setup


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
