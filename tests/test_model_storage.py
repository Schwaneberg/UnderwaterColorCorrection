from pathlib import Path

import pytest

from uwcc import neural
from uwcc.project import model_directory


def test_default_models_use_user_data_without_creating_directories(tmp_path, monkeypatch):
    monkeypatch.delenv("UWCC_MODEL_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path/"data"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path/"data"))
    assert model_directory() == tmp_path/"data/uwcc/models"
    assert not (tmp_path/"data").exists()


@pytest.mark.parametrize("task,folder", [("enhance", "udnet"), ("denoise", "fastdvdnet"), ("orientation", "rightwayup")])
def test_user_models_take_precedence_over_checkout(task, folder, tmp_path, monkeypatch):
    monkeypatch.delenv("UWCC_MODEL_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path/"data"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path/"data"))
    checkout = tmp_path/"checkout"
    (checkout/"pyproject.toml").parent.mkdir(parents=True)
    (checkout/"pyproject.toml").touch()
    monkeypatch.setattr(neural, "__file__", str(checkout/"uwcc/neural.py"))
    local = checkout/"models"/folder/"manifest.json"
    local.parent.mkdir(parents=True)
    local.write_text("{}")
    assert neural.builtin_model(task) == str(local)
    user = model_directory()/folder/"manifest.json"
    user.parent.mkdir(parents=True)
    user.write_text("{}")
    assert neural.builtin_model(task) == str(user)


def test_explicit_override_disables_checkout_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("UWCC_MODEL_DIR", str(tmp_path/"custom"))
    assert model_directory() == tmp_path/"custom"
    assert neural.builtin_model("enhance") is None
    assert not (tmp_path/"custom").exists()


def test_installed_package_does_not_search_sibling_models(tmp_path, monkeypatch):
    monkeypatch.delenv("UWCC_MODEL_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path/"data"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path/"data"))
    monkeypatch.setattr(neural, "__file__", str(tmp_path/"site-packages/uwcc/neural.py"))
    manifest = tmp_path/"site-packages/models/udnet/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    assert neural.builtin_model("enhance") is None
