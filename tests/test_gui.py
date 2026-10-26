import time

import pytest


@pytest.fixture
def automatic_window(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path/"data"))
    try:
        from uwcc.gui import MainWindow
        from uwcc.qt import QtWidgets as W
    except ImportError:
        pytest.skip("Qt binding is not installed")
    app = W.QApplication.instance() or W.QApplication([])
    window = MainWindow()
    errors = []
    monkeypatch.setattr(window, "error", errors.append)
    monkeypatch.setattr(window, "request_preview", lambda: None)

    def wait_until(predicate, timeout=10):
        deadline = time.monotonic()+timeout
        while not predicate() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert predicate(), errors

    yield window, wait_until, errors
    window.cancel_operation()
    wait_until(lambda: window.operation is None)
    window.dirty = False
    window.close()
    window.deleteLater()
    app.processEvents()


def test_automatic_analysis_waits_for_profile_and_refreshes_inputs(clip, automatic_window, monkeypatch):
    import uwcc.gui as gui
    from uwcc.media import probe
    from uwcc.project import Grade, Project
    window, wait_until, errors = automatic_window
    calls = []
    result = {"engine": "test", "shots": 1, "confidence": 1., "warnings": []}

    def analyze(info, grade, cancel, progress):
        calls.append((grade.input_profile, grade.filter_hint, grade.auto_method))
        return result.copy()

    monkeypatch.setattr(gui, "analyze", analyze)
    window.opened((probe(str(clip)), Project(str(clip), Grade(input_profile="auto"))))
    assert not window.analysis_pending
    assert not calls
    assert "Choose the input color profile" in window.statusBar().currentMessage()
    window.input_profile.setCurrentIndex(window.input_profile.findData("rec709"))
    wait_until(lambda: window.project.analysis is not None and window.operation is None)
    assert len(calls) == 1
    # Rapid filter/method changes coalesce into one analysis with the latest settings.
    window.filter_hint.setCurrentIndex(window.filter_hint.findData("red"))
    window.auto_method.setCurrentIndex(window.auto_method.findData("adaptive"))
    wait_until(lambda: len(calls) == 2 and window.project.analysis is not None and window.operation is None)
    assert calls[-1] == ("rec709", "red", "adaptive")
    window.controls["exposure"][0].setValue(.5)
    assert not window.analysis_pending
    assert not errors


def test_saved_analysis_is_preserved(clip, automatic_window, monkeypatch, tmp_path):
    import uwcc.gui as gui
    from uwcc.project import Grade, Project, fingerprint
    window, wait_until, errors = automatic_window
    # A valid saved analysis may represent a learned look and must not be replaced.
    analysis = {"version": 1, "engine": "saved reference", "shots": 1,
                "confidence": 1., "warnings": [],
                "samples": [{"time": 0., "scene": 0, "gain": [1., 1., 1.], "exposure": 0.}]}
    project = Project(str(clip), Grade(input_profile="rec709"), fingerprint(clip), analysis=analysis)
    path = tmp_path/"saved.json"
    project.save(path)
    monkeypatch.setattr(gui, "analyze", lambda *_: pytest.fail("Saved analysis was replaced"))
    window.load_project_path(str(path))
    wait_until(lambda: window.project is not None and window.operation is None)
    assert window.project.analysis == analysis
    assert not window.analysis_pending
    assert not window.dirty
    assert not errors


def test_settings_changed_during_analysis_discard_stale_result(clip, automatic_window, monkeypatch):
    import threading
    import uwcc.gui as gui
    window, wait_until, errors = automatic_window
    started, release = threading.Event(), threading.Event()
    calls = []

    def analyze(info, grade, cancel, progress):
        calls.append(grade.filter_hint)
        if len(calls) == 1:
            started.set()
            assert release.wait(5)
        return {"engine": grade.filter_hint, "shots": 1, "confidence": 1., "warnings": []}

    monkeypatch.setattr(gui, "analyze", analyze)
    window.open_video(path=str(clip))
    wait_until(started.is_set)
    window.filter_hint.setCurrentIndex(window.filter_hint.findData("red"))
    release.set()
    wait_until(lambda: len(calls) == 2 and window.project.analysis is not None and window.operation is None)
    assert window.project.analysis["engine"] == "red"
    assert not errors


def test_canceled_analysis_waits_for_retry(clip, automatic_window, monkeypatch):
    import threading
    import uwcc.gui as gui
    from uwcc.errors import Cancelled
    window, wait_until, errors = automatic_window
    started = threading.Event()
    calls = []

    def analyze(info, grade, cancel, progress):
        calls.append(grade.input_profile)
        if len(calls) == 1:
            started.set()
            assert cancel.wait(5)
            raise Cancelled()
        return {"engine": "retry", "shots": 1, "confidence": 1., "warnings": []}

    monkeypatch.setattr(gui, "analyze", analyze)
    window.open_video(path=str(clip))
    wait_until(started.is_set)
    window.cancel_operation()
    wait_until(lambda: window.operation is None)
    assert window.project.analysis is None
    assert not window.analysis_pending
    assert not window.analysis_timer.isActive()
    window.reanalyze_shortcut.activated.emit()
    wait_until(lambda: window.project.analysis is not None and window.operation is None)
    assert len(calls) == 2
    assert errors == ["Operation canceled."]


def test_window_open_preview_controls_and_analysis(clip, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path/"data"))
    try:
        from uwcc.gui import MainWindow
        from uwcc.qt import QtWidgets as W
    except ImportError:
        pytest.skip("Qt binding is not installed")
    app = W.QApplication.instance() or W.QApplication([])
    window = MainWindow()
    errors = []
    window.error = errors.append
    def wait_until(predicate, timeout=20):
        deadline = time.monotonic()+timeout
        while not predicate() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert predicate(), errors
    window.open_video(path=str(clip))
    wait_until(lambda: window.project is not None and window.project.analysis is not None
               and window.operation is None and window.viewer.corrected is not None)
    assert not any(action.text() == "Auto correct" for action in window.findChildren(type(window.save_action)))
    window.controls["exposure"][0].setValue(.5)
    assert window.project.grade.values["exposure"] == .5
    window.stabilize.setChecked(True)
    window.denoise.setValue(3.5)
    assert window.project.grade.stabilize
    assert window.project.grade.denoise == 3.5
    window.undo()
    assert window.project.grade.denoise == 0
    assert window.denoise.value() == 0
    window.redo()
    assert window.project.grade.denoise == 3.5
    window.add_key("exposure")
    assert window.project.grade.keyframes["exposure"]
    window.undo()
    assert not window.project.grade.keyframes
    window.toggle_playback()
    wait_until(lambda: window.position >= .25)
    window.stop_playback()
    app.processEvents()
    assert not errors
    wait_until(lambda: window.preview_worker is None)
    window.dirty = False
    window.close()
    app.processEvents()


@pytest.mark.parametrize("video_open", [False, True])
def test_load_preset_button_opens_chooser_with_or_without_video(clip, tmp_path, monkeypatch, video_open):
    import threading
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path/"data"))
    try:
        from uwcc.gui import MainWindow
        from uwcc.qt import QtWidgets as W
    except ImportError:
        pytest.skip("Qt binding is not installed")
    from uwcc.project import Grade, save_preset
    app = W.QApplication.instance() or W.QApplication([])
    window = MainWindow()
    monkeypatch.setattr(window, "request_preview", lambda: None)
    monkeypatch.setattr(window, "task", lambda operation, success: success(operation(threading.Event(), lambda *_: None)))
    errors = []
    monkeypatch.setattr(window, "error", errors.append)
    if video_open:
        window.open_video(path=str(clip))
    assert window.load_preset_button.isEnabled()
    assert window.save_preset_button.isEnabled() == video_open
    grade = Grade(input_profile="rec709")
    grade.values["exposure"] = .75
    grade.denoise = 3
    preset = tmp_path/"test.uwcc-preset.json"
    save_preset(preset, grade)
    dialogs = []
    def choose(parent, caption, directory, filters):
        dialogs.append(caption)
        assert "*.json" in filters
        return str(preset), filters
    monkeypatch.setattr(W.QFileDialog, "getOpenFileName", choose)
    window.load_preset_button.click()
    assert dialogs == ["Load preset"]
    if not video_open:
        assert window.pending_preset.values["exposure"] == .75
        assert "Open a video" in window.statusBar().currentMessage()
        window.open_video(path=str(clip))
        assert window.pending_preset is None
    assert window.project.grade.values["exposure"] == .75
    assert window.project.grade.denoise == 3
    assert window.controls["exposure"][0].value() == .75
    assert window.save_preset_button.isEnabled()
    # Canceling the chooser leaves the selected grade and undo history alone.
    before, history = window.project.grade.to_dict(), len(window.history)
    monkeypatch.setattr(W.QFileDialog, "getOpenFileName", lambda *_: ("", ""))
    window.load_preset_button.click()
    assert window.project.grade.to_dict() == before
    assert len(window.history) == history
    if video_open:
        window.undo()
        assert window.project.grade.values["exposure"] == 0
    assert not errors
    window.preview_timer.stop()
    window.dirty = False
    window.close()
    window.deleteLater()
    app.processEvents()
