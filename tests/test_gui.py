import time

import pytest


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
    wait_until(lambda: window.project is not None and window.operation is None and window.viewer.corrected is not None)
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
    window.auto_correct()
    wait_until(lambda: window.operation is None and window.project.analysis is not None)
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
