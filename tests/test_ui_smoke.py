"""Builds the real window with a fake engine, visits every tab and pokes the
arm switch and the thread-safe log.  Skipped when Tk or a display is missing
(run with `xvfb-run -a python -m pytest tests` on a headless box)."""
import threading
import time

import pytest

tk = pytest.importorskip("tkinter")
pytest.importorskip("customtkinter")
np = pytest.importorskip("numpy")
pytest.importorskip("PIL")
pytest.importorskip("psutil")


@pytest.fixture
def app(pg):
    try:
        root = tk.Tk(); root.destroy()
    except tk.TclError:
        pytest.skip("no display available")
    from ui.app_window import GestureApp

    class Eng:
        shutdown_requested = False
        running = True
        latest_frame = np.zeros((360, 640, 3), np.uint8)
        def stop(self): pass
        def join(self, timeout=None): pass

    a = GestureApp(vision_engine=Eng())
    yield a
    try:
        a.destroy()
    except Exception:
        pass


def pump(app, sec=0.3):
    end = time.time() + sec
    while time.time() < end:
        app.update_idletasks(); app.update(); time.sleep(0.01)


def test_every_tab_builds(app):
    from ui.theme import ALL_TABS
    assert ALL_TABS == ["Live", "Gestures", "Settings", "Whiteboard", "Diagnostics"]
    for tab in ALL_TABS:
        app._navigate(tab); pump(app)
        assert app.get_active_tab() == tab
        assert app.views[tab].winfo_ismapped()


def test_arm_switch_controls_global_state(app):
    from core import os_input
    assert os_input.is_armed() is False
    app._arm_var.set(True); app._on_arm_switch(); pump(app, 0.1)
    assert os_input.is_armed() is True
    assert app._arm_label.cget("text") == "ARMED"
    os_input.ARMING.set_armed(False)             # e.g. panic hotkey on another thread
    pump(app, 0.1)
    assert app._arm_var.get() is False and app._arm_label.cget("text") == "DISARMED"


def test_log_is_thread_safe_and_replayed(app):
    def worker():
        for i in range(50):
            app.append_log(f"line {i}")
    t = threading.Thread(target=worker); t.start(); t.join()
    pump(app, 0.2)                                # drained on the GUI thread
    app._navigate("Diagnostics"); pump(app, 0.3)
    text = app.views["Diagnostics"].telemetry_box.get("1.0", "end")
    assert "line 0" in text and "line 49" in text


def test_dead_features_are_gone(app):
    labels = " ".join(b.cget("text") for b in app._nav_btns.values())
    for word in ("Music", "Games", "Mobile", "Shortcut", "AG"):
        assert word not in labels


def test_settings_validate_launcher_inline(app):
    from ui.theme import TAB_SETTINGS
    app._navigate(TAB_SETTINGS); pump(app, 0.2)
    v = app.views[TAB_SETTINGS]
    e = v.entries["2_fingers"]
    for bad in ("file:///etc/passwd", "javascript:alert(1)"):
        e.delete(0, "end"); e.insert(0, bad)
        assert v.validate_path("2_fingers") != ""
    e.delete(0, "end"); e.insert(0, "https://example.org")
    assert v.validate_path("2_fingers") == ""
    e.delete(0, "end")
    assert v.validate_path("2_fingers") == ""          # empty means "not set"


def test_settings_invalid_target_is_not_saved(app, tmp_path, monkeypatch):
    import config_manager
    from ui.theme import TAB_SETTINGS
    saved = []
    monkeypatch.setattr(config_manager, "save_config", lambda c: saved.append(dict(c)))
    app._navigate(TAB_SETTINGS); pump(app, 0.2)
    v = app.views[TAB_SETTINGS]
    e = v.entries["3_fingers"]
    e.delete(0, "end"); e.insert(0, "file:///x")
    v._commit_path("3_fingers")
    assert saved == []
    assert v.errors["3_fingers"].cget("text")
    e.delete(0, "end"); e.insert(0, "https://example.org")
    v._commit_path("3_fingers")
    assert saved and saved[-1]["app_paths"]["3_fingers"] == "https://example.org"
    assert v.errors["3_fingers"].cget("text") == ""


def test_settings_reset_needs_two_clicks(app, monkeypatch):
    import config_manager
    from ui.theme import TAB_SETTINGS
    monkeypatch.setattr(config_manager, "save_config", lambda c: None)
    app._navigate(TAB_SETTINGS); pump(app, 0.2)
    v = app.views[TAB_SETTINGS]
    app.config["mouse_smoothing"] = 12
    v._reset()
    assert app.config["mouse_smoothing"] == 12          # first click only arms it
    v._reset()
    assert app.config["mouse_smoothing"] == config_manager.DEFAULTS["mouse_smoothing"]


def test_gestures_view_marks_switched_off_features(app):
    from ui.theme import TAB_GESTURES
    app.config["enable_scroller"] = False
    app._navigate(TAB_GESTURES); pump(app, 0.2)
    texts = [l.cget("text") for l, _ in app.views[TAB_GESTURES]._state_labels]
    assert any("Switched off" in t for t in texts)


def test_live_view_shows_status_and_events(app):
    from core.status import STATUS, TIMELINE
    from ui.theme import TAB_LIVE
    TIMELINE.clear()
    STATUS.update(camera="connected", gestures={"Right": "Point"}, hold_label="Launch: hold",
                  hold_progress=0.5, fps=30.0)
    TIMELINE.add("gesture", "R · Point")
    app._navigate(TAB_LIVE); pump(app, 0.3)
    live = app.views[TAB_LIVE]
    assert "Point" in live._right_txt.cget("text")
    assert "R · Point" in live.timeline.get("1.0", "end")
    assert "30 fps" in app._cam_label.cget("text")


def test_camera_disconnected_is_shown(app):
    from core.status import STATUS
    STATUS.update(camera="disconnected", camera_message="No frames; retrying")
    pump(app, 0.2)
    assert "disconnected" in app._cam_label.cget("text").lower()
    STATUS.update(camera="connected", camera_message="")


def test_events_not_lost_while_on_another_tab(app):
    from core.status import TIMELINE
    from ui.theme import TAB_LIVE, TAB_SETTINGS
    app._navigate(TAB_LIVE); pump(app, 0.1)
    app._navigate(TAB_SETTINGS); pump(app, 0.1)
    TIMELINE.add("action", "Click")
    pump(app, 0.2)
    app._navigate(TAB_LIVE); pump(app, 0.2)
    assert "Click" in app.views[TAB_LIVE].timeline.get("1.0", "end")


def test_diagnostics_clear_log_shows_empty_state(app):
    app.append_log("something"); pump(app, 0.2)
    app._navigate("Diagnostics"); pump(app, 0.2)
    d = app.views["Diagnostics"]
    assert "something" in d.telemetry_box.get("1.0", "end")
    d.clear_log()
    assert "No log lines" in d.telemetry_box.get("1.0", "end")


def test_window_fits_minimum_size_and_scaled_ui(app):
    import customtkinter as ctk
    app.geometry("980x620"); pump(app, 0.3)
    ctk.set_widget_scaling(1.25); pump(app, 0.5)
    try:
        for tab in ("Live", "Gestures", "Settings", "Whiteboard", "Diagnostics"):
            app._navigate(tab); pump(app, 0.2)
        # the top bar's right edge (Arm switch) must still be inside the window
        sw = app._arm_switch
        assert sw.winfo_rootx() + sw.winfo_width() <= app.winfo_rootx() + app.winfo_width()
    finally:
        ctk.set_widget_scaling(1.0)


def test_settings_classifier_choice_saved_and_fallback_note_shown(app, monkeypatch):
    import config_manager
    from core.status import STATUS
    from ui.theme import TAB_SETTINGS
    saved = []
    monkeypatch.setattr(config_manager, "save_config", lambda c: saved.append(dict(c)))
    app._navigate(TAB_SETTINGS); pump(app, 0.2)
    v = app.views[TAB_SETTINGS]
    assert v.cls_btn.get() == "rules"
    v.cls_btn.set("ml"); v._save("classifier", "ml")
    assert saved[-1]["classifier"] == "ml"
    STATUS.update(classifier="rules", classifier_note="ML classifier unavailable, using rules instead: x")
    pump(app, 0.2)
    assert "using rules instead" in v.cls_note.cget("text") and v.cls_note.winfo_ismapped()
    STATUS.update(classifier_note="")
    app.config["classifier"] = "rules"; pump(app, 0.2)
    assert v.cls_note.cget("text") == ""
    v._reset(); v._reset()
    assert v.cls_btn.get() == "rules"


# ── feed, navigation and robustness (user-reported on Windows) ──────────────
def _frames(n):
    out = []
    for i in range(n):
        f = np.zeros((360, 640, 3), np.uint8)
        f[:, : 20 + 40 * i] = (200, 100, 50)
        out.append(f)
    return out


def test_live_feed_keeps_updating(app):
    """Each new frame must reach the screen, not just the first one."""
    app._navigate("Live")
    live = app.views["Live"]
    seen = set()
    for f in _frames(6):
        app.vision_engine.latest_frame = f
        pump(app, 0.12)
        seen.add(id(live.feed._photo))
    assert live.feed.frames_shown >= 6
    assert len(seen) >= 4          # a fresh image object per frame


def test_tabs_stay_reachable_while_frames_stream(app):
    stop = threading.Event()

    def stream():
        i = 0
        while not stop.is_set():
            app.vision_engine.latest_frame = _frames(6)[i % 6]
            app.vision_engine.latest_camera_frame = app.vision_engine.latest_frame
            i += 1
            time.sleep(0.01)

    t = threading.Thread(target=stream, daemon=True); t.start()
    try:
        for tab in ("Gestures", "Settings", "Live", "Settings", "Gestures", "Diagnostics", "Live"):
            app._navigate(tab); pump(app, 0.15)
            assert app.get_active_tab() == tab
            assert app.views[tab].winfo_ismapped()
    finally:
        stop.set()


def test_keyboard_shortcuts_switch_tabs(app):
    app.focus_force(); pump(app, 0.1)
    for key, tab in (("2", "Gestures"), ("3", "Settings"), ("1", "Live")):
        app.event_generate(f"<Control-Key-{key}>"); pump(app, 0.1)
        assert app.get_active_tab() == tab


def test_refresh_loop_survives_a_view_error(app):
    app._navigate("Live")
    live = app.views["Live"]
    original = live.refresh
    calls = {"n": 0}

    def boom(status, frame):
        calls["n"] += 1
        raise RuntimeError("boom")

    live.refresh = boom
    pump(app, 0.3)
    assert calls["n"] >= 3         # still being called after the first failure
    live.refresh = original
    app._navigate("Settings")
    assert app.get_active_tab() == "Settings"


def test_camera_view_only_requested_on_diagnostics(app):
    app.vision_engine.want_camera_view = False
    app._navigate("Diagnostics"); pump(app, 0.1)
    assert app.vision_engine.want_camera_view is True
    app.vision_engine.latest_camera_frame = _frames(2)[1]
    pump(app, 0.2)
    assert app.views["Diagnostics"].camera_feed.frames_shown >= 1
    app._navigate("Live"); pump(app, 0.1)
    assert app.vision_engine.want_camera_view is False
