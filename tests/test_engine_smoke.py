"""Headless smoke test of VisionEngine with a fake camera and fake tracker.

Needs cv2/numpy (skipped if absent).  The GUI module is stubbed because
Tk may not exist (CI).  Verifies: starts disarmed, arming flips cleanly,
one broken controller does not stop the frame, keys are freed on exit.
"""
import sys
import types

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")
pytest.importorskip("mediapipe")

from conftest import make_hand


@pytest.fixture
def engine(pg, monkeypatch):
    stub = types.ModuleType("ui.app_window")
    stub.GestureApp = object
    monkeypatch.setitem(sys.modules, "ui", types.ModuleType("ui"))
    monkeypatch.setitem(sys.modules, "ui.app_window", stub)
    sys.modules.pop("main", None)
    import main

    class Tracker:
        hands = None
        swap_handedness = True
        def get_landmarks(self, frame): return self.hands
        def draw_landmarks(self, frame): return frame
        def release(self): pass

    class Cap:
        def read(self): return True, np.zeros((480, 640, 3), np.uint8)
        def isOpened(self): return False
        def release(self): pass

    from features.air_transformer import AirTransformer
    from features.tab_switcher import EdgeJoystick
    from features.system_control import SystemController
    from features.media_control import MediaController
    from features.mouse_control import MouseController
    from features.dynamic_scroller import DynamicScroller
    from features.app_launcher import AppLauncher
    from features.canvas_pen import CanvasController
    from features.modifier_key import ModifierController
    ctrls = {
        "transformer": AirTransformer(frame_w=640, frame_h=480), "edge_joystick": EdgeJoystick(frame_w=640),
        "sys_ctrl": SystemController(), "media": MediaController(), "mouse": MouseController(),
        "scroller": DynamicScroller(), "launcher": AppLauncher(), "canvas": CanvasController(),
        "modifier": ModifierController(),
    }
    monkeypatch.setattr(main.cv2, "VideoCapture", lambda *a: Cap())
    eng = main.VisionEngine(Tracker(), ctrls, 640, 480)
    eng._tracker = Tracker()
    eng._log = lambda m: None
    return eng, pg


def frame():
    return np.zeros((480, 640, 3), np.uint8)


def test_starts_disarmed_and_publishes_frames(engine):
    eng, pg = engine
    assert eng.arming.armed is False
    eng._process_frame(frame())
    assert eng.latest_frame is not None and eng.latest_frame.shape == (480, 640, 3)


def test_broken_controller_does_not_stop_the_frame(engine):
    eng, pg = engine
    def boom(*a, **k): raise RuntimeError("scroller exploded")
    eng._ctrl["scroller"].update = boom
    eng._process_frame(frame())
    assert eng.latest_frame is not None
    assert eng._safe.failures["scroller"] == 1


def test_disarm_releases_held_keys(engine):
    eng, pg = engine
    from core import os_input
    eng.arming.set_armed(True)
    os_input.keyDown("ctrl"); os_input.mouseDown()
    eng._was_armed = True
    eng.arming.set_armed(False)
    eng._process_frame(frame())
    assert ("keyUp", ("ctrl",)) in pg.calls and ("mouseUp", ()) in pg.calls


def test_run_frees_keys_even_if_loop_crashes(engine):
    eng, pg = engine
    from core import os_input
    eng.arming.set_armed(True)
    os_input.keyDown("ctrl")
    def crash(): raise RuntimeError("camera driver exploded")
    eng._loop = crash
    eng.run()
    assert ("keyUp", ("ctrl",)) in pg.calls


def test_two_palms_arm_then_disarm(engine):
    eng, pg = engine
    eng._arm_gesture.hold_frames = 3
    eng._tracker.hands = [make_hand("Left", origin=(150, 300)), make_hand("Right", origin=(450, 300))]
    for _ in range(3):
        eng._process_frame(frame())
    assert eng.arming.armed is True
    eng._tracker.hands = None
    eng._process_frame(frame())
    eng._tracker.hands = [make_hand("Left", origin=(150, 300)), make_hand("Right", origin=(450, 300))]
    for _ in range(3):
        eng._process_frame(frame())
    assert eng.arming.armed is False


def test_live_frame_is_skeleton_only_and_camera_view_is_optional(engine):
    """Design C: the Live frame is the overlay on the dark backdrop, never the camera picture."""
    import numpy as np
    from conftest import make_hand
    eng, _ = engine
    eng._frame_count = 1
    cam = frame()
    cam[:] = (200, 150, 100)                       # a very recognisable "camera" colour (BGR)
    eng._tracker.hands = [make_hand("Right", pose="open")]
    eng.want_camera_view = False
    eng._process_frame(cam.copy())
    live = eng.latest_frame
    assert eng.latest_camera_frame is None
    bg_rgb = (14, 16, 19)
    on_bg = (np.abs(live.astype(int) - bg_rgb).sum(axis=2) == 0).mean()
    assert on_bg > 0.6                             # mostly backdrop, hand drawn on top
    assert (np.abs(live.astype(int) - bg_rgb).sum(axis=2) > 0).any()   # something was drawn
    assert not (live == (100, 150, 200)).all(axis=2).any()   # no camera pixels on Live
    eng.want_camera_view = True
    eng._process_frame(cam.copy())
    camera = eng.latest_camera_frame
    assert camera is not None and camera.shape == eng.latest_frame.shape
    assert (camera == (100, 150, 200)).all(axis=2).mean() > 0.3   # camera pixels present (RGB)
