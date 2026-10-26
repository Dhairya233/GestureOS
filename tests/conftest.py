import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "hand_gesture_system")
if APP not in sys.path:
    sys.path.insert(0, APP)


class FakePG:
    """Stand-in for pyautogui that records every call."""

    def __init__(self, size=(1920, 1080)):
        self._size = size
        self.calls = []

    def size(self):
        return self._size

    def __getattr__(self, name):
        def rec(*a, **k):
            self.calls.append((name, a))
        return rec


@pytest.fixture
def pg():
    from core import os_input
    fake = FakePG()
    os_input.set_backend(fake)
    os_input.ARMING.set_armed(False)
    yield fake
    os_input.set_backend(None)
    os_input.ARMING.set_armed(False)


from core.poses import POSES, pose_points  # noqa: E402,F401  (single source of truth)


def make_hand(label="Right", rotation=0.0, scale=1.0, open_palm=True,
              origin=(300, 300), pose=None, pinch=None, hand_px=100.0):
    """Synthetic 21-landmark hand in pixel coordinates.

    pose:   one of POSES (default "open" / "fist" from open_palm)
    pinch:  if given, move the thumb tip to this fraction of hand size away
            from the index tip (a pinch ratio), on top of the pose.
    scale:  multiplies the hand size (distance from the camera)
    rotation: radians, rotates the whole hand about the wrist
    """
    import math
    pose = pose or ("open" if open_palm else "fist")
    pts = pose_points(pose)
    if pinch is not None:
        ix, iy = pts[8]
        pts[4] = (ix + pinch, iy)
    ca, sa = math.cos(rotation), math.sin(rotation)
    lms = []
    for i in range(21):
        dx, dy = pts[i]
        if label == "Left":
            dx = -dx                      # mirror: a left hand has its thumb on the other side
        dx, dy = dx * hand_px * scale, dy * hand_px * scale
        x = origin[0] + dx * ca - dy * sa
        y = origin[1] + dx * sa + dy * ca
        lms.append((i, int(round(x)), int(round(y)), 0.0))
    return {"label": label, "landmarks": lms}


class Clock:
    """Controllable replacement for time.time / time.monotonic."""
    def __init__(self, t=1000.0):
        self.t = t
    def __call__(self):
        return self.t
    def advance(self, dt):
        self.t += dt


@pytest.fixture
def clock(monkeypatch):
    import time
    c = Clock()
    monkeypatch.setattr(time, "time", c)
    monkeypatch.setattr(time, "monotonic", c)
    return c
