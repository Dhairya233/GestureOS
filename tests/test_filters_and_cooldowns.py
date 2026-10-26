"""Stability filters (hold N frames), cooldowns and hysteresis, driven with a fake clock."""
import math

import pytest

from conftest import make_hand
from core import os_input
from features.app_launcher import AppLauncher
from features.media_control import MediaController
from features.mouse_control import MouseController
from features.system_control import SystemController
from features.tab_switcher import EdgeJoystick

CFG = {"app_paths": {"2_fingers": "https://two.example", "3_fingers": "https://three.example"}}


@pytest.fixture
def armed(pg):
    os_input.ARMING.set_armed(True)
    return pg


# ── app launcher: hold N identical frames, then cooldown ───────────────────
@pytest.fixture
def launched(monkeypatch):
    from features.app_launcher import launcher_handler as lh
    calls = []
    monkeypatch.setattr(lh, "launch", lambda t: calls.append(t) or t)
    return calls


# The peace sign belongs to the media controller, so these timing tests use media-off config.
NOMEDIA = dict(CFG, enable_media=False)


def test_launcher_needs_the_full_hold_and_fires_once(armed, launched, clock):
    a = AppLauncher()
    two = [make_hand("Left", pose="peace")]
    for _ in range(a.target_hold_threshold):      # hold_frames reaches threshold on frame N+1
        a.update(two, NOMEDIA)
    assert launched == []
    a.update(two, NOMEDIA)
    assert launched == ["https://two.example"]
    for _ in range(40):                           # keep holding: no repeats
        a.update(two, NOMEDIA)
    assert len(launched) == 1


def test_launcher_resets_when_the_count_changes(armed, launched, clock):
    a = AppLauncher()
    two, three = [make_hand("Left", pose="peace")], [make_hand("Left", pose="three")]
    for _ in range(10):
        a.update(two, CFG)
    a.update(three, CFG)                          # count changed -> counter restarts
    for _ in range(10):
        a.update(two, CFG)
    assert launched == []


def test_launcher_cooldown_blocks_second_launch(armed, launched, clock):
    a = AppLauncher()
    two = [make_hand("Left", pose="peace")]
    for _ in range(a.target_hold_threshold + 1):
        a.update(two, NOMEDIA)
    assert len(launched) == 1
    a.update(None, NOMEDIA)                           # hand away, then hold again quickly
    for _ in range(a.target_hold_threshold + 1):
        a.update(two, NOMEDIA)
    assert len(launched) == 1                     # still inside the 3 s cooldown
    clock.advance(a.cooldown_period + 0.1)
    a.update(None, NOMEDIA)
    for _ in range(a.target_hold_threshold + 1):
        a.update(two, NOMEDIA)
    assert len(launched) == 2


def test_launcher_ignores_the_right_hand(armed, launched, clock):
    a = AppLauncher()
    for _ in range(40):
        a.update([make_hand("Right", pose="peace")], CFG)
    assert launched == []


# ── media: play/pause cooldown, mute, volume repeat ────────────────────────
def test_play_pause_cooldown(armed, clock):
    m = MediaController()
    peace = [make_hand("Left", pose="peace")]
    m.update(peace, enable_volume=False)
    m.update(peace, enable_volume=False)          # same instant: blocked
    assert [c for c in armed.calls if c == ("press", ("playpause",))] == [("press", ("playpause",))]
    clock.advance(1.6)
    m.update(peace, enable_volume=False)
    assert len([c for c in armed.calls if c[0] == "press"]) == 2


def test_mute_only_for_pointing(armed, clock):
    m = MediaController()
    m.update([make_hand("Left", pose="peace")], enable_media=False)
    assert ("press", ("volumemute",)) not in armed.calls
    m.update([make_hand("Left", pose="point")], enable_media=False)
    assert ("press", ("volumemute",)) in armed.calls


def test_volume_repeat_rate(armed, clock):
    m = MediaController()
    h = make_hand("Left", pose="point", pinch=0.1, origin=(0, 0))
    dx = 600 - h["landmarks"][8][1]               # index tip in the right zone of 640
    h["landmarks"] = [(i, x + dx, y + 100, z) for i, x, y, z in h["landmarks"]]
    for _ in range(5):                            # 5 frames in the same instant -> 1 event
        m.update([h], enable_media=False, frame_width=640)
    assert [c for c in armed.calls if c == ("press", ("volumeup",))] == [("press", ("volumeup",))]
    clock.advance(m.volume_repeat_sec + 0.01)
    m.update([h], enable_media=False, frame_width=640)
    assert len([c for c in armed.calls if c == ("press", ("volumeup",))]) == 2


# ── mouse: click cooldowns and drag hysteresis ─────────────────────────────
def hand_with_pinch(other_tip, ratio, sc=1.0):
    """Right 'point' hand with the thumb tip `ratio` hand-sizes from landmark other_tip."""
    h = make_hand("Right", pose="point", scale=sc)
    l = list(h["landmarks"])
    tx, ty = l[other_tip][1], l[other_tip][2]
    scale_px = math.hypot(l[9][1] - l[0][1], l[9][2] - l[0][2])
    l[4] = (4, int(tx + ratio * scale_px), ty, 0.0)
    return [{"label": "Right", "landmarks": l}]


def test_left_click_cooldown(armed, clock):
    m = MouseController()
    for _ in range(3):
        m.update(hand_with_pinch(12, 0.05), 640, 480)
    assert len([c for c in armed.calls if c[0] == "click"]) == 1
    clock.advance(0.6)
    m.update(hand_with_pinch(12, 0.05), 640, 480)
    assert len([c for c in armed.calls if c[0] == "click"]) == 2


def test_right_click_cooldown(armed, clock):
    m = MouseController()
    for _ in range(3):
        m.update(hand_with_pinch(16, 0.05), 640, 480)
    assert len([c for c in armed.calls if c[0] == "rightClick"]) == 1
    clock.advance(0.7)
    m.update(hand_with_pinch(16, 0.05), 640, 480)
    assert len([c for c in armed.calls if c[0] == "rightClick"]) == 2


def test_drag_hysteresis_no_flicker(armed, clock):
    m = MouseController()
    seq = [0.40, 0.10, 0.22, 0.26, 0.30, 0.40]    # engage <0.19, release >0.27
    states = []
    for r in seq:
        m.update(hand_with_pinch(8, r), 640, 480)
        states.append(m.is_dragging)
    assert states == [False, True, True, True, False, False]
    downs = [c for c in armed.calls if c[0] == "mouseDown"]
    ups = [c for c in armed.calls if c[0] == "mouseUp"]
    assert len(downs) == 1 and len(ups) == 1


def test_drag_released_when_hand_disappears(armed, clock):
    m = MouseController()
    m.update(hand_with_pinch(8, 0.05), 640, 480)
    assert m.is_dragging
    m.update(None, 640, 480)
    assert not m.is_dragging and ("mouseUp", ()) in armed.calls


# ── kill switch: hold N frames, any bad frame resets ───────────────────────
def test_kill_switch_hold_and_reset():
    k = SystemController()
    k.required_frames = 5
    up = [make_hand("Right", pose="thumbs_up")]
    assert [k.update(up) for _ in range(5)] == [False, False, False, False, True]
    k.update(up); k.update(up)
    k.update([make_hand("Right", pose="fist")])   # not a thumbs-up -> reset
    assert k.confirmation_frames == 0
    assert [k.update(up) for _ in range(4)] == [False] * 4


def test_kill_switch_needs_thumb_up_on_screen():
    k = SystemController(); k.required_frames = 1
    assert k.update([make_hand("Right", pose="thumbs_up")]) is True
    k.confirmation_frames = 0
    sideways = [make_hand("Right", pose="thumbs_up", rotation=math.pi / 2)]
    assert k.update(sideways) is False            # intentional: thumb must point up
    k.confirmation_frames = 0
    tilted_ok = [make_hand("Right", pose="thumbs_up", rotation=math.radians(30))]
    assert k.update(tilted_ok) is True            # a natural 30 degree tilt still counts
    k.confirmation_frames = 0
    assert k.update([make_hand("Right", pose="thumbs_up", rotation=math.radians(70))]) is False
    k.confirmation_frames = 0
    assert k.update([make_hand("Right", pose="fist")]) is False
    assert k.update([make_hand("Left", pose="thumbs_up")]) is False   # right hand only


# ── window switch: first fire on entry, repeat only after the hold ─────────
def test_edge_joystick_timing(armed, clock):
    e = EdgeJoystick(frame_w=640)
    def palm(fx):
        h = make_hand("Right", pose="open", origin=(0, 0))
        dx = round(fx * 640) - h["landmarks"][9][1]
        h["landmarks"] = [(i, x + dx, y + 200, z) for i, x, y, z in h["landmarks"]]
        return [h]
    tabs = lambda: len([c for c in armed.calls if c[0] == "hotkey"])
    e.update(palm(0.95)); assert tabs() == 1      # entering the zone fires once
    clock.advance(1.0); e.update(palm(0.95)); assert tabs() == 1      # before the 1.5 s hold
    clock.advance(0.6); e.update(palm(0.95)); assert tabs() == 2      # after it: cycling starts
    clock.advance(0.5); e.update(palm(0.95)); assert tabs() == 2      # inside 1.2 s interval
    clock.advance(0.8); e.update(palm(0.95)); assert tabs() == 3
    e.update(palm(0.5))                                               # back to centre
    assert e.state == "CENTER"
