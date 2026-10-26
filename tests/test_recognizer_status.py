import pytest

from core import os_input
from core.recognizer import describe_hands, recognize
from core.status import StatusBoard, Timeline
from core.gestures_catalog import CATALOG
from core.poses import POSES
from conftest import make_hand


@pytest.mark.parametrize("pose,expected", [
    ("open", "Open palm"), ("fist", "Fist"), ("point", "Point"),
    ("peace", "Two fingers"), ("three", "Three fingers"),
    ("four", "Four fingers"), ("thumbs_up", "Thumbs up"),
])
@pytest.mark.parametrize("label", ["Left", "Right"])
def test_recognize_poses(pose, expected, label):
    assert recognize(make_hand(label, pose=pose)["landmarks"]) == expected


@pytest.mark.parametrize("rotation", [0, 30, -30, 90])
def test_recognize_rotation_invariant_for_palm(rotation):
    assert recognize(make_hand("Right", rotation=rotation, pose="open")["landmarks"]) == "Open palm"


def test_recognize_pinch():
    h = make_hand("Right", pose="point", pinch=0.1)
    assert recognize(h["landmarks"]) == "Pinch (index)"


def test_recognize_handles_empty():
    assert recognize(None) == "Unknown"
    assert describe_hands(None) == {}


def test_describe_hands_labels():
    hands = [make_hand("Left", pose="fist"), make_hand("Right", pose="open")]
    assert describe_hands(hands) == {"Left": "Fist", "Right": "Open palm"}


def test_timeline_since_and_bounds():
    t = Timeline(maxlen=3)
    for i in range(5):
        t.add("system", str(i))
    assert [e.text for e in t.since(0)] == ["2", "3", "4"]
    assert [e.text for e in t.since(4)] == ["4"]


def test_timeline_blocked_is_throttled():
    now = [0.0]
    t = Timeline(clock=lambda: now[0])
    t.add_blocked("Click"); t.add_blocked("Click")
    assert len(t.since(0)) == 1
    now[0] = 2.0
    t.add_blocked("Click")
    assert len(t.since(0)) == 2


def test_status_snapshot_is_a_copy():
    b = StatusBoard()
    b.update(gestures={"Left": "Fist"}, armed=True)
    snap = b.snapshot()
    snap.gestures["Left"] = "x"
    assert b.snapshot().gestures["Left"] == "Fist"


def test_os_input_records_actions_and_blocks(pg):
    from core.status import TIMELINE
    TIMELINE.clear()
    os_input.ARMING.set_armed(False)
    os_input.click()
    assert TIMELINE.since(0)[-1].kind == "blocked"
    os_input.ARMING.set_armed(True)
    os_input.click()
    assert TIMELINE.since(0)[-1].kind == "action"


def test_catalog_is_consistent():
    from config_manager import DEFAULTS
    for g in CATALOG:
        assert g.pose in POSES
        assert g.flag == "" or g.flag in DEFAULTS, g.flag
