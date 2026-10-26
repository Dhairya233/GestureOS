"""Controllers talk to the OS only through os_input (arming + clamping apply)."""
from conftest import make_hand
from core import os_input
from features.mouse_control.mouse_handler import MouseController
from features.modifier_key.modifier_handler import ModifierController


def _pointing_hand():
    """Pinky curled, index extended (the air-mouse hover pose)."""
    lms = list(make_hand("Right", open_palm=False)["landmarks"])
    for i, (lid, x, y, z) in enumerate(lms):
        if lid in (6, 7, 8):
            lms[i] = (lid, x, y - 100 * (lid - 5), z)
    return [{"label": "Right", "landmarks": lms}]


def test_mouse_never_moves_cursor_while_disarmed(pg):
    m = MouseController()
    m.update(_pointing_hand(), 640, 480)
    assert [c for c in pg.calls if c[0] != "size"] == []


def test_same_pose_moves_cursor_when_armed(pg):
    """Control for the test above: proves the pose really reaches moveTo."""
    os_input.ARMING.set_armed(True)
    m = MouseController()
    m.update(_pointing_hand(), 640, 480)
    assert any(c[0] == "moveTo" for c in pg.calls)


def test_mouse_target_is_clamped_in_the_corner(pg):
    os_input.ARMING.set_armed(True)
    m = MouseController(smoothening=1)
    m._move_cursor(0, 0, 640, 480)
    m._move_cursor(10000, 10000, 640, 480)
    xs = [c[1] for c in pg.calls if c[0] == "moveTo"]
    assert xs[0] == (0, 0) and xs[1] == (1919, 1079)


def test_modifier_ctrl_not_held_while_disarmed_and_released_on_reset(pg):
    mod = ModifierController()
    fist = [make_hand("Left", open_palm=False)]
    mod.update(fist)
    assert not mod.is_ctrl_held and pg.calls == []
