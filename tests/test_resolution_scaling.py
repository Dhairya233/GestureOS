"""Zones are fractions of the real frame, so 320x240 / 1280x720 / 1920x1080 / phone
portrait all behave like 640x480."""
import numpy as np
import pytest

from conftest import make_hand
from core import os_input
from features.canvas_pen import CanvasController
from features.dynamic_scroller import DynamicScroller
from features.media_control import MediaController
from features.mouse_control import MouseController
from features.tab_switcher import EdgeJoystick

SIZES = [(320, 240), (640, 480), (1280, 720), (1920, 1080), (480, 640)]


def hand_at(w, h, fx, fy, pose="open", label="Right", scale=None):
    """Hand whose middle knuckle (landmark 9) sits at fraction (fx, fy) of the frame."""
    sc = scale or h / 480
    hand = make_hand(label, pose=pose, scale=sc, origin=(0, 0))
    ox, oy = round(fx * w) - hand["landmarks"][9][1], round(fy * h) - hand["landmarks"][9][2]
    hand["landmarks"] = [(i, x + ox, y + oy, z) for i, x, y, z in hand["landmarks"]]
    return [hand]


@pytest.mark.parametrize("w,h", SIZES)
def test_scroller_band_is_proportional(pg, w, h):
    os_input.ARMING.set_armed(True)
    s = DynamicScroller(frame_h=h)
    assert s._upper_boundary == pytest.approx(0.396 * h, abs=2)
    assert s._lower_boundary == pytest.approx(0.604 * h, abs=2)
    s.update(hand_at(w, h, 0.5, 0.5))            # centre: inside the band
    assert not [c for c in pg.calls if c[0] == "scroll"]
    s.update(hand_at(w, h, 0.5, 0.05))           # near top: scroll up
    s.update(hand_at(w, h, 0.5, 0.95))           # near bottom: scroll down
    sc = [c[1][0] for c in pg.calls if c[0] == "scroll"]
    assert sc[0] > 0 and sc[1] < 0


def test_scroll_amount_is_resolution_independent(pg):
    os_input.ARMING.set_armed(True)
    amounts = []
    for w, h in [(640, 480), (1280, 960)]:
        pg.calls.clear()
        s = DynamicScroller(frame_h=h, scroll_speed_modifier=0.3)
        s.update(hand_at(w, h, 0.5, 0.05))
        amounts.append(pg.calls[0][1][0])
    assert abs(amounts[0] - amounts[1]) <= 1


@pytest.mark.parametrize("w,h", SIZES)
def test_edge_joystick_zones(pg, w, h):
    os_input.ARMING.set_armed(True)
    e = EdgeJoystick(frame_w=w)
    assert e._left_zone_max == round(0.25 * w) and e._right_zone_min == round(0.75 * w)
    e.update(hand_at(w, h, 0.5, 0.5))
    assert not [c for c in pg.calls if c[0] == "hotkey"]
    e.update(hand_at(w, h, 0.9, 0.5))
    assert ("hotkey", ("alt", "tab")) in pg.calls
    e.set_frame_size(w * 2)                       # camera changed
    assert e._right_zone_min == round(0.75 * w * 2)


@pytest.mark.parametrize("w,h", SIZES)
def test_media_volume_zones(pg, w, h):
    os_input.ARMING.set_armed(True)
    m = MediaController()
    for fx, key in ((0.9, "volumeup"), (0.1, "volumedown")):
        pg.calls.clear(); m.last_volume_time = 0
        hand = make_hand("Left", pose="point", pinch=0.1, scale=h / 480, origin=(0, 0))
        tip_x = hand["landmarks"][8][1]
        hand["landmarks"] = [(i, x - tip_x + round(fx * w), y + 100, z) for i, x, y, z in hand["landmarks"]]
        m.update([hand], enable_media=False, frame_width=w)
        assert ("press", (key,)) in pg.calls, (w, h, fx)
    pg.calls.clear(); m.last_volume_time = 0
    hand = make_hand("Left", pose="point", pinch=0.1, scale=h / 480, origin=(0, 0))
    tip_x = hand["landmarks"][8][1]
    hand["landmarks"] = [(i, x - tip_x + round(0.5 * w), y + 100, z) for i, x, y, z in hand["landmarks"]]
    m.update([hand], enable_media=False, frame_width=w)
    assert not [c for c in pg.calls if c[0] == "press"]


@pytest.mark.parametrize("w,h", SIZES)
def test_canvas_palette_scales_and_selects_colour(w, h):
    c = CanvasController()
    frame = np.zeros((h, w, 3), np.uint8)
    edges = c.zone_edges(w)
    assert edges[0] == 0 and edges[-1] == w and len(edges) == 6
    layout = c.palette_layout(w, h)
    assert [l[3] for l in layout] == ["BLUE", "GREEN", "RED", "YELLOW", "CLEAR"]
    hh = c.header_height(h)
    assert hh == pytest.approx(h * 70 / 480, abs=1)
    # hover pose (index+middle up), fingertip inside the strip, over each zone
    for zone in range(4):
        hand = make_hand("Right", pose="peace", scale=h / 480, origin=(0, 0))
        tx, ty = hand["landmarks"][8][1], hand["landmarks"][8][2]
        cx = (edges[zone] + edges[zone + 1]) // 2
        hand["landmarks"] = [(i, x - tx + cx, y - ty + hh // 2, z) for i, x, y, z in hand["landmarks"]]
        c.update([hand], frame)
        assert c.draw_color == c.colors[zone], (w, h, zone)


def test_canvas_resizes_with_the_frame():
    c = CanvasController()
    c.update(None, np.zeros((480, 640, 3), np.uint8))
    c.update(None, np.zeros((720, 1280, 3), np.uint8))
    assert c.canvas.shape == (720, 1280, 3)


@pytest.mark.parametrize("sc", [0.4, 1.0, 2.5])
def test_mouse_pinch_click_works_at_any_distance(pg, sc):
    os_input.ARMING.set_armed(True)
    m = MouseController()
    near = make_hand("Right", pose="point", scale=sc)
    l = list(near["landmarks"])
    # thumb tip onto middle tip => click
    l[4] = (4, l[12][1], l[12][2], 0.0)
    m.update([{"label": "Right", "landmarks": l}], 640, 480)
    assert ("click", ()) in pg.calls
