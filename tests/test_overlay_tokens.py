import numpy as np
import pytest

from core import overlay, tokens as T
from core.status import GestureAnnouncer, Timeline
from conftest import make_hand
from core.recognizer import describe_hands


@pytest.mark.parametrize("size", [(480, 640), (720, 1280), (360, 480)])
@pytest.mark.parametrize("armed", [True, False])
def test_overlay_draws_at_any_size(size, armed):
    h, w = size
    frame = np.full((h, w, 3), 40, np.uint8)
    hands = [make_hand("Right", pose="point", origin=(w // 2, h - 40), hand_px=h / 5),
             make_hand("Left", pose="three", origin=(w // 4, h - 40), hand_px=h / 5)]
    before = frame.copy()
    overlay.draw(frame, overlay.OverlayContext(
        armed=armed, hands=hands, gestures=describe_hands(hands), fps=30,
        hold_label="Launch app 3: hold", hold_progress=0.5, arm_engaged=True,
        arm_progress=0.5, ctrl_held=True, mouse_rect=100, scroll_lines=(40, h - 40)))
    assert (frame != before).any()


def test_overlay_tolerates_no_hands_and_edge_positions():
    frame = np.zeros((480, 640, 3), np.uint8)
    overlay.draw(frame, overlay.OverlayContext(hands=None))
    far = make_hand("Right", pose="open", origin=(5, 5))     # partly off-frame
    overlay.draw(frame, overlay.OverlayContext(hands=[far], gestures={"Right": "Open palm"}))


def test_palette_strip_draws():
    frame = np.zeros((480, 640, 3), np.uint8)
    layout = [(i * 128, (i + 1) * 128, (50 * i, 80, 200), n)
              for i, n in enumerate(("BLUE", "GREEN", "RED", "YELLOW", "CLEAR"))]
    overlay.draw_palette(frame, layout, 70, 2)
    assert frame[:70].any()


def test_announcer_debounces_and_ignores_unknown():
    tl = Timeline()
    a = GestureAnnouncer(tl, stable_frames=3)
    for pose in ["Fist", "Point", "Fist", "Point", "Fist"]:      # flicker
        a.update({"Right": pose})
    assert tl.since(0) == []
    for _ in range(3):
        a.update({"Right": "Point"})
    assert [e.text for e in tl.since(0)] == ["R · Point"]
    for _ in range(10):
        a.update({"Right": "Point"})                              # no repeat
    for _ in range(5):
        a.update({"Right": "Unknown"})
    assert len(tl.since(0)) == 1
    a.update({})                                                  # hand left
    for _ in range(3):
        a.update({"Right": "Point"})
    assert len(tl.since(0)) == 2                                  # announced again


@pytest.mark.parametrize("fg", [T.TEXT, T.MUTED, T.ARMED, T.ERROR, T.WARN])
@pytest.mark.parametrize("bg", [T.BG, T.SURFACE, T.SURFACE_2])
def test_text_colours_meet_contrast(fg, bg):
    assert T.contrast(fg, bg) >= 4.5, (fg, bg, T.contrast(fg, bg))


def test_faint_border_colour_is_still_visible():
    assert T.contrast(T.FAINT, T.BG) >= 3.0      # still visible as a border
