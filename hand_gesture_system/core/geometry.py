"""
core/geometry.py
----------------
Pure hand-geometry helpers: no camera, no OpenCV, no OS calls.

Landmarks are the tuples produced by HandTracker: ``(id, x_px, y_px, z)``.

Why this module exists
----------------------
The first version of the controllers asked "is the fingertip *higher on the
screen* than its PIP joint" (``tip.y < pip.y``) and "is this distance under 25
pixels".  Both silently assume an upright hand at one particular distance from
the camera.  Here every test compares *distances between landmarks*:

* Finger extended / curled   -> tip-to-wrist vs joint-to-wrist distance.
  Rotating the hand does not change distances, so the answer does not change.
* Pinch                      -> tip-to-tip distance **divided by hand size**.
  Moving the hand closer to / farther from the camera scales both numbers
  equally, so the ratio stays put.
* Zones (scroll band, edge zones, colour strip) -> fractions of the real frame
  size, never 640x480 constants.

Things that are *meant* to depend on screen direction stay that way and are
documented where they are used (the thumbs-up quit gesture needs the thumb to
point up; screen zones are positions on screen).
"""

from __future__ import annotations

import math

WRIST = 0
THUMB_IP, THUMB_TIP = 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

# finger -> (mcp, pip, tip)
FINGERS = {
    "index": (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_TIP),
}
FINGER_NAMES = tuple(FINGERS)

# Reference hand size (wrist -> middle knuckle, pixels) at 640x480 and roughly
# arm's length.  The ratio thresholds below were derived from the old fixed
# pixel thresholds by dividing by this number; they are starting points to be
# tuned on a real camera (see docs/MANUAL_TEST_CHECKLIST.md).
REFERENCE_HAND_SCALE_PX = 130.0

# Pinch thresholds as a fraction of hand size.
PINCH_ENGAGE = 25.0 / REFERENCE_HAND_SCALE_PX      # ~0.19  start (drag)
PINCH_RELEASE = 35.0 / REFERENCE_HAND_SCALE_PX     # ~0.27  end   (drag, hysteresis)
PINCH_CLICK = 30.0 / REFERENCE_HAND_SCALE_PX       # ~0.23  click / volume pinch
PINCH_TRANSFORM = 45.0 / REFERENCE_HAND_SCALE_PX   # ~0.35  two-hand transformer


def dist(lms, a: int, b: int) -> float:
    """Euclidean distance (pixels) between landmarks a and b."""
    _, ax, ay, _ = lms[a]
    _, bx, by, _ = lms[b]
    return math.hypot(ax - bx, ay - by)


def hand_scale(lms) -> float:
    """Wrist to middle-finger knuckle: a stable size proxy for the hand."""
    return max(dist(lms, WRIST, MIDDLE_MCP), 1e-6)


def pinch_ratio(lms, a: int, b: int) -> float:
    """Distance between landmarks a and b as a fraction of hand size."""
    return dist(lms, a, b) / hand_scale(lms)


def finger_extended(lms, finger: str, margin: float = 1.0) -> bool:
    """True when the fingertip is farther from the wrist than its PIP joint.

    A straight finger puts the tip well beyond the PIP; a curled finger folds
    the tip back toward the palm.  Works at any hand rotation.
    """
    _, pip, tip = FINGERS[finger]
    return dist(lms, tip, WRIST) > dist(lms, pip, WRIST) * margin


def finger_curled_tight(lms, finger: str, margin: float = 1.0) -> bool:
    """True when the fingertip has folded in closer to the wrist than the
    finger's base knuckle (MCP): a tightly closed finger, as in a fist."""
    mcp, _, tip = FINGERS[finger]
    return dist(lms, tip, WRIST) < dist(lms, mcp, WRIST) * margin


def thumb_extended(lms) -> bool:
    """True when the thumb tip is farther from the pinky knuckle than the
    thumb's IP joint is (the thumb is swung out, not folded across the palm).
    Independent of left/right hand and of rotation."""
    return dist(lms, THUMB_TIP, PINKY_MCP) > dist(lms, THUMB_IP, PINKY_MCP)


def extended_flags(lms) -> dict:
    """{'index': bool, ..., 'pinky': bool} extension of the four fingers."""
    return {f: finger_extended(lms, f) for f in FINGERS}


def count_extended_fingers(lms) -> int:
    """Number of extended fingers including the thumb (0-5)."""
    return int(thumb_extended(lms)) + sum(extended_flags(lms).values())


def is_open_palm(lms) -> bool:
    """All four fingers and the thumb extended."""
    return thumb_extended(lms) and all(extended_flags(lms).values())


def four_fingers_extended(lms) -> bool:
    """All four fingers extended (thumb ignored), as the scroll / window-switch
    gestures required."""
    return all(extended_flags(lms).values())


def is_fist(lms) -> bool:
    """None of the four fingers extended."""
    return not any(extended_flags(lms).values())


class Hysteresis:
    """Two-threshold switch: turns ON below ``on``, turns OFF above ``off``.

    The gap between the two numbers stops a value hovering near one threshold
    from flickering on and off every frame.
    """

    def __init__(self, on: float, off: float) -> None:
        if off < on:
            raise ValueError("off must be >= on")
        self.on, self.off = on, off
        self.active = False

    def update(self, value: float) -> bool:
        if self.active:
            if value > self.off:
                self.active = False
        elif value < self.on:
            self.active = True
        return self.active

    def reset(self) -> None:
        self.active = False


def fraction_to_px(fraction: float, length: int) -> int:
    """Convert a 0-1 fraction of a frame side to pixels."""
    return int(round(fraction * length))
