"""
core/recognizer.py
------------------
Names the pose of ONE hand.  Pure function, display-only: it never triggers
anything.  It exists so the UI can say what the system thinks you are doing
(the controllers keep their own, feature-specific checks).

Order matters: pinches are checked first because a pinch is a deliberate,
specific shape; then finger-count poses; fist last.
"""

from __future__ import annotations

from . import geometry as geo

# Landmark ids of the fingertips the thumb can pinch against.
_PINCH_TARGETS = (("index", 8), ("middle", 12), ("ring", 16))


def thumb_points_up(lms) -> bool:
    """Thumb tip is clearly above the wrist, within 45 degrees of vertical
    (same rule the quit gesture uses, so 'Thumbs up' means the same thing
    everywhere).  Image y grows down."""
    wx, wy = lms[0][1], lms[0][2]
    tx, ty = lms[4][1], lms[4][2]
    dx, dy = tx - wx, wy - ty
    return dy > 0 and abs(dx) <= dy


def recognize(lms) -> str:
    """Return a short human label for the hand pose, or 'Unknown'."""
    if not lms or len(lms) < 21:
        return "Unknown"

    # A pinch needs the pinching finger to be open enough to meet the thumb;
    # a tightly curled finger next to a tucked thumb is a fist, not a pinch.
    for name, tip in _PINCH_TARGETS:
        if (geo.pinch_ratio(lms, 4, tip) < geo.PINCH_CLICK
                and not geo.finger_curled_tight(lms, name)):
            return f"Pinch ({name})"

    flags = geo.extended_flags(lms)
    n_fingers = sum(flags.values())
    thumb = geo.thumb_extended(lms)

    if n_fingers == 4 and thumb:
        return "Open palm"
    if n_fingers == 0:
        if thumb and thumb_points_up(lms):
            return "Thumbs up"
        return "Fist"
    if n_fingers == 1 and flags["index"]:
        return "Point"
    if n_fingers == 2 and flags["index"] and flags["middle"]:
        return "Two fingers"
    if n_fingers == 3 and not flags["pinky"]:
        return "Three fingers"
    if n_fingers == 4:
        return "Four fingers"
    return "Unknown"


def describe_hands(hands) -> dict:
    """{'Left': 'Open palm', 'Right': 'Point'} for the hands in view."""
    out = {}
    for h in hands or []:
        out[h["label"]] = recognize(h["landmarks"])
    return out
