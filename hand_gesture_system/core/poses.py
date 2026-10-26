"""
core/poses.py
-------------
A small idealised hand model in "hand units" (1.0 = wrist to middle-finger
knuckle).  Used to draw the gesture cheat-sheet and to build synthetic hands in
tests, so the diagrams and the tests can never drift apart.

Image coordinates: y grows DOWN, so fingers pointing up have negative y.
Landmark ids follow MediaPipe (0 wrist, 1-4 thumb, 5-8 index, ...).
"""

from __future__ import annotations

SPREAD = {"index": -0.35, "middle": 0.0, "ring": 0.3, "pinky": 0.6}
FINGER_IDS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
              "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
THUMB_OUT = [(-0.9, -0.3), (-1.0, -0.65), (-1.1, -1.0), (-1.2, -1.35)]
THUMB_IN = [(-0.7, -0.3), (-0.6, -0.5), (-0.4, -0.65), (-0.2, -0.8)]
THUMB_UP = [(-0.7, -0.5), (-0.8, -0.9), (-0.85, -1.3), (-0.9, -1.7)]

# name -> (set of extended fingers, thumb chain)
POSES = {
    "open": ({"index", "middle", "ring", "pinky"}, THUMB_OUT),
    "fist": (set(), THUMB_IN),
    "point": ({"index"}, THUMB_IN),
    "peace": ({"index", "middle"}, THUMB_IN),
    "three": ({"index", "middle", "ring"}, THUMB_IN),
    "four": ({"index", "middle", "ring", "pinky"}, THUMB_IN),
    "two_thumb": ({"index", "middle"}, THUMB_OUT),
    "thumbs_up": (set(), THUMB_UP),
}

# 21 landmarks -> skeleton edges (wrist/thumb/fingers)
EDGES = (
    [(0, 1), (1, 2), (2, 3), (3, 4)]
    + [(0, 5), (5, 6), (6, 7), (7, 8)]
    + [(9, 10), (10, 11), (11, 12)]
    + [(13, 14), (14, 15), (15, 16)]
    + [(0, 17), (17, 18), (18, 19), (19, 20)]
    + [(5, 9), (9, 13), (13, 17)]
)


def pose_points(pose: str, pinch_to=None, pinch_ratio: float = 0.12) -> dict:
    """{landmark_id: (x, y)} in hand units for a named pose.

    pinch_to: landmark id (8, 12 or 16) to bring the thumb tip next to, at
    ``pinch_ratio`` hand-sizes away (used for pinch diagrams / tests).
    """
    extended, thumb = POSES[pose]
    pts = {0: (0.0, 0.0)}
    for k, lm in enumerate((1, 2, 3, 4)):
        pts[lm] = thumb[k]
    for name, chain in FINGER_IDS.items():
        x = SPREAD[name]
        for k, lm in enumerate(chain):
            pts[lm] = (x, -1.0 - 0.45 * k) if name in extended else (x, [-1.0, -1.45, -1.15, -0.8][k])
    if pinch_to is not None:
        tx, ty = pts[pinch_to]
        pts[4] = (tx + pinch_ratio, ty)
        pts[3] = (tx + pinch_ratio + 0.25, ty + 0.35)
    return pts
