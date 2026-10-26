"""
ml/synthetic.py
---------------
SYNTHETIC hands, for checking that the pipeline runs end to end and for tests.

These are NOT a stand-in for real recordings.  Any accuracy measured on this data
says only that the code works; it says nothing about how the classifiers behave
on a real hand and a real camera.  Sessions are given different "styles" (hand
size, tilt range, jitter, how far fingers curl) so a leave-one-session-out split
is not trivially easy, but the world they describe is made up.
"""

from __future__ import annotations

import math

import numpy as np

from core.poses import FINGER_IDS, pose_points

from .dataset import CsvWriter, row_for_hand
from .labels import CLASSES

_POSE_FOR = {"open_palm": "open", "fist": "fist", "point": "point", "two_fingers": "peace",
             "three_fingers": "three", "four_fingers": "four", "thumbs_up": "thumbs_up"}


def _blend(a: dict, b: dict, w: dict) -> dict:
    """Per-landmark blend of two poses; w[id] = weight on `a`."""
    return {i: (w[i] * a[i][0] + (1 - w[i]) * b[i][0], w[i] * a[i][1] + (1 - w[i]) * b[i][1]) for i in a}


def _units(cls: str, rng: np.random.Generator, style: dict) -> dict:
    """One frame's hand in hand units for class `cls`."""
    if cls == "pinch":
        pts = pose_points("point", pinch_to=8, pinch_ratio=float(rng.uniform(0.04, 0.17)))
    elif cls == "none":
        # fingers part-way between open and curled: a relaxed / half-formed hand
        w = {0: 1.0}
        thumb = float(rng.uniform(0.2, 0.8))
        for i in (1, 2, 3, 4):
            w[i] = thumb
        for ids in FINGER_IDS.values():
            f = float(rng.uniform(0.25, 0.75))
            for i in ids:
                w[i] = f
        pts = _blend(pose_points("open"), pose_points("fist"), w)
    else:
        pts = pose_points(_POSE_FOR[cls])
    # how loosely this session's user forms the pose: pull every landmark a little
    # toward the average hand and add per-landmark wobble
    wob = style["wobble"]
    return {i: (x + rng.normal(0, wob), y + rng.normal(0, wob)) for i, (x, y) in pts.items()}


def make_session(session: str, rng: np.random.Generator, n_per_class: int = 60, style=None):
    """Yield (label, handedness, points (21,2) px) for one synthetic sitting."""
    style = style or {
        "hand_px": float(rng.uniform(70, 140)),
        "tilt": float(rng.uniform(5, 40)),
        "jitter_px": float(rng.uniform(0.5, 2.0)),
        "wobble": float(rng.uniform(0.01, 0.05)),
        "dx": float(rng.uniform(-0.05, 0.05)),
    }
    for cls in CLASSES:
        for _ in range(n_per_class):
            hand = "Left" if rng.random() < 0.4 else "Right"
            u = _units(cls, rng, style)
            px = float(style["hand_px"] * rng.uniform(0.75, 1.3))
            rot = math.radians(rng.normal(0, style["tilt"]))
            ca, sa = math.cos(rot), math.sin(rot)
            origin = rng.uniform([150, 250], [500, 420])
            pts = np.zeros((21, 2))
            for i in range(21):
                dx, dy = u[i]
                dx = (-dx if hand == "Left" else dx) + style["dx"]
                dx, dy = dx * px, dy * px
                pts[i] = (origin[0] + dx * ca - dy * sa + rng.normal(0, style["jitter_px"]),
                          origin[1] + dx * sa + dy * ca + rng.normal(0, style["jitter_px"]))
            yield cls, hand, np.round(pts)


def write_session_csv(path: str, session: str, seed: int, n_per_class: int = 60) -> int:
    """Write one synthetic session to `path`; returns the number of rows."""
    rng = np.random.default_rng(seed)
    w = CsvWriter(path)
    try:
        for k, (cls, hand, pts) in enumerate(make_session(session, rng, n_per_class)):
            lms = [(i, int(pts[i, 0]), int(pts[i, 1]), 0.0) for i in range(21)]
            w.write(row_for_hand(cls, session, k / 30.0, {"label": hand, "landmarks": lms}, (640, 480)))
        return w.count
    finally:
        w.close()
