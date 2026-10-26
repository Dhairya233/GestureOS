"""
ml/features.py
--------------
Turn 21 landmarks into a feature vector.  Pure numpy, vectorised.

Steps, and why each one exists
------------------------------
1. Translate so the wrist is the origin.        -> where the hand is in the frame
                                                   must not matter.
2. Divide by hand size (wrist -> middle knuckle). -> distance from the camera
                                                   must not matter.
3. Mirror left hands to look like right hands.    -> one model serves both hands
                                                   and each class gets 2x the data.
4. (optional) rotate so wrist -> middle knuckle points straight up.
                                                 -> tilting the hand must not
                                                   matter.  Switchable so the
                                                   evaluation can show what it buys.
5. Features = the 40 normalised x,y of landmarks 1..20, plus the 10 pairwise
   distances between the five fingertips.  The coordinates describe the shape;
   the tip distances are rotation-invariant and make "thumb touches index"
   (a pinch) easy to read.

Depth (z) is not used: MediaPipe's z is the noisiest coordinate, and neither the
rule-based classifier nor the app's geometry uses it.
"""

from __future__ import annotations

import math

import numpy as np

TIPS = (4, 8, 12, 16, 20)
_TIP_PAIRS = [(a, b) for i, a in enumerate(TIPS) for b in TIPS[i + 1:]]
N_FEATURES = 40 + len(_TIP_PAIRS)          # 50


def lms_to_points(lms) -> np.ndarray:
    """App landmark tuples [(id, x_px, y_px, z), ...] -> (21, 2) float array."""
    return np.array([(l[1], l[2]) for l in lms], dtype=np.float64)


def normalise(points: np.ndarray, mirror: np.ndarray, rotate: bool = True) -> np.ndarray:
    """points (n, 21, 2), mirror (n,) bool -> normalised (n, 21, 2)."""
    p = points - points[:, :1, :]
    p[mirror, :, 0] *= -1.0
    scale = np.maximum(np.linalg.norm(p[:, 9, :], axis=1), 1e-6)
    p = p / scale[:, None, None]
    if rotate:
        # Rotate by theta so the wrist->middle-knuckle vector (vx, vy) becomes (0, -r),
        # i.e. straight up in image coordinates.  With x' = x c - y s, y' = x s + y c
        # that needs cos(theta) = -vy/r and sin(theta) = -vx/r.
        vx, vy = p[:, 9, 0], p[:, 9, 1]
        theta = np.arctan2(-vx, -vy)                 # 0 when already pointing up
        c, s = np.cos(theta), np.sin(theta)
        x = p[:, :, 0] * c[:, None] - p[:, :, 1] * s[:, None]
        y = p[:, :, 0] * s[:, None] + p[:, :, 1] * c[:, None]
        p = np.stack([x, y], axis=2)
    return p


def feature_matrix(points: np.ndarray, hands, rotate: bool = True) -> np.ndarray:
    """points (n, 21, 2), hands: sequence of 'Left'/'Right' -> (n, N_FEATURES)."""
    mirror = np.asarray([h == "Left" for h in hands], dtype=bool)
    p = normalise(np.array(points, dtype=np.float64), mirror, rotate)
    coords = p[:, 1:, :].reshape(len(p), -1)
    tips = p[:, list(TIPS), :]
    dists = np.stack([np.linalg.norm(tips[:, TIPS.index(a)] - tips[:, TIPS.index(b)], axis=1)
                      for a, b in _TIP_PAIRS], axis=1)
    return np.concatenate([coords, dists], axis=1)


def features_for_hand(lms, label: str, rotate: bool = True) -> np.ndarray:
    """One hand from the app -> (N_FEATURES,) vector."""
    return feature_matrix(lms_to_points(lms)[None], [label], rotate)[0]


def rotate_points(points: np.ndarray, degrees: float, about: int = 0) -> np.ndarray:
    """Rotate raw (n, 21, 2) pixel points about a landmark (test / robustness helper)."""
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    o = points[:, about:about + 1, :]
    d = points - o
    x = d[..., 0] * c - d[..., 1] * s
    y = d[..., 0] * s + d[..., 1] * c
    return np.stack([x, y], axis=-1) + o
