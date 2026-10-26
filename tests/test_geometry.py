"""core/geometry.py: invariance (rotation, scale) and equivalence with the old
upright-only heuristics for upright hands."""
import math

import pytest

from conftest import POSES, make_hand
from core import geometry as geo

ROTATIONS = [0.0, 0.5, 1.0, math.pi / 2, 2.0, math.pi, -0.7, -2.5]
SCALES = [0.4, 0.7, 1.0, 1.6, 2.5]


def lms(pose, **kw):
    return make_hand(pose=pose, **kw)["landmarks"]


# ── the OLD logic, kept here as the reference for upright hands ────────────
def old_extended(l, tip, pip):
    return l[tip][2] < l[pip][2]


def old_fist(l):
    return all(l[t][2] > l[p][2] for t, p in ((8, 6), (12, 10), (16, 14), (20, 18)))


def old_four_up(l):
    return all(old_extended(l, t, p) for t, p in ((8, 6), (12, 10), (16, 14), (20, 18)))


def old_peace(l):
    return (old_extended(l, 8, 6) and old_extended(l, 12, 10)
            and l[16][2] > l[13][2] and l[20][2] > l[17][2])


def old_pointing(l):
    return (old_extended(l, 8, 6) and l[12][2] > l[9][2]
            and l[16][2] > l[13][2] and l[20][2] > l[17][2])


def old_finger_count(l):          # Left-hand rule: thumb extends to the right
    thumb = l[4][1] > l[3][1]
    return thumb + sum(old_extended(l, t, p) for t, p in ((8, 6), (12, 10), (16, 14), (20, 18)))


# ── equivalence for upright hands ──────────────────────────────────────────
@pytest.mark.parametrize("pose", ["open", "fist", "point", "peace", "three", "four", "thumbs_up"])
def test_matches_old_heuristics_on_upright_hands(pose):
    l = lms(pose)
    assert geo.four_fingers_extended(l) == old_four_up(l)
    assert geo.is_fist(l) == old_fist(l)
    assert (geo.finger_extended(l, "index") and geo.finger_extended(l, "middle")
            and geo.finger_curled_tight(l, "ring") and geo.finger_curled_tight(l, "pinky")) == old_peace(l)
    assert (geo.finger_extended(l, "index") and all(
        geo.finger_curled_tight(l, f) for f in ("middle", "ring", "pinky"))) == old_pointing(l)


@pytest.mark.parametrize("pose", ["open", "four", "three", "peace", "point", "fist"])
def test_finger_count_matches_old_for_upright_left_hand(pose):
    l = lms(pose, label="Left")
    assert geo.count_extended_fingers(l) == old_finger_count(l)


# ── invariance ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("pose", list(POSES))
@pytest.mark.parametrize("rot", ROTATIONS)
@pytest.mark.parametrize("sc", SCALES)
def test_classification_is_rotation_and_scale_invariant(pose, rot, sc):
    base = lms(pose)
    moved = lms(pose, rotation=rot, scale=sc, origin=(320, 240))
    for fn in (geo.is_fist, geo.is_open_palm, geo.four_fingers_extended,
               geo.thumb_extended, geo.count_extended_fingers):
        assert fn(moved) == fn(base), (fn.__name__, pose, rot, sc)
    for f in geo.FINGER_NAMES:
        assert geo.finger_extended(moved, f) == geo.finger_extended(base, f)
        assert geo.finger_curled_tight(moved, f) == geo.finger_curled_tight(base, f)


def test_old_heuristic_fails_on_rotated_hand_new_one_does_not():
    """The bug being fixed: a hand turned sideways looked 'curled' to the old test."""
    sideways = lms("open", rotation=math.pi / 2)
    assert not old_four_up(sideways)         # old logic: wrong
    assert geo.four_fingers_extended(sideways)   # new logic: right


def test_pose_truths():
    assert geo.is_open_palm(lms("open"))
    assert not geo.is_open_palm(lms("four"))          # thumb folded
    assert geo.count_extended_fingers(lms("open")) == 5
    assert geo.count_extended_fingers(lms("peace")) == 2
    assert geo.count_extended_fingers(lms("three")) == 3
    assert geo.count_extended_fingers(lms("four")) == 4
    assert geo.count_extended_fingers(lms("fist")) == 0
    assert geo.is_fist(lms("thumbs_up"))


# ── pinch ratio ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("sc", SCALES)
@pytest.mark.parametrize("rot", [0.0, 1.3, -2.0])
def test_pinch_ratio_is_distance_independent(sc, rot):
    for r in (0.05, 0.15, 0.4, 0.8):
        l = lms("point", pinch=r, scale=sc, rotation=rot)
        assert geo.pinch_ratio(l, 4, 8) == pytest.approx(r, abs=0.03)


def test_fixed_pixel_threshold_was_distance_dependent():
    """Same pinch, hand far from camera vs near: old 25 px rule disagrees with itself."""
    near = lms("point", pinch=0.15, scale=2.5)
    far = lms("point", pinch=0.15, scale=0.4)
    px = lambda l: math.hypot(l[8][1] - l[4][1], l[8][2] - l[4][2])
    assert (px(near) < 25) != (px(far) < 25)             # old rule: inconsistent
    assert (geo.pinch_ratio(near, 4, 8) < geo.PINCH_ENGAGE) == \
           (geo.pinch_ratio(far, 4, 8) < geo.PINCH_ENGAGE)   # new rule: consistent


def test_hysteresis():
    h = geo.Hysteresis(0.2, 0.3)
    seq = [0.5, 0.25, 0.19, 0.25, 0.29, 0.31, 0.25, 0.1]
    assert [h.update(v) for v in seq] == [False, False, True, True, True, False, False, True]
    with pytest.raises(ValueError):
        geo.Hysteresis(0.3, 0.2)
