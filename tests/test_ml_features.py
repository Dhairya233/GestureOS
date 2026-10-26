import numpy as np
import pytest

from core.poses import POSES, pose_points
from ml import features as F


def raw_points(pose, hand="Right", px=100.0, rot_deg=0.0, origin=(300, 300)):
    pts = pose_points(pose)
    a = np.array([pts[i] for i in range(21)], dtype=float)
    if hand == "Left":
        a[:, 0] *= -1
    a *= px
    return F.rotate_points((a + origin)[None], rot_deg, about=0)[0]


@pytest.mark.parametrize("pose", list(POSES))
@pytest.mark.parametrize("deg", [15, 45, 90, 135, 180, -70])
def test_rotation_normalisation_makes_features_rotation_invariant(pose, deg):
    base = F.feature_matrix(raw_points(pose)[None], ["Right"])[0]
    rot = F.feature_matrix(raw_points(pose, rot_deg=deg)[None], ["Right"])[0]
    np.testing.assert_allclose(base, rot, atol=1e-6)


@pytest.mark.parametrize("pose", list(POSES))
def test_without_rotation_normalisation_orientation_changes_features(pose):
    a = F.feature_matrix(raw_points(pose)[None], ["Right"], rotate=False)[0]
    b = F.feature_matrix(raw_points(pose, rot_deg=90)[None], ["Right"], rotate=False)[0]
    assert not np.allclose(a, b, atol=1e-3)


@pytest.mark.parametrize("scale", [0.4, 2.5])
def test_features_ignore_scale_and_position(scale):
    a = F.feature_matrix(raw_points("peace")[None], ["Right"])[0]
    b = F.feature_matrix(raw_points("peace", px=100 * scale, origin=(77, 411))[None], ["Right"])[0]
    np.testing.assert_allclose(a, b, atol=1e-6)


@pytest.mark.parametrize("pose", list(POSES))
def test_left_hand_features_equal_the_mirrored_right_hand(pose):
    r = F.feature_matrix(raw_points(pose, "Right")[None], ["Right"])[0]
    l = F.feature_matrix(raw_points(pose, "Left")[None], ["Left"])[0]
    np.testing.assert_allclose(r, l, atol=1e-6)


def test_feature_vector_has_documented_size():
    assert F.N_FEATURES == 50
    assert F.features_for_hand([(i, i, i * 2, 0.0) for i in range(21)], "Right").shape == (50,)
