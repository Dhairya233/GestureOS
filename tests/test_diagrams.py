import pytest

from core.gestures_catalog import CATALOG
from ui import diagrams


@pytest.mark.parametrize("g", CATALOG, ids=lambda g: g.key)
@pytest.mark.parametrize("hand", ["Left", "Right"])
def test_every_catalog_diagram_fits_its_box(g, hand):
    d = diagrams.layout(g.pose, hand, 84, 112, g.pinch_to)
    assert d["lines"] and d["rings"] or g.pose in ("fist",)
    for x1, y1, x2, y2 in d["lines"]:
        for v, lim in ((x1, 84), (x2, 84), (y1, 112), (y2, 112)):
            assert 0 <= v <= lim


def test_left_hand_is_a_mirror_image():
    r = diagrams.layout("open", "Right", 84, 112)
    l = diagrams.layout("open", "Left", 84, 112)
    assert [round(84 - x, 3) for x, _ in r["dots"]] == pytest.approx([round(x, 3) for x, _ in l["dots"]], abs=0.01)
