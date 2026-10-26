"""Render the camera overlay on synthetic frames (no camera needed).

    python tools/render_overlay.py OUTDIR
"""
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hand_gesture_system"))
from core import overlay                       # noqa: E402
from core.poses import pose_points               # noqa: E402
from core.recognizer import describe_hands       # noqa: E402


def hand(label, pose, origin, px=110.0, rot=0.0, pinch=None):
    pts = pose_points(pose)
    if pinch is not None:
        pts[4] = (pts[8][0] + pinch, pts[8][1])
    lms = []
    for i in range(21):
        dx, dy = pts[i]
        if label == "Left":
            dx = -dx
        dx, dy = dx * px, dy * px
        lms.append((i, int(origin[0] + dx * math.cos(rot) - dy * math.sin(rot)),
                    int(origin[1] + dx * math.sin(rot) + dy * math.cos(rot)), 0.0))
    return {"label": label, "landmarks": lms}


def backdrop(w=640, h=480):
    f = np.full((h, w, 3), (58, 52, 46), np.uint8)
    cv2.GaussianBlur(f, (0, 0), 3, f)
    return f


def shot(name, out, hands, armed, **kw):
    f = backdrop()
    ctx = overlay.OverlayContext(armed=armed, hands=hands, gestures=describe_hands(hands),
                                 fps=29.6, mouse_rect=int(0.35 * 480), scroll_lines=(50, 430), **kw)
    overlay.draw(f, ctx)
    cv2.imwrite(os.path.join(out, name), f)


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    shot("overlay_disarmed.png", out, [hand("Right", "point", (380, 380))], False)
    shot("overlay_armed_point.png", out, [hand("Right", "point", (380, 380))], True)
    shot("overlay_armed_pinch.png", out, [hand("Right", "point", (380, 380), pinch=0.1)], True)
    shot("overlay_launch_hold.png", out,
         [hand("Left", "three", (200, 380)), hand("Right", "open", (440, 380))], True,
         hold_label="Launch app 3: hold", hold_progress=0.6)
    shot("overlay_arming.png", out,
         [hand("Left", "open", (200, 380)), hand("Right", "open", (440, 380))], False,
         arm_engaged=True, arm_progress=0.45)
    print("wrote", out)
