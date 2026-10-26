"""
ui/diagrams.py
--------------
Skeleton diagrams for the Gestures tab, computed from core.poses (the same
model the tests use), so what the cheat-sheet shows is what the recogniser is
tested against.

`layout()` is pure (no Tk) and unit-tested; `draw()` paints it on a tk.Canvas.
"""

from __future__ import annotations

from core.poses import EDGES, FINGER_IDS, POSES, pose_points

_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}


def layout(pose: str, hand: str, width: int, height: int, pinch_to: int | None = None,
           pad: int = 8):
    """Return {'lines': [(x1,y1,x2,y2)], 'dots': [(x,y)], 'rings': [(x,y)]} in
    canvas pixels for a width x height box.  hand='Left' mirrors the drawing."""
    pts = pose_points(pose, pinch_to=pinch_to, pinch_ratio=0.10)
    sign = -1.0 if hand == "Left" else 1.0
    xs = [sign * p[0] for p in pts.values()]
    ys = [p[1] for p in pts.values()]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    k = min((width - 2 * pad) / max(w, 1e-6), (height - 2 * pad) / max(h, 1e-6))
    ox = (width - w * k) / 2 - min(xs) * k
    oy = (height - h * k) / 2 - min(ys) * k

    def px(i):
        x, y = pts[i]
        return (sign * x * k + ox, y * k + oy)

    lines = [(*px(a), *px(b)) for a, b in EDGES]
    dots = [px(i) for i in pts]

    extended, _ = POSES[pose]
    ring_ids = [_TIPS[f] for f in FINGER_IDS if f in extended]
    if pinch_to is not None:
        ring_ids = [4, pinch_to]
    elif pose == "thumbs_up":
        ring_ids = [4]
    return {"lines": lines, "dots": dots, "rings": [px(i) for i in ring_ids]}


def draw(canvas, pose: str, hand: str, width: int, height: int, *, line: str, ring: str,
         pinch_to: int | None = None) -> None:
    """Paint a diagram on a Tk canvas of width x height."""
    canvas.delete("all")
    d = layout(pose, hand, width, height, pinch_to)
    for x1, y1, x2, y2 in d["lines"]:
        canvas.create_line(x1, y1, x2, y2, fill=line, width=2.5, capstyle="round")
    for x, y in d["dots"]:
        canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=line, outline=line)
    for x, y in d["rings"]:
        canvas.create_oval(x - 7, y - 7, x + 7, y + 7, outline=ring, width=2)
