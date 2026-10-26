"""
core/overlay.py
---------------
Draws the camera-frame overlay in the "Flight Recorder" language: a plain
frame, a thin white skeleton, and colour only where it carries a state
(armed = green, error = red).  Everything scales with the frame height so the
overlay looks the same at 640x480 and 1280x720.

Display only.  It reads controller state; it never changes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2

from . import tokens as T
from .poses import EDGES

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Which landmark ids to ring for each recognised pose (the joints that matter).
_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
RING_JOINTS = {
    "Open palm": (4, 8, 12, 16, 20),
    "Point": (8,),
    "Two fingers": (8, 12),
    "Three fingers": (8, 12, 16),
    "Four fingers": (8, 12, 16, 20),
    "Thumbs up": (4,),
    "Pinch": (4, 8),
    "Pinch (index)": (4, 8),
    "Pinch (middle)": (4, 12),
    "Pinch (ring)": (4, 16),
}


@dataclass
class OverlayContext:
    armed: bool = False
    hands: list | None = None
    gestures: dict = field(default_factory=dict)     # label -> pose name
    arm_progress: float = 0.0                        # 0..1 while both palms held
    arm_engaged: bool = False
    hold_label: str = ""                             # "Launch app 3"
    hold_progress: float = 0.0
    fps: float = 0.0
    ctrl_held: bool = False
    mouse_rect: int | None = None                    # frame_r margin, None = off
    scroll_lines: tuple | None = None                # (upper_y, lower_y) or None


def _scale(frame) -> float:
    return frame.shape[0] / 480.0


def _text_size(text: str, scale: float, thick: int = 1):
    (w, h), base = cv2.getTextSize(text, FONT, scale, thick)
    return w, h, base


def _put(frame, text, org, scale, color, thick=1):
    cv2.putText(frame, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def _pill(frame, x, y, w, h, alpha=0.72):
    """Translucent dark rounded-ish plate (rectangle; radius is cosmetic)."""
    x2, y2 = x + w, y + h
    x, y = max(x, 0), max(y, 0)
    x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
    if x2 <= x or y2 <= y:
        return
    roi = frame[y:y2, x:x2]
    plate = roi.copy()
    plate[:] = T.hex_to_bgr(T.BG)
    cv2.addWeighted(plate, alpha, roi, 1 - alpha, 0, roi)


def draw_skeleton(frame, hands, gestures, armed: bool) -> None:
    k = _scale(frame)
    line = T.hex_to_bgr(T.TEXT)
    ring = T.hex_to_bgr(T.ARMED if armed else T.MUTED)
    for h in hands or []:
        pts = {i: (x, y) for i, x, y, _ in h["landmarks"]}
        for a, b in EDGES:
            cv2.line(frame, pts[a], pts[b], line, max(1, round(1.5 * k)), cv2.LINE_AA)
        for p in pts.values():
            cv2.circle(frame, p, max(2, round(2.5 * k)), line, -1, cv2.LINE_AA)
        for j in RING_JOINTS.get(gestures.get(h["label"], ""), ()):
            cv2.circle(frame, pts[j], max(6, round(9 * k)), ring, max(1, round(2 * k)), cv2.LINE_AA)


def draw_hand_labels(frame, hands, gestures) -> None:
    k = _scale(frame)
    s = 0.45 * k
    for h in hands or []:
        _, wx, wy, _ = h["landmarks"][0]
        text = f"{h['label'][0]}  {gestures.get(h['label'], '')}".strip()
        w, th, base = _text_size(text, s)
        x, y = wx + int(10 * k), wy + int(14 * k)
        _pill(frame, x - 6, y - th - 6, w + 12, th + base + 10)
        _put(frame, text, (x, y), s, T.hex_to_bgr(T.TEXT))


def draw_state_chip(frame, armed: bool, arm_engaged: bool, arm_progress: float) -> None:
    k = _scale(frame)
    s = 0.5 * k
    text = "ARMED" if armed else "DISARMED"
    hint = ""
    if arm_engaged:
        hint = "  hold to disarm" if armed else "  hold to arm"
    full = text + hint
    w, th, base = _text_size(full, s)
    x, y = int(12 * k), int(12 * k)
    pad = int(8 * k)
    dot_r = int(5 * k)
    plate_w = pad * 3 + dot_r * 2 + w
    plate_h = th + base + pad * 2
    _pill(frame, x, y, plate_w, plate_h)
    cx, cy = x + pad + dot_r, y + plate_h // 2
    col = T.hex_to_bgr(T.ARMED if armed else T.MUTED)
    cv2.circle(frame, (cx, cy), dot_r, col, -1 if armed else max(1, round(1.5 * k)), cv2.LINE_AA)
    _put(frame, full, (cx + dot_r + pad, cy + th // 2), s,
         T.hex_to_bgr(T.TEXT if armed else T.MUTED))
    if arm_engaged:
        bar_y = y + plate_h + int(4 * k)
        cv2.rectangle(frame, (x, bar_y), (x + plate_w, bar_y + max(2, int(3 * k))),
                      T.hex_to_bgr(T.BORDER), -1)
        cv2.rectangle(frame, (x, bar_y),
                      (x + int(plate_w * min(max(arm_progress, 0.0), 1.0)), bar_y + max(2, int(3 * k))),
                      T.hex_to_bgr(T.ARMED), -1)


def draw_fps(frame, fps: float) -> None:
    if fps <= 0:
        return
    k = _scale(frame)
    s = 0.42 * k
    text = f"{fps:.0f} fps  {frame.shape[1]}x{frame.shape[0]}"
    w, th, _ = _text_size(text, s)
    _put(frame, text, (frame.shape[1] - w - int(12 * k), int(24 * k)), s, T.hex_to_bgr(T.MUTED))


def draw_hold_bar(frame, label: str, progress: float) -> None:
    if not label:
        return
    k = _scale(frame)
    s = 0.45 * k
    w, th, base = _text_size(label, s)
    bar_w, bar_h = int(200 * k), max(3, int(4 * k))
    x, y = int(12 * k), frame.shape[0] - int(16 * k)
    _pill(frame, x - 6, y - th - bar_h - int(14 * k), max(w, bar_w) + 12, th + bar_h + int(22 * k))
    _put(frame, label, (x, y - bar_h - int(6 * k)), s, T.hex_to_bgr(T.TEXT))
    cv2.rectangle(frame, (x, y - bar_h), (x + bar_w, y), T.hex_to_bgr(T.BORDER), -1)
    cv2.rectangle(frame, (x, y - bar_h),
                  (x + int(bar_w * min(max(progress, 0.0), 1.0)), y), T.hex_to_bgr(T.ARMED), -1)


def draw_ctrl_badge(frame) -> None:
    k = _scale(frame)
    s = 0.5 * k
    text = "CTRL HELD"
    w, th, base = _text_size(text, s)
    x = (frame.shape[1] - w) // 2
    y = frame.shape[0] - int(16 * k)
    _pill(frame, x - 10, y - th - 8, w + 20, th + base + 14)
    _put(frame, text, (x, y), s, T.hex_to_bgr(T.ERROR), 1)


def draw_zones(frame, mouse_rect, scroll_lines) -> None:
    """Very faint guides for the active control zones."""
    layer = frame.copy()
    col = T.hex_to_bgr(T.TEXT)
    k = _scale(frame)
    h, w = frame.shape[:2]
    if mouse_rect is not None:
        cv2.rectangle(layer, (mouse_rect, mouse_rect), (w - mouse_rect, h - mouse_rect),
                      col, 1, cv2.LINE_AA)
    if scroll_lines:
        for y in scroll_lines:
            cv2.line(layer, (0, y), (w, y), col, 1, cv2.LINE_AA)
    cv2.addWeighted(layer, 0.28, frame, 0.72, 0, frame)
    s = 0.36 * k
    if mouse_rect is not None:
        _put(frame, "cursor area", (mouse_rect + 4, mouse_rect - 6), s, T.hex_to_bgr(T.MUTED))
    if scroll_lines:
        w_, _, _ = _text_size("scroll", s)
        _put(frame, "scroll", (w - w_ - 6, scroll_lines[0] - 5), s, T.hex_to_bgr(T.MUTED))


def draw_palette(frame, layout, header_h: int, active_index: int) -> None:
    """Canvas colour strip: dark band with one swatch per zone.
    layout = [(x1, x2, bgr_colour, label), ...] from CanvasController."""
    k = _scale(frame)
    _pill(frame, 0, 0, frame.shape[1], header_h, alpha=0.78)
    s = 0.4 * k
    for i, (x1, x2, col, label) in enumerate(layout):
        cx = (x1 + x2) // 2
        cy = header_h // 2 - int(6 * k)
        r = int(14 * k)
        if label.strip().upper() in ("CLEAR", "ERASE", "ERASER"):
            cv2.circle(frame, (cx, cy), r, T.hex_to_bgr(T.MUTED), max(1, round(1.5 * k)), cv2.LINE_AA)
        else:
            cv2.circle(frame, (cx, cy), r, col, -1, cv2.LINE_AA)
        w, th, _ = _text_size(label, s)
        _put(frame, label, (cx - w // 2, header_h - int(8 * k)), s,
             T.hex_to_bgr(T.TEXT if i == active_index else T.MUTED))
        if i == active_index:
            cv2.circle(frame, (cx, cy), r + int(4 * k), T.hex_to_bgr(T.TEXT), max(1, round(1.5 * k)), cv2.LINE_AA)


def draw(frame, ctx: OverlayContext) -> None:
    """Everything except the canvas palette and feature-specific panels."""
    if ctx.mouse_rect is not None or ctx.scroll_lines:
        draw_zones(frame, ctx.mouse_rect, ctx.scroll_lines)
    draw_skeleton(frame, ctx.hands, ctx.gestures, ctx.armed)
    draw_hand_labels(frame, ctx.hands, ctx.gestures)
    draw_state_chip(frame, ctx.armed, ctx.arm_engaged, ctx.arm_progress)
    draw_fps(frame, ctx.fps)
    draw_hold_bar(frame, ctx.hold_label, ctx.hold_progress)
    if ctx.ctrl_held:
        draw_ctrl_badge(frame)
