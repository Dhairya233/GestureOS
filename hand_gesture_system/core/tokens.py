"""
core/tokens.py
--------------
Design tokens for the "Flight Recorder" look: one source of truth shared by the
CustomTkinter UI (ui/theme.py) and the OpenCV overlay (core/overlay.py).

The rule of the design: the interface is monochrome; colour appears ONLY when it
means a state (armed, error, warning).  There is deliberately no brand accent.
"""

from __future__ import annotations

# ── Neutral surfaces and text ──────────────────────────────────────────────
BG = "#0E1013"          # window background
SURFACE = "#14171B"     # panels
SURFACE_2 = "#1A1E23"   # raised / selected
BORDER = "#252A30"
TRACK = "#3A4048"       # slider / switch / meter tracks (3:1 on BG)
TEXT = "#E6E8EB"
MUTED = "#9AA1AA"       # secondary text (>= 4.5:1 on every surface)
FAINT = "#6B727B"       # decoration only (borders of inactive controls), never text

# ── State colours (the only colours in the app) ────────────────────────────
ARMED = "#3DDC97"
ERROR = "#FF6B6B"
WARN = "#F2B84B"

# ── Type scale (points): four sizes only ───────────────────────────────────
SIZE_CAPTION = 11
SIZE_BODY = 13
SIZE_TITLE = 16
SIZE_DISPLAY = 24
FONT_UI = "Segoe UI Variable"      # falls back to Segoe UI on older Windows 10
FONT_MONO = "Consolas"

# ── Spacing scale (px) and shape ───────────────────────────────────────────
SP_1, SP_2, SP_3, SP_4, SP_6 = 4, 8, 12, 16, 24
RADIUS = 6


def hex_to_bgr(h: str) -> tuple:
    """'#RRGGBB' -> (B, G, R) for OpenCV."""
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def _lin(c: float) -> float:
    c /= 255.0
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def luminance(h: str) -> float:
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def contrast(a: str, b: str) -> float:
    """WCAG contrast ratio between two '#RRGGBB' colours (1..21)."""
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)
