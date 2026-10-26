"""
ui/theme.py
-----------
The Flight Recorder look for the CustomTkinter UI.  Values come from
core/tokens.py (shared with the camera overlay): a monochrome interface where
colour appears only when it means a state.

Type scale: 4 sizes (caption 11, body 13, title 16, display 24).
Spacing scale: 4 / 8 / 12 / 16 / 24.  One corner radius.
"""

from __future__ import annotations

import customtkinter as ctk

from core import tokens as T

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")   # overridden per widget; no blue is ever used

# colours
BG, SURFACE, SURFACE_2, BORDER, TRACK = T.BG, T.SURFACE, T.SURFACE_2, T.BORDER, T.TRACK
TEXT, MUTED, FAINT = T.TEXT, T.MUTED, T.FAINT
ARMED, ERROR, WARN = T.ARMED, T.ERROR, T.WARN

# spacing / shape
SP_1, SP_2, SP_3, SP_4, SP_6 = T.SP_1, T.SP_2, T.SP_3, T.SP_4, T.SP_6
RADIUS = T.RADIUS

SIDEBAR_W = 184
TOPBAR_H = 48
FEED_INTERVAL = 33          # ms between UI refresh ticks (~30 Hz)

# tabs
TAB_LIVE = "Live"
TAB_GESTURES = "Gestures"
TAB_SETTINGS = "Settings"
TAB_WHITEBOARD = "Whiteboard"
TAB_DIAGNOSTICS = "Diagnostics"
ALL_TABS = [TAB_LIVE, TAB_GESTURES, TAB_SETTINGS, TAB_WHITEBOARD, TAB_DIAGNOSTICS]


def font(kind: str = "body", bold: bool = False, mono: bool = False) -> ctk.CTkFont:
    """kind: caption | body | title | display."""
    size = {"caption": T.SIZE_CAPTION, "body": T.SIZE_BODY,
            "title": T.SIZE_TITLE, "display": T.SIZE_DISPLAY}[kind]
    return ctk.CTkFont(family=T.FONT_MONO if mono else T.FONT_UI, size=size,
                       weight="bold" if bold else "normal")


def hex_blend(fg_hex: str, bg_hex: str, alpha: float) -> str:
    """Blend fg over bg at alpha (0-1) and return #RRGGBB."""
    fg = tuple(int(fg_hex.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    bg = tuple(int(bg_hex.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    return "#{:02X}{:02X}{:02X}".format(*(int(f * alpha + b * (1 - alpha)) for f, b in zip(fg, bg)))
