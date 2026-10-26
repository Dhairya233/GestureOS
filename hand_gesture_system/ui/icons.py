"""
ui/icons.py
-----------
Lucide icons (ISC licence, see assets/LICENSE-lucide.txt) shipped as white 96 px
PNGs.  They are tinted at runtime and handed to CustomTkinter as CTkImage, so
they stay sharp on high-DPI screens.
"""

from __future__ import annotations

import os
import tkinter

import customtkinter as ctk
from PIL import Image

_DIR = os.path.join(os.path.dirname(__file__), "assets", "icons")


def available() -> list[str]:
    return sorted(f[:-4] for f in os.listdir(_DIR) if f.endswith(".png"))


_cache: dict = {}
_cache_root = None


def icon(name: str, color: str, size: int = 18) -> ctk.CTkImage:
    """Return icon `name` tinted to `color` ('#RRGGBB') at `size` logical px.

    Images are cached per Tk root: a CTkImage owns Tk image handles that die
    with the window that created them, so a new window starts a fresh cache."""
    global _cache_root
    root = getattr(tkinter, "_default_root", None)
    if root is not _cache_root:
        _cache.clear()
        _cache_root = root
    key = (name, color, size)
    if key not in _cache:
        src = Image.open(os.path.join(_DIR, f"{name}.png")).convert("RGBA")
        tinted = Image.new("RGBA", src.size, color)
        tinted.putalpha(src.getchannel("A"))
        _cache[key] = ctk.CTkImage(light_image=tinted, dark_image=tinted, size=(size, size))
    return _cache[key]
