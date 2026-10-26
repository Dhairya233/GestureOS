"""
ui/feed.py
----------
FeedLabel: shows a stream of RGB numpy frames, scaled to fit the space it is in.

A plain tk.Label with a fresh PhotoImage per frame (the reference is kept on
the widget so it is not garbage-collected). This avoids re-configuring one
shared CTkImage every tick, which is the pattern that can leave a label stuck
on its first frame.
"""

from __future__ import annotations

import tkinter as tk
import zlib

from PIL import Image, ImageTk


class FeedLabel(tk.Label):
    def __init__(self, parent, bg: str) -> None:
        super().__init__(parent, bg=bg, bd=0, highlightthickness=0)
        self._photo = None
        self._box = (640, 360)
        self.frames_shown = 0
        self.last_signature = 0       # checksum of the picture last shown (used by tests)
        parent.bind("<Configure>", self._on_resize, add="+")

    def _on_resize(self, event) -> None:
        self._box = (max(event.width, 64), max(event.height, 64))

    def show(self, frame) -> None:
        h, w = frame.shape[:2]
        bw, bh = self._box
        k = min(bw / w, bh / h)
        pw, ph = max(int(w * k), 1), max(int(h * k), 1)
        img = Image.fromarray(frame).resize((pw, ph), Image.BILINEAR)
        self.last_signature = zlib.crc32(img.tobytes())
        self._photo = ImageTk.PhotoImage(img, master=self)
        self.configure(image=self._photo)
        self.frames_shown += 1

    def clear(self) -> None:
        self._photo = None
        self.configure(image="")
