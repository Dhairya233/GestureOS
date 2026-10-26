"""
ui/views/gestures.py
--------------------
Gestures: the cheat-sheet.  Every entry comes from core.gestures_catalog and is
drawn from core.poses, so it matches what the recogniser is tested against.
"""

from __future__ import annotations

import tkinter as tk

import customtkinter as ctk

from core.gestures_catalog import CATALOG

from .. import diagrams
from .. import theme as th
from ..widgets import Section

_W, _H = 84, 112


class GesturesView(ctk.CTkScrollableFrame):
    def __init__(self, parent, app) -> None:
        super().__init__(parent, fg_color="transparent", scrollbar_button_color=th.BORDER,
                         scrollbar_button_hover_color=th.FAINT)
        self.app = app
        self._state_labels: list[tuple[ctk.CTkLabel, str]] = []
        self.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self, text="Hands are your physical hands as you see them in the mirrored camera "
                       "image. Gestures only act while GestureOS is armed (top bar).",
            font=th.font("body"), text_color=th.MUTED, anchor="w", justify="left",
            wraplength=820).grid(row=0, column=0, sticky="w", padx=th.SP_4, pady=(th.SP_4, th.SP_3))

        for r, hand in enumerate(("Right", "Left", "Both"), start=1):
            sec = Section(self, f"{hand} hand" if hand != "Both" else "Both hands")
            sec.grid(row=r, column=0, sticky="ew", padx=th.SP_4, pady=(0, th.SP_4))
            body = sec.body
            body.grid_columnconfigure(0, weight=1, uniform="g")
            body.grid_columnconfigure(1, weight=1, uniform="g")
            items = [g for g in CATALOG if g.hand == hand]
            for i, g in enumerate(items):
                self._item(body, g).grid(row=i // 2, column=i % 2, sticky="nsew",
                                         padx=(0, th.SP_4), pady=(0, th.SP_3))

    def _item(self, parent, g) -> ctk.CTkFrame:
        f = ctk.CTkFrame(parent, fg_color=th.SURFACE, corner_radius=th.RADIUS)
        f.grid_columnconfigure(1, weight=1)
        c = tk.Canvas(f, width=_W, height=_H, bg=th.SURFACE, highlightthickness=0)
        c.grid(row=0, column=0, rowspan=3, padx=th.SP_3, pady=th.SP_3)
        diagrams.draw(c, g.pose, "Left" if g.hand == "Left" else "Right", _W, _H,
                      line=th.TEXT, ring=th.TEXT, pinch_to=g.pinch_to)
        ctk.CTkLabel(f, text=g.title, font=th.font("body", bold=True), text_color=th.TEXT,
                     anchor="w").grid(row=0, column=1, sticky="w", pady=(th.SP_3, 0))
        ctk.CTkLabel(f, text=g.does, font=th.font("caption"), text_color=th.MUTED, anchor="nw",
                     justify="left", wraplength=240).grid(row=1, column=1, sticky="nw")
        state = ctk.CTkLabel(f, text="", font=th.font("caption"), text_color=th.MUTED, anchor="w")
        state.grid(row=2, column=1, sticky="w", pady=(0, th.SP_3))
        self._state_labels.append((state, g.flag))
        return f

    def on_show(self) -> None:
        """Mark gestures whose feature is switched off."""
        for label, flag in self._state_labels:
            off = bool(flag) and not self.app.config.get(flag, True)
            label.configure(text="Switched off on the Live tab" if off else "")
