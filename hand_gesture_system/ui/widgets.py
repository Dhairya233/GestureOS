"""
ui/widgets.py
-------------
The small, flat component set every view is built from.  No borders-and-card
boxes: sections are a caption plus a hairline, rows are label + control.
"""

from __future__ import annotations

from typing import Callable

import customtkinter as ctk

from . import theme as th
from .icons import icon


class Section(ctk.CTkFrame):
    """Caption + hairline, then content added to `.body`."""

    def __init__(self, parent, title: str, hint: str = "") -> None:
        super().__init__(parent, fg_color="transparent")
        self.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(self, text=title.upper(), font=th.font("caption", bold=True),
                     text_color=th.MUTED, anchor="w").grid(row=0, column=0, sticky="w")
        ctk.CTkFrame(self, height=1, fg_color=th.BORDER).grid(
            row=1, column=0, sticky="ew", pady=(th.SP_1, th.SP_2))
        r = 2
        if hint:
            ctk.CTkLabel(self, text=hint, font=th.font("caption"), text_color=th.MUTED,
                         anchor="w", justify="left", wraplength=560).grid(
                row=2, column=0, sticky="w", pady=(0, th.SP_2))
            r = 3
        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.grid(row=r, column=0, sticky="ew")
        self.body.grid_columnconfigure(0, weight=1)


class SwitchRow(ctk.CTkFrame):
    def __init__(self, parent, label: str, hint: str, value: bool,
                 command: Callable[[bool], None]) -> None:
        super().__init__(parent, fg_color="transparent")
        self.grid_columnconfigure(0, weight=1)
        self.var = ctk.BooleanVar(value=bool(value))
        ctk.CTkLabel(self, text=label, font=th.font("body"), text_color=th.TEXT,
                     anchor="w").grid(row=0, column=0, sticky="w")
        if hint:
            ctk.CTkLabel(self, text=hint, font=th.font("caption"), text_color=th.MUTED,
                         anchor="w").grid(row=1, column=0, sticky="w")
        self.switch = ctk.CTkSwitch(
            self, text="", width=40, variable=self.var, onvalue=True, offvalue=False,
            fg_color=th.TRACK, progress_color=th.TEXT, button_color=th.TEXT,
            button_hover_color=th.MUTED,
            command=lambda: command(bool(self.var.get())))
        self.switch.grid(row=0, column=1, rowspan=2, sticky="e", padx=(th.SP_3, 0))


class SliderRow(ctk.CTkFrame):
    """Label + live value with unit, slider, and a one-line explanation."""

    def __init__(self, parent, label: str, hint: str, lo: float, hi: float,
                 value: float, fmt: Callable[[float], str],
                 on_change: Callable[[float], None],
                 on_release: Callable[[], None], steps: int = 100) -> None:
        super().__init__(parent, fg_color="transparent")
        self.grid_columnconfigure(0, weight=1)
        self._fmt = fmt
        ctk.CTkLabel(self, text=label, font=th.font("body"), text_color=th.TEXT,
                     anchor="w").grid(row=0, column=0, sticky="w")
        self.value_label = ctk.CTkLabel(self, text=fmt(value), font=th.font("body", mono=True),
                                        text_color=th.TEXT, anchor="e")
        self.value_label.grid(row=0, column=1, sticky="e")
        self.slider = ctk.CTkSlider(
            self, from_=lo, to=hi, number_of_steps=steps, height=14,
            fg_color=th.TRACK, progress_color=th.MUTED, button_color=th.TEXT,
            button_hover_color=th.MUTED, command=self._changed)
        self.slider.set(value)
        self.slider.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(th.SP_1, 0))
        self.slider.bind("<ButtonRelease-1>", lambda e: on_release())
        if hint:
            ctk.CTkLabel(self, text=hint, font=th.font("caption"), text_color=th.MUTED,
                         anchor="w").grid(row=2, column=0, columnspan=2, sticky="w")
        self._on_change = on_change

    def _changed(self, v: float) -> None:
        self.value_label.configure(text=self._fmt(v))
        self._on_change(v)

    def set(self, v: float) -> None:
        self.slider.set(v)
        self.value_label.configure(text=self._fmt(v))


class Meter(ctk.CTkFrame):
    """Flat label + bar + number (CPU / RAM / hold progress)."""

    def __init__(self, parent, label: str) -> None:
        super().__init__(parent, fg_color="transparent")
        self.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(self, text=label, font=th.font("caption"), text_color=th.MUTED,
                     width=40, anchor="w").grid(row=0, column=0)
        self.bar = ctk.CTkProgressBar(self, height=6, corner_radius=3, fg_color=th.TRACK,
                                      progress_color=th.TEXT)
        self.bar.set(0)
        self.bar.grid(row=0, column=1, sticky="ew", padx=th.SP_2)
        self.text = ctk.CTkLabel(self, text="0%", font=th.font("caption", mono=True),
                                 text_color=th.TEXT, width=44, anchor="e")
        self.text.grid(row=0, column=2)

    def set(self, pct: float) -> None:
        pct = max(0.0, min(100.0, pct))
        self.bar.set(pct / 100.0)
        self.text.configure(text=f"{pct:.0f}%")


class InlineError(ctk.CTkLabel):
    """Red one-liner that takes no space when empty."""

    def __init__(self, parent) -> None:
        super().__init__(parent, text="", font=th.font("caption"), text_color=th.ERROR,
                         anchor="w", justify="left", wraplength=520)

    def show(self, message: str, *, grid: dict) -> None:
        self.configure(text=message)
        if message:
            self.grid(**grid)
        else:
            self.grid_remove()


class IconButton(ctk.CTkButton):
    """Flat text button with a leading neutral icon."""

    def __init__(self, parent, text: str, icon_name: str, command, *, color: str = None,
                 width: int = 0) -> None:
        color = color or th.TEXT
        super().__init__(parent, text=text, image=icon(icon_name, color, 16), compound="left",
                         font=th.font("body"), text_color=color, fg_color="transparent",
                         hover_color=th.SURFACE_2, corner_radius=th.RADIUS, height=32,
                         anchor="w", command=command, **({"width": width} if width else {}))
