"""
ui/views/whiteboard.py
----------------------
Whiteboard: a plain MOUSE-drawn board.  It is not the gesture "air canvas" (that
draws on the camera feed in the Live tab, controlled by the Right hand); it is
kept as a scratch surface.
"""

from __future__ import annotations

import tkinter as tk

import customtkinter as ctk

import config_manager

from .. import theme as th

# ink colours are content, not interface chrome
INK = {"red": "#EF4444", "blue": "#3B82F6", "green": "#22C55E", "eraser": "#FFFFFF"}


class WhiteboardView(ctk.CTkFrame):
    def __init__(self, parent, app) -> None:
        super().__init__(parent, fg_color="transparent")
        self.app = app
        self._color = app.config.get("canvas_active_color", "green")
        if self._color not in INK:
            self._color = "green"
        self._last: tuple[int, int] | None = None
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.grid(row=0, column=0, sticky="ew", padx=th.SP_4, pady=(th.SP_4, th.SP_3))
        bar.grid_columnconfigure(10, weight=1)
        self._swatches: dict[str, tk.Canvas] = {}
        for i, (name, hexv) in enumerate(INK.items()):
            c = tk.Canvas(bar, width=28, height=28, bg=th.BG, highlightthickness=0, cursor="hand2")
            c.grid(row=0, column=i, padx=(0, th.SP_2))
            c.bind("<Button-1>", lambda e, n=name: self.select(n))
            self._swatches[name] = c
        self._paint_swatches()

        ctk.CTkLabel(bar, text="Brush", font=th.font("caption"), text_color=th.MUTED).grid(
            row=0, column=5, padx=(th.SP_4, th.SP_2))
        size = int(app.config.get("canvas_brush_thickness", 5))
        self._size_lbl = ctk.CTkLabel(bar, text=f"{size} px", font=th.font("caption", mono=True),
                                      text_color=th.TEXT, width=44)
        slider = ctk.CTkSlider(bar, from_=1, to=20, number_of_steps=19, width=140, height=14,
                               fg_color=th.TRACK, progress_color=th.MUTED, button_color=th.TEXT,
                               button_hover_color=th.MUTED, command=self._on_size)
        slider.set(size)
        slider.bind("<ButtonRelease-1>", lambda e: config_manager.save_config(self.app.config))
        slider.grid(row=0, column=6)
        self._size_lbl.grid(row=0, column=7)
        ctk.CTkButton(bar, text="Clear", width=72, height=30, font=th.font("body"),
                      fg_color=th.SURFACE, hover_color=th.SURFACE_2, text_color=th.TEXT,
                      corner_radius=th.RADIUS, command=self.clear).grid(row=0, column=11)

        self.canvas = tk.Canvas(self, bg="#FFFFFF", highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=th.SP_4, pady=(0, th.SP_2))
        self.canvas.bind("<Button-1>", self._down)
        self.canvas.bind("<B1-Motion>", self._move)
        self.canvas.bind("<ButtonRelease-1>", lambda e: setattr(self, "_last", None))
        ctk.CTkLabel(self, text="Mouse whiteboard. Drawing in the air happens on the Live tab with "
                                "your Right hand.", font=th.font("caption"), text_color=th.MUTED,
                     anchor="w").grid(row=2, column=0, sticky="w", padx=th.SP_4, pady=(0, th.SP_3))

    def _paint_swatches(self) -> None:
        for name, c in self._swatches.items():
            c.delete("all")
            sel = name == self._color
            c.create_oval(5, 5, 23, 23, fill=INK[name], outline=th.MUTED if name == "eraser" else INK[name])
            if sel:
                c.create_oval(1, 1, 27, 27, outline=th.TEXT, width=2)

    def select(self, name: str) -> None:
        self._color = name
        self.app.config["canvas_active_color"] = name
        config_manager.save_config(self.app.config)
        self._paint_swatches()

    def _on_size(self, v: float) -> None:
        n = int(round(v))
        self.app.config["canvas_brush_thickness"] = n
        self._size_lbl.configure(text=f"{n} px")

    def _down(self, e) -> None:
        self._last = (e.x, e.y)

    def _move(self, e) -> None:
        if self._last is None:
            self._last = (e.x, e.y)
            return
        self.canvas.create_line(*self._last, e.x, e.y, fill=INK[self._color],
                                width=int(self.app.config.get("canvas_brush_thickness", 5)),
                                capstyle=tk.ROUND, joinstyle=tk.ROUND, smooth=True)
        self._last = (e.x, e.y)

    def clear(self) -> None:
        self.canvas.delete("all")
