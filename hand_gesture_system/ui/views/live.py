"""
ui/views/live.py
----------------
Live: the hand skeleton on a dark backdrop (no camera picture; that lives on Diagnostics), a one-line readout of what each hand is
doing, an event timeline (recognised gestures and what was sent to the PC), and
the feature switches.
"""

from __future__ import annotations

import time
import tkinter as tk

import customtkinter as ctk

import config_manager
from ui.feed import FeedLabel
from core import tokens
from core.status import TIMELINE

from .. import theme as th
from ..icons import icon
from ..widgets import Section, SwitchRow

# (config key, label, hint)
FEATURES = (
    ("enable_mouse", "Move cursor", "Right hand"),
    ("enable_click", "Click", "Right hand"),
    ("enable_rclick", "Right click", "Right hand"),
    ("enable_drag", "Drag", "Right hand"),
    ("enable_scroller", "Scroll", "Right hand"),
    ("enable_edge", "Switch window", "Right hand"),
    ("enable_canvas", "Air canvas", "Right hand"),
    ("enable_kill", "Quit gesture", "Right hand"),
    ("enable_media", "Play / pause, mute", "Left hand"),
    ("enable_volume", "Volume", "Left hand"),
    ("enable_launcher", "Launch apps", "Left hand"),
    ("enable_modifier", "Hold Ctrl", "Left hand"),
)
_MAX_LINES = 200


class LiveView(ctk.CTkFrame):
    def __init__(self, parent, app) -> None:
        super().__init__(parent, fg_color="transparent")
        self.app = app
        self._tl_seq = 0
        self._last_lines = 0
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0, minsize=290)
        self.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(self, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(th.SP_4, th.SP_3), pady=th.SP_4)
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(0, weight=1)

        # camera feed
        self._feed = ctk.CTkFrame(left, fg_color=th.BG, corner_radius=th.RADIUS,
                                  border_width=1, border_color=th.BORDER)
        self._feed.grid(row=0, column=0, sticky="nsew")
        self._feed.grid_propagate(False)
        self._feed.grid_columnconfigure(0, weight=1)
        self._feed.grid_rowconfigure(0, weight=1)
        # Placeholder (camera waiting / disconnected) and the skeleton feed
        # share the same slot; only one is visible at a time.
        self.video_label = ctk.CTkLabel(
            self._feed, text="Waiting for the camera", font=th.font("body"),
            text_color=th.MUTED, image=icon("camera", th.MUTED, 24), compound="top")
        self.video_label.grid(row=0, column=0)
        self.feed = FeedLabel(self._feed, bg=th.BG)
        self.feed.grid(row=0, column=0)
        self.feed.grid_remove()

        # readout
        row = ctk.CTkFrame(left, fg_color="transparent")
        row.grid(row=1, column=0, sticky="ew", pady=(th.SP_3, 0))
        row.grid_columnconfigure(2, weight=1)
        self._left_txt = ctk.CTkLabel(row, text="Left  -", font=th.font("body", mono=True),
                                      text_color=th.MUTED, anchor="w", width=220)
        self._left_txt.grid(row=0, column=0, sticky="w")
        self._right_txt = ctk.CTkLabel(row, text="Right  -", font=th.font("body", mono=True),
                                       text_color=th.MUTED, anchor="w", width=220)
        self._right_txt.grid(row=0, column=1, sticky="w")
        self._hold_txt = ctk.CTkLabel(row, text="", font=th.font("caption"),
                                      text_color=th.TEXT, anchor="e")
        self._hold_txt.grid(row=0, column=2, sticky="e", padx=(0, th.SP_2))
        self._hold_bar = ctk.CTkProgressBar(row, width=120, height=6, corner_radius=3,
                                            fg_color=th.TRACK, progress_color=th.ARMED)
        self._hold_bar.set(0)
        self._hold_bar.grid(row=0, column=3, sticky="e")
        self._hold_bar.grid_remove()

        # timeline
        sec = Section(left, "Events")
        sec.grid(row=2, column=0, sticky="ew", pady=(th.SP_4, 0))
        wrap = ctk.CTkFrame(sec.body, fg_color=th.SURFACE, corner_radius=th.RADIUS)
        wrap.grid(row=0, column=0, sticky="ew")
        wrap.grid_columnconfigure(0, weight=1)
        self.timeline = tk.Text(
            wrap, bg=th.SURFACE, fg=th.TEXT, bd=0, highlightthickness=0, wrap="none",
            font=(tokens.FONT_MONO, 11), padx=th.SP_3, pady=th.SP_2, state="disabled", height=7,
            cursor="arrow", insertwidth=0)
        self.timeline.grid(row=0, column=0, sticky="ew")
        self.timeline.tag_config("t", foreground=th.MUTED)
        self.timeline.tag_config("gesture", foreground=th.TEXT)
        self.timeline.tag_config("action", foreground=th.ARMED)
        self.timeline.tag_config("blocked", foreground=th.MUTED)
        self.timeline.tag_config("system", foreground=th.MUTED)
        self.timeline.tag_config("empty", foreground=th.MUTED)
        self._set_empty()

        # features
        right = ctk.CTkScrollableFrame(self, fg_color="transparent", width=270,
                                       scrollbar_button_color=th.BORDER,
                                       scrollbar_button_hover_color=th.FAINT)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, th.SP_4), pady=th.SP_4)
        right.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(right, text="Switch a feature off to ignore its gestures.",
                     font=th.font("caption"), text_color=th.MUTED, anchor="w",
                     wraplength=250, justify="left").grid(row=0, column=0, sticky="w",
                                                         pady=(0, th.SP_3))
        r = 1
        for hand in ("Right hand", "Left hand"):
            sec = Section(right, hand)
            sec.grid(row=r, column=0, sticky="ew", pady=(0, th.SP_4))
            r += 1
            for i, (key, label, _hint) in enumerate(f for f in FEATURES if f[2] == hand):
                SwitchRow(sec.body, label, "", app.config.get(key, True),
                          lambda v, k=key: self._save(k, v)).grid(
                    row=i, column=0, sticky="ew", pady=(0, th.SP_2))

    # ── persistence ───────────────────────────────────────────────────────
    def _save(self, key: str, value: bool) -> None:
        self.app.config[key] = value
        config_manager.save_config(self.app.config)

    # ── timeline ──────────────────────────────────────────────────────────
    def _set_empty(self) -> None:
        self.timeline.configure(state="normal")
        self.timeline.delete("1.0", "end")
        self.timeline.insert("end", "No events yet. Show a hand to the camera.", "empty")
        self.timeline.configure(state="disabled")
        self._last_lines = 0

    def _append_events(self, events) -> None:
        if not events:
            return
        t = self.timeline
        t.configure(state="normal")
        if self._last_lines == 0:
            t.delete("1.0", "end")
        for e in events:
            stamp = time.strftime("%H:%M:%S", time.localtime(e.t))
            text = {"action": f"sent  {e.text}", "blocked": f"blocked (disarmed)  {e.text}"}.get(
                e.kind, e.text)
            t.insert("end", f"{stamp}  ", "t")
            t.insert("end", text + "\n", e.kind)
            self._last_lines += 1
        extra = self._last_lines - _MAX_LINES
        if extra > 0:
            t.delete("1.0", f"{extra + 1}.0")
            self._last_lines -= extra
        t.see("end")
        t.configure(state="disabled")

    # ── feed ──────────────────────────────────────────────────────────────
    def _show_frame(self, frame) -> None:
        self.video_label.grid_remove()
        self.feed.grid()
        self.feed.show(frame)

    # ── refresh (called by the app every tick while this view is visible) ─
    def on_show(self) -> None:
        self._pull_events()

    def _pull_events(self) -> None:
        events = TIMELINE.since(self._tl_seq)
        if events:
            self._tl_seq = events[-1].seq
            self._append_events(events)

    def refresh(self, status, frame) -> None:
        self._pull_events()

        if status.camera == "disconnected":
            self.feed.grid_remove()
            self.video_label.grid()
            self.video_label.configure(
                image=icon("camera-off", th.ERROR, 24), compound="top",
                text=status.camera_message or "Camera disconnected", text_color=th.ERROR)
        elif frame is not None:
            self._show_frame(frame)

        for lbl, name in ((self._left_txt, "Left"), (self._right_txt, "Right")):
            g = status.gestures.get(name)
            lbl.configure(text=f"{name}  {g or '-'}", text_color=th.TEXT if g else th.MUTED)

        if status.hold_label:
            self._hold_txt.configure(text=status.hold_label)
            self._hold_bar.set(status.hold_progress)
            self._hold_bar.grid()
        else:
            self._hold_txt.configure(text="")
            self._hold_bar.grid_remove()
