"""
ui/views/settings.py
--------------------
Settings: grouped, with units and ranges, launcher targets validated inline,
and a two-step reset to defaults.  Every change is written to config.json; the
vision thread picks it up within a second or two.
"""

from __future__ import annotations

import customtkinter as ctk

import config_manager
from features.app_launcher.launch import LaunchError, parse_target

from .. import theme as th
from ..widgets import InlineError, Section, SliderRow, SwitchRow

# key, label, hint, lo, hi, steps, formatter, to_display / from_display
SLIDERS = (
    ("Cursor", (
        ("mouse_smoothing", "Smoothing", "Higher is steadier but laggier.", 1, 15, 28,
         lambda v: f"{v:.1f}", None),
        ("screen_reduction", "Cursor area", "Share of the camera frame that maps to the whole screen.",
         0.2, 1.0, 80, lambda v: f"{v * 100:.0f}% of frame", None),
    )),
    ("Scrolling", (
        ("scroll_speed", "Scroll speed", "Clicks per frame grow with distance from the centre band.",
         0.01, 1.0, 99, lambda v: f"{v:.2f}x", None),
    )),
    ("Volume", (
        ("volume_delay", "Repeat interval", "Time between volume steps while the pinch is held.",
         0.05, 0.5, 90, lambda v: f"{int(v * 1000)} ms", None),
    )),
    ("Quit gesture", (
        ("kill_frames", "Hold length", "Frames the thumbs-up must be held (about 30 per second).",
         5, 60, 55, lambda v: f"{int(round(v))} frames", int),
    )),
)
LAUNCHERS = (("2_fingers", "Two fingers"), ("3_fingers", "Three fingers"), ("4_fingers", "Four fingers"))


class SettingsView(ctk.CTkScrollableFrame):
    def __init__(self, parent, app) -> None:
        super().__init__(parent, fg_color="transparent", scrollbar_button_color=th.BORDER,
                         scrollbar_button_hover_color=th.FAINT)
        self.app = app
        self.sliders: dict[str, SliderRow] = {}
        self.entries: dict[str, ctk.CTkEntry] = {}
        self.errors: dict[str, InlineError] = {}
        self._reset_armed = False
        self.grid_columnconfigure(0, weight=1)

        col = ctk.CTkFrame(self, fg_color="transparent")
        col.grid(row=0, column=0, sticky="w", padx=th.SP_4, pady=th.SP_4)
        col.grid_columnconfigure(0, weight=1)
        r = 0

        sec = Section(col, "Hands", "If Left and Right are swapped in the camera view, flip this.")
        sec.grid(row=r, column=0, sticky="ew", pady=(0, th.SP_6)); r += 1
        self.swap = SwitchRow(sec.body, "Swap hand labels", "MediaPipe reports mirrored images the "
                              "other way round.", app.config.get("swap_handedness", True),
                              lambda v: self._save("swap_handedness", v))
        self.swap.grid(row=0, column=0, sticky="ew")

        sec = Section(col, "Pose recognition", "Which classifier names your poses on the Live tab. "
                      "The gesture controls (cursor, clicks, scroll...) always use the geometry rules.")
        sec.grid(row=r, column=0, sticky="ew", pady=(0, th.SP_6)); r += 1
        self.cls_btn = ctk.CTkSegmentedButton(
            sec.body, values=["rules", "ml"], command=lambda v: self._save("classifier", v),
            font=th.font("body"), fg_color=th.SURFACE, selected_color=th.SURFACE_2,
            selected_hover_color=th.SURFACE_2, unselected_color=th.SURFACE,
            unselected_hover_color=th.SURFACE_2, text_color=th.TEXT, corner_radius=th.RADIUS)
        self.cls_btn.set(app.config.get("classifier", "rules"))
        self.cls_btn.grid(row=0, column=0, sticky="w")
        self.cls_note = ctk.CTkLabel(sec.body, text="", font=th.font("caption"), text_color=th.WARN,
                                     anchor="w", justify="left", wraplength=520)
        self.cls_note.grid(row=1, column=0, sticky="w", pady=(th.SP_2, 0))
        self.cls_note.grid_remove()

        for title, items in SLIDERS:
            sec = Section(col, title)
            sec.grid(row=r, column=0, sticky="ew", pady=(0, th.SP_6)); r += 1
            for i, (key, label, hint, lo, hi, steps, fmt, cast) in enumerate(items):
                value = float(app.config.get(key, config_manager.DEFAULTS[key]))
                row = SliderRow(
                    sec.body, label, hint, lo, hi, value, fmt,
                    on_change=lambda v, k=key, c=cast: self._set(k, c(round(v)) if c else round(v, 4)),
                    on_release=self._flush, steps=steps)
                row.grid(row=i, column=0, sticky="ew", pady=(0, th.SP_3))
                self.sliders[key] = row

        sec = Section(col, "App launcher", "What each Left-hand finger count opens. A web link "
                      "(https://...) or a program name; no shell commands.")
        sec.grid(row=r, column=0, sticky="ew", pady=(0, th.SP_6)); r += 1
        paths = app.config.get("app_paths", {})
        for i, (key, label) in enumerate(LAUNCHERS):
            ctk.CTkLabel(sec.body, text=label, font=th.font("body"), text_color=th.TEXT,
                         anchor="w").grid(row=i * 3, column=0, sticky="w")
            e = ctk.CTkEntry(sec.body, width=460, height=32, font=th.font("body", mono=True),
                             fg_color=th.SURFACE, border_color=th.BORDER, text_color=th.TEXT,
                             corner_radius=th.RADIUS, placeholder_text="Not set")
            e.insert(0, paths.get(key, ""))
            e.grid(row=i * 3 + 1, column=0, sticky="w", pady=(th.SP_1, 0))
            e.bind("<FocusOut>", lambda ev, k=key: self._commit_path(k))
            e.bind("<Return>", lambda ev, k=key: self._commit_path(k))
            err = InlineError(sec.body)
            self.entries[key], self.errors[key] = e, err
            sec.body.grid_rowconfigure(i * 3 + 2, minsize=th.SP_3)

        sec = Section(col, "Reset")
        sec.grid(row=r, column=0, sticky="ew"); r += 1
        self.reset_btn = ctk.CTkButton(
            sec.body, text="Reset these settings to defaults", width=260, height=32,
            font=th.font("body"), fg_color=th.SURFACE, hover_color=th.SURFACE_2,
            text_color=th.TEXT, corner_radius=th.RADIUS, command=self._reset)
        self.reset_btn.grid(row=0, column=0, sticky="w")

    def refresh(self, status, frame) -> None:
        """Called every tick while this view is visible: show why 'ml' is not in use."""
        note = status.classifier_note if self.app.config.get("classifier") == "ml" else ""
        if note != self.cls_note.cget("text"):
            self.cls_note.configure(text=note)
            if note:
                self.cls_note.grid()
            else:
                self.cls_note.grid_remove()

    # ── config plumbing ───────────────────────────────────────────────────
    def _set(self, key: str, value) -> None:
        self.app.config[key] = value

    def _flush(self) -> None:
        config_manager.save_config(self.app.config)

    def _save(self, key: str, value) -> None:
        self._set(key, value)
        self._flush()

    # ── launcher validation ───────────────────────────────────────────────
    def validate_path(self, key: str) -> str:
        """Return '' if the entry is empty or valid, else a user-facing message."""
        text = self.entries[key].get().strip()
        if not text:
            return ""
        try:
            parse_target(text)
        except LaunchError as exc:
            return str(exc)
        return ""

    def _commit_path(self, key: str) -> None:
        msg = self.validate_path(key)
        row = list(self.entries).index(key) * 3 + 2
        self.errors[key].show(msg, grid={"row": row, "column": 0, "sticky": "w"})
        self.entries[key].configure(border_color=th.ERROR if msg else th.BORDER)
        if msg:
            return                       # never save an invalid target
        self.app.config.setdefault("app_paths", {})[key] = self.entries[key].get().strip()
        self._flush()

    # ── reset (two clicks) ────────────────────────────────────────────────
    def _reset(self) -> None:
        if not self._reset_armed:
            self._reset_armed = True
            self.reset_btn.configure(text="Click again to reset", text_color=th.WARN)
            self.after(4000, self._disarm_reset)
            return
        self._disarm_reset()
        d = config_manager.DEFAULTS
        for key, row in self.sliders.items():
            self._set(key, d[key])
            row.set(float(d[key]))
        self._set("swap_handedness", d["swap_handedness"])
        self._set("classifier", d["classifier"])
        self.cls_btn.set(d["classifier"])
        self.swap.var.set(bool(d["swap_handedness"]))
        self.app.config["app_paths"] = dict(d["app_paths"])
        for key, e in self.entries.items():
            e.delete(0, "end")
            e.insert(0, d["app_paths"].get(key, ""))
            self.errors[key].show("", grid={})
            e.configure(border_color=th.BORDER)
        self._flush()

    def _disarm_reset(self) -> None:
        self._reset_armed = False
        try:
            self.reset_btn.configure(text="Reset these settings to defaults", text_color=th.TEXT)
        except Exception:
            pass
