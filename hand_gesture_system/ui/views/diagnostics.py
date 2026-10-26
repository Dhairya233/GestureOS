"""
ui/views/diagnostics.py
-----------------------
Diagnostics: engine facts, CPU / RAM, and the log.  No camera feed here (that is
the Live tab).  The log is appended on the GUI thread only; other threads use
GestureApp.append_log(), which goes through a queue.
"""

from __future__ import annotations

import customtkinter as ctk

from ui.feed import FeedLabel
from .. import theme as th
from ..widgets import IconButton, Meter, Section

_EMPTY = "No log lines yet. Warnings and state changes will appear here."


class DiagnosticsView(ctk.CTkFrame):
    def __init__(self, parent, app) -> None:
        super().__init__(parent, fg_color="transparent")
        self.app = app
        self._empty = True
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        top = ctk.CTkFrame(self, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=th.SP_4, pady=(th.SP_4, 0))
        top.grid_columnconfigure(0, weight=1, uniform="d")
        top.grid_columnconfigure(1, weight=1, uniform="d")

        sec = Section(top, "System")
        sec.grid(row=0, column=0, sticky="new", padx=(0, th.SP_6))
        self.cpu, self.ram = Meter(sec.body, "CPU"), Meter(sec.body, "RAM")
        self.cpu.grid(row=0, column=0, sticky="ew", pady=(0, th.SP_2))
        self.ram.grid(row=1, column=0, sticky="ew")

        sec = Section(top, "Engine")
        sec.grid(row=0, column=1, sticky="new")
        sec.body.grid_columnconfigure(0, weight=0)
        sec.body.grid_columnconfigure(1, weight=1)
        self._facts: dict[str, ctk.CTkLabel] = {}
        for i, name in enumerate(("Camera", "Frame", "Rate", "Hands")):
            ctk.CTkLabel(sec.body, text=name, font=th.font("caption"), text_color=th.MUTED,
                         width=64, anchor="w").grid(row=i, column=0, sticky="w")
            v = ctk.CTkLabel(sec.body, text="-", font=th.font("caption", mono=True),
                             text_color=th.TEXT, anchor="w")
            v.grid(row=i, column=1, sticky="w")
            self._facts[name] = v

        # Camera picture with the overlay drawn on it (the Live tab shows only
        # the skeleton). The engine renders this view only while this tab is open.
        cam = ctk.CTkFrame(self, fg_color=th.BG, corner_radius=th.RADIUS, height=260,
                           border_width=1, border_color=th.BORDER)
        cam.grid(row=1, column=0, sticky="ew", padx=th.SP_4, pady=(th.SP_4, 0))
        cam.grid_propagate(False)
        cam.grid_columnconfigure(0, weight=1)
        cam.grid_rowconfigure(0, weight=1)
        self.camera_note = ctk.CTkLabel(cam, text="Waiting for the camera", font=th.font("caption"),
                                        text_color=th.MUTED)
        self.camera_note.grid(row=0, column=0)
        self.camera_feed = FeedLabel(cam, bg=th.BG)
        self.camera_feed.grid(row=0, column=0)
        self.camera_feed.grid_remove()

        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.grid(row=2, column=0, sticky="ew", padx=th.SP_4, pady=(th.SP_4, th.SP_2))
        hdr.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(hdr, text="LOG", font=th.font("caption", bold=True), text_color=th.MUTED,
                     anchor="w").grid(row=0, column=0, sticky="w")
        IconButton(hdr, "Copy", "terminal", self.copy_log, color=th.MUTED, width=80).grid(row=0, column=1)
        IconButton(hdr, "Clear", "eraser", self.clear_log, color=th.MUTED, width=80).grid(row=0, column=2)

        self.telemetry_box = ctk.CTkTextbox(
            self, fg_color=th.SURFACE, text_color=th.TEXT, font=th.font("caption", mono=True),
            corner_radius=th.RADIUS, wrap="none", border_width=0, activate_scrollbars=True,
            scrollbar_button_color=th.BORDER, scrollbar_button_hover_color=th.FAINT)
        self.telemetry_box.grid(row=3, column=0, sticky="nsew", padx=th.SP_4, pady=(0, th.SP_4))
        self._show_empty()
        for line in app.log_history:
            self.write(line)
        app.log_sinks.append(self.write)

    # ── log ───────────────────────────────────────────────────────────────
    def _show_empty(self) -> None:
        b = self.telemetry_box
        b.configure(state="normal")
        b.delete("1.0", "end")
        b.insert("end", _EMPTY)
        b.configure(state="disabled", text_color=th.MUTED)
        self._empty = True

    def write(self, line: str) -> None:
        b = self.telemetry_box
        b.configure(state="normal")
        if self._empty:
            b.delete("1.0", "end")
            self._empty = False
            b.configure(text_color=th.TEXT)
        b.insert("end", line)
        b.see("end")
        b.configure(state="disabled")

    def clear_log(self) -> None:
        self.app.log_history.clear()
        self._show_empty()

    def copy_log(self) -> None:
        text = "" if self._empty else self.telemetry_box.get("1.0", "end")
        self.clipboard_clear()
        self.clipboard_append(text)

    # ── live values ───────────────────────────────────────────────────────
    def set_hardware(self, cpu: float, ram: float) -> None:
        self.cpu.set(cpu)
        self.ram.set(ram)

    def on_show(self) -> None:
        self.app.set_camera_view(True)

    def on_hide(self) -> None:
        self.app.set_camera_view(False)

    def refresh(self, status, frame) -> None:
        cam = getattr(self.app.vision_engine, "latest_camera_frame", None)
        if status.camera == "disconnected":
            self.camera_feed.grid_remove()
            self.camera_note.grid()
            self.camera_note.configure(text=status.camera_message or "Camera disconnected",
                                       text_color=th.ERROR)
        elif cam is not None:
            self.camera_note.grid_remove()
            self.camera_feed.grid()
            self.camera_feed.show(cam)
        w, h = status.frame_size
        self._facts["Camera"].configure(
            text=status.camera.replace("_", " "),
            text_color=th.ERROR if status.camera == "disconnected" else th.TEXT)
        self._facts["Frame"].configure(text=f"{w} x {h}" if w else "-")
        self._facts["Rate"].configure(text=f"{status.fps:.0f} fps" if status.fps else "-")
        self._facts["Hands"].configure(text=str(status.hands))
