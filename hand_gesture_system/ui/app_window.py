"""
ui/app_window.py
-----------------
GestureApp: the main window.  Layout:

    ┌──────────────────────────────────────────────────────────┐
    │ GestureOS        ● DISARMED   camera · fps        [Arm]  │  top bar
    ├──────────┬───────────────────────────────────────────────┤
    │ Live     │                                               │
    │ Gestures │            the selected view                  │
    │ Settings │                                               │
    │ ...      │                                               │
    │ Quit     │                                               │
    └──────────┴───────────────────────────────────────────────┘

The top bar is the one place that always says whether gestures control the PC.
Views are built lazily, once, and shown/hidden; they live in ui/views/.

Threading: the vision thread only calls append_log() (a queue) and writes
core.status.STATUS / TIMELINE; this GUI thread reads them every FEED_INTERVAL ms.

Importable interface
---------------------
    from ui.app_window import GestureApp
    app = GestureApp(vision_engine)
    app.mainloop()
"""

from __future__ import annotations

import datetime
import queue
from collections import deque
from typing import Any

import customtkinter as ctk
import psutil

import config_manager
from core import os_input
from core.status import STATUS

from . import theme as th
from .animator import UIAnimator
from .icons import icon
from .views.diagnostics import DiagnosticsView
from .views.gestures import GesturesView
from .views.live import LiveView
from .views.settings import SettingsView
from .views.whiteboard import WhiteboardView

_NAV = (
    (th.TAB_LIVE, "radio"),
    (th.TAB_GESTURES, "hand"),
    (th.TAB_SETTINGS, "sliders-horizontal"),
    (th.TAB_WHITEBOARD, "pen-line"),
    (th.TAB_DIAGNOSTICS, "activity"),
)
_VIEWS = {
    th.TAB_LIVE: LiveView,
    th.TAB_GESTURES: GesturesView,
    th.TAB_SETTINGS: SettingsView,
    th.TAB_WHITEBOARD: WhiteboardView,
    th.TAB_DIAGNOSTICS: DiagnosticsView,
}
_CAMERA_TEXT = {"loading": "Starting camera", "connected": "Camera connected",
                "no_hand": "Camera connected, no hand", "disconnected": "Camera disconnected"}


class GestureApp(ctk.CTk):
    def __init__(self, vision_engine: Any) -> None:
        super().__init__()
        self.vision_engine = vision_engine
        self.config: dict = config_manager.load_config()
        self._active_tab: str = th.TAB_LIVE
        self.animator = UIAnimator(self)

        self._log_queue: queue.Queue = queue.Queue()
        self.log_history: deque = deque(maxlen=500)
        self.log_sinks: list = []              # callables(line) added by views
        self._shown_armed: bool | None = None
        self.views: dict[str, ctk.CTkFrame] = {}
        self._tick_errors = 0

        self.title("GestureOS")
        self.geometry("1200x740")
        self.minsize(980, 620)
        self.configure(fg_color=th.BG)

        self._build_topbar()
        self._build_sidebar()
        self._build_content_host()
        self._navigate(th.TAB_LIVE)
        # Keyboard route to every tab: while gestures drive the cursor, clicking
        # the sidebar can be hard, so Ctrl+1..5 always works.
        for i, (tab, _icon) in enumerate(_NAV, start=1):
            self.bind(f"<Control-Key-{i}>", lambda _e, t=tab: self._navigate(t))

        self.after(th.FEED_INTERVAL, self._tick)
        self.after(1500, self._poll_hardware)

    # ── layout ────────────────────────────────────────────────────────────
    def _build_topbar(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)
        bar = ctk.CTkFrame(self, height=th.TOPBAR_H, fg_color=th.SURFACE, corner_radius=0)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        bar.grid_propagate(False)
        bar.grid_columnconfigure(3, weight=1)
        ctk.CTkLabel(bar, text="GestureOS", font=th.font("title", bold=True),
                     text_color=th.TEXT, width=th.SIDEBAR_W - th.SP_4, anchor="w").grid(
            row=0, column=0, padx=(th.SP_4, 0), pady=th.SP_3)

        pill = ctk.CTkFrame(bar, fg_color="transparent")
        pill.grid(row=0, column=1, padx=th.SP_4)
        self._arm_dot = ctk.CTkLabel(pill, text="", width=14, image=icon("shield-off", th.MUTED, 16))
        self._arm_dot.grid(row=0, column=0, padx=(0, th.SP_2))
        self._arm_label = ctk.CTkLabel(pill, text="DISARMED", font=th.font("body", bold=True),
                                       text_color=th.MUTED)
        self._arm_label.grid(row=0, column=1)
        self._arm_hint = ctk.CTkLabel(pill, text="Gestures are ignored", font=th.font("caption"),
                                      text_color=th.MUTED)
        self._arm_hint.grid(row=0, column=2, padx=(th.SP_3, 0))

        self._cam_label = ctk.CTkLabel(bar, text=_CAMERA_TEXT["loading"], font=th.font("caption"),
                                       text_color=th.MUTED, anchor="e")
        self._cam_label.grid(row=0, column=3, sticky="e", padx=th.SP_4)

        self._arm_var = ctk.BooleanVar(value=os_input.is_armed())
        self._arm_switch = ctk.CTkSwitch(
            bar, text="Arm", font=th.font("body"), text_color=th.TEXT, variable=self._arm_var,
            command=self._on_arm_switch, fg_color=th.TRACK, progress_color=th.ARMED,
            button_color=th.TEXT, button_hover_color=th.MUTED)
        self._arm_switch.grid(row=0, column=4, padx=(0, th.SP_4))

    def _build_sidebar(self) -> None:
        sb = ctk.CTkFrame(self, width=th.SIDEBAR_W, fg_color=th.BG, corner_radius=0)
        sb.grid(row=1, column=0, sticky="nsw")
        sb.grid_propagate(False)
        sb.grid_rowconfigure(len(_NAV) + 1, weight=1)
        self._nav_btns: dict[str, ctk.CTkButton] = {}
        for i, (tab, icon_name) in enumerate(_NAV):
            b = ctk.CTkButton(
                sb, text=tab, image=icon(icon_name, th.MUTED, 18), compound="left",
                font=th.font("body"), anchor="w", height=36, corner_radius=th.RADIUS,
                fg_color="transparent", hover_color=th.SURFACE, text_color=th.MUTED,
                command=lambda t=tab: self._navigate(t))
            b.grid(row=i, column=0, sticky="ew", padx=th.SP_2, pady=(th.SP_3 if i == 0 else 1, 1))
            self._nav_btns[tab] = b
        ctk.CTkButton(
            sb, text="Quit", image=icon("power", th.MUTED, 18), compound="left",
            font=th.font("body"), anchor="w", height=36, corner_radius=th.RADIUS,
            fg_color="transparent", hover_color=th.SURFACE, text_color=th.MUTED,
            command=self.on_closing).grid(
            row=len(_NAV) + 2, column=0, sticky="ew", padx=th.SP_2, pady=th.SP_3)
        sb.grid_columnconfigure(0, weight=1)

    def _build_content_host(self) -> None:
        host = ctk.CTkFrame(self, fg_color=th.BG, corner_radius=0)
        host.grid(row=1, column=1, sticky="nsew")
        host.grid_columnconfigure(0, weight=1)
        host.grid_rowconfigure(0, weight=1)
        self._content_host = host

    def _navigate(self, tab: str) -> None:
        if tab not in self.views:
            view = _VIEWS[tab](self._content_host, self)
            view.grid(row=0, column=0, sticky="nsew")
            self.views[tab] = view
        prev = self.views.get(self._active_tab)
        if prev is not None and self._active_tab != tab:
            try:
                getattr(prev, "on_hide", lambda: None)()
            except Exception as exc:
                print(f"[GestureApp] on_hide failed: {exc!r}")
        self.views[tab].tkraise()
        self._active_tab = tab
        for name, btn in self._nav_btns.items():
            on = name == tab
            btn.configure(fg_color=th.SURFACE if on else "transparent",
                          text_color=th.TEXT if on else th.MUTED,
                          image=icon(dict(_NAV)[name], th.TEXT if on else th.MUTED, 18))
        on_show = getattr(self.views[tab], "on_show", None)
        if on_show:
            try:
                on_show()
            except Exception as exc:       # a view problem must not block navigation
                print(f"[GestureApp] on_show({tab}) failed: {exc!r}")

    def set_camera_view(self, wanted: bool) -> None:
        """Ask the engine to also render the camera picture (Diagnostics only)."""
        try:
            self.vision_engine.want_camera_view = bool(wanted)
        except Exception:
            pass

    def get_active_tab(self) -> str:
        return self._active_tab

    # ── arm state ─────────────────────────────────────────────────────────
    def _on_arm_switch(self) -> None:
        os_input.ARMING.set_armed(bool(self._arm_var.get()))
        self._sync_arm_widgets()

    def _sync_arm_widgets(self) -> None:
        """Mirror the real armed state (it can also change by gesture/hotkey)."""
        armed = os_input.is_armed()
        if self._arm_var.get() != armed:
            self._arm_var.set(armed)
        if armed == self._shown_armed:
            return
        first = self._shown_armed is None
        self._shown_armed = armed
        target = th.ARMED if armed else th.MUTED
        self._arm_label.configure(text="ARMED" if armed else "DISARMED")
        self._arm_hint.configure(text="Gestures control this PC" if armed else "Gestures are ignored")
        self._arm_dot.configure(image=icon("shield-check" if armed else "shield-off", target, 16))
        if first:
            self._arm_label.configure(text_color=target)
        else:
            self.animator.fade_color(self._arm_label, "text_color",
                                     th.MUTED if armed else th.ARMED, target,
                                     duration_ms=240, steps=8)

    # ── logging (thread-safe entry) ───────────────────────────────────────
    def append_log(self, message: str) -> None:
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self._log_queue.put(f"[{ts}]  {message}\n")

    def _drain_log_queue(self) -> None:
        for _ in range(200):
            try:
                line = self._log_queue.get_nowait()
            except queue.Empty:
                break
            self.log_history.append(line)
            for sink in self.log_sinks:
                sink(line)

    # ── periodic work ─────────────────────────────────────────────────────
    def _poll_hardware(self) -> None:
        view = self.views.get(th.TAB_DIAGNOSTICS)
        if self._active_tab == th.TAB_DIAGNOSTICS and view is not None:
            view.set_hardware(psutil.cpu_percent(), psutil.virtual_memory().percent)
        self.after(2000, self._poll_hardware)

    def _tick(self) -> None:
        if getattr(self.vision_engine, "shutdown_requested", False):
            self.on_closing()
            return                     # window is being destroyed; do not reschedule
        try:
            self._tick_body()
        except Exception as exc:       # never let one bad frame kill the refresh loop
            self._tick_errors += 1
            if self._tick_errors <= 3 or self._tick_errors % 100 == 0:
                print(f"[GestureApp] refresh error #{self._tick_errors}: {exc!r}")
        try:
            self.after(th.FEED_INTERVAL, self._tick)
        except Exception:
            pass                       # window already destroyed

    def _tick_body(self) -> None:
        self._drain_log_queue()
        self._sync_arm_widgets()
        status = STATUS.snapshot()
        self._cam_label.configure(
            text=(_CAMERA_TEXT.get(status.camera, status.camera)
                  + (f" · {status.fps:.0f} fps" if status.fps and status.camera != "disconnected" else "")
                  + (f" · {status.camera_message}" if status.camera_message else "")),
            text_color=th.ERROR if status.camera == "disconnected" else th.MUTED)
        view = self.views.get(self._active_tab)
        refresh = getattr(view, "refresh", None)
        if refresh:
            refresh(status, getattr(self.vision_engine, "latest_frame", None))

    # ── shutdown ──────────────────────────────────────────────────────────
    def on_closing(self) -> None:
        """Stop the engine (it releases the camera itself), then close the window.
        Safe to call more than once."""
        print("[GestureApp] on_closing() - tearing down.")
        self.animator.cancel_all()
        engine = self.vision_engine
        if engine is not None:
            engine.stop()
            if hasattr(engine, "join"):
                engine.join(timeout=3.0)
        try:
            self.destroy()
        except Exception:
            pass
