"""
ui/animator.py
--------------
UIAnimator: lightweight tkinter ``after()``-based colour fades and pulses.
"""

from __future__ import annotations

from typing import Callable
import customtkinter as ctk
import math

from .theme import (
    hex_blend,
)


class UIAnimator:
    """
    Lightweight CPU-friendly animation engine using tkinter after() scheduling.

    All animations use chained ``root.after()`` calls — no threads, no hardware
    acceleration.  Typical overhead is ~10-20 scheduler callbacks per second,
    which is negligible on any modern CPU.
    """

    def __init__(self, root: ctk.CTk) -> None:
        self._root = root
        self._pulse_ids: dict[str, str] = {}   # tag -> after_id

    # ── Smooth colour fade ────────────────────────────────────────────

    def fade_color(
        self,
        widget,
        prop: str,
        start_hex: str,
        end_hex: str,
        duration_ms: int = 200,
        steps: int = 10,
        on_complete: Callable | None = None,
    ) -> None:
        """
        Smoothly interpolate *prop* (e.g. ``"text_color"``, ``"fg_color"``)
        from *start_hex* to *end_hex* over *duration_ms* using chained
        ``after()`` calls.
        """
        step_ms = max(1, duration_ms // steps)

        def _step(i: int) -> None:
            if i > steps:
                if on_complete is not None:
                    on_complete()
                return
            t = i / steps
            colour = hex_blend(end_hex, start_hex, t)
            try:
                widget.configure(**{prop: colour})
            except Exception:
                return                       # widget was destroyed mid-fade
            self._root.after(step_ms, lambda: _step(i + 1))

        _step(0)

    # ── Continuous sine-wave pulse ────────────────────────────────────

    def pulse_widget(
        self,
        widget,
        prop: str,
        bright_hex: str,
        dim_hex: str,
        period_ms: int = 2000,
        tag: str = "",
    ) -> str:
        """
        Gently oscillate *prop* between *bright_hex* and *dim_hex* using
        a sine-wave curve.  Returns a *tag* that can be passed to
        :meth:`cancel_pulse` to stop the animation.
        """
        if not tag:
            tag = f"pulse_{id(widget)}_{id(self)}"
        self.cancel_pulse(tag)

        step_ms = 50                         # 20 fps — plenty smooth
        total_steps = max(1, period_ms // step_ms)

        def _tick(step: int) -> None:
            t = (math.sin(2 * math.pi * step / total_steps) + 1) / 2
            colour = hex_blend(bright_hex, dim_hex, t)
            try:
                widget.configure(**{prop: colour})
            except Exception:
                self._pulse_ids.pop(tag, None)
                return
            aid = self._root.after(
                step_ms, lambda: _tick((step + 1) % total_steps),
            )
            self._pulse_ids[tag] = aid

        _tick(0)
        return tag

    # ── Pulse control ─────────────────────────────────────────────────

    def cancel_pulse(self, tag: str) -> None:
        """Stop a running pulse animation identified by *tag*."""
        aid = self._pulse_ids.pop(tag, None)
        if aid is not None:
            try:
                self._root.after_cancel(aid)
            except Exception:
                pass

    def cancel_all(self) -> None:
        """Cancel every running pulse animation."""
        for tag in list(self._pulse_ids):
            self.cancel_pulse(tag)
