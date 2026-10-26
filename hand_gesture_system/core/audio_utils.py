"""
core/audio_utils.py
-------------------
Non-blocking audio alert utilities for the Hand Gesture System.

Problem
-------
Calling ``winsound.Beep(freq, duration)`` on the same thread as the vision
pipeline freezes the camera feed for the entire duration of the tone because
``Beep()`` is a synchronous, blocking Win32 call.

Solution
--------
``play_beep()`` wraps every ``winsound.Beep()`` invocation in a short-lived
daemon thread.  Because the thread is marked ``daemon=True``, it will not
prevent the interpreter from shutting down if the main application exits
while a tone is still playing.

Usage
-----
    from core.audio_utils import play_beep

    play_beep()                        # default 1000 Hz, 200 ms
    play_beep(frequency=2500, duration_ms=100)  # custom tone
"""

from __future__ import annotations

import platform
import threading


def play_beep(frequency: int = 1000, duration_ms: int = 200) -> None:
    """
    Play a system beep **without blocking the calling thread**.

    On Windows this uses ``winsound.Beep``; on other platforms the call is
    silently ignored so the app remains cross-platform safe.

    Parameters
    ----------
    frequency : int
        Tone frequency in hertz (37–32 767 on Windows).
    duration_ms : int
        Tone duration in milliseconds.
    """
    if platform.system() != "Windows":
        return

    def _beep() -> None:
        try:
            import winsound
            winsound.Beep(frequency, duration_ms)
        except Exception:
            pass

    t = threading.Thread(target=_beep, daemon=True, name="AudioBeep")
    t.start()
