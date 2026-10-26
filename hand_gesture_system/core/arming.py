"""
core/arming.py
--------------
The global ARMED / DISARMED state and the two ways to change it from outside
the GUI: a two-open-palms gesture and a keyboard panic hotkey.

Design
------
* The app starts DISARMED.  While disarmed no OS-level action (mouse, keys,
  scroll, app launch) is sent; the camera, overlay and quit gesture still work.
* ``Arming`` is a tiny thread-safe flag.  The GUI thread, the vision thread
  and the hotkey poll may all flip it; the vision thread notices the change
  on its next frame and releases any held key / mouse button.
* ``ArmGesture`` is pure logic (no camera): feed it the hands each frame.
"""

from __future__ import annotations

import sys
import threading

from . import geometry


class Arming:
    """Thread-safe armed flag with change notification."""

    def __init__(self) -> None:
        self._armed = False
        self._lock = threading.Lock()

    @property
    def armed(self) -> bool:
        return self._armed

    def set_armed(self, value: bool) -> bool:
        """Set the state. Returns True if it actually changed."""
        with self._lock:
            changed = self._armed != bool(value)
            self._armed = bool(value)
            return changed

    def toggle(self) -> bool:
        """Flip the state and return the new value."""
        with self._lock:
            self._armed = not self._armed
            return self._armed


class ArmGesture:
    """Both hands show an open palm, held for ``hold_frames`` frames.

    * Fires once per hold (returns True on the frame the hold completes), then
      needs the gesture to be released before it can fire again (debounce).
    * ``engaged`` is True whenever both palms are open, so the engine can keep
      other gestures from firing while the user is arming / disarming.
    """

    def __init__(self, hold_frames: int = 30) -> None:
        self.hold_frames = hold_frames
        self.count = 0
        self.engaged = False
        self._fired = False

    def update(self, hands) -> bool:
        self.engaged = bool(hands) and len(hands) >= 2 and all(
            geometry.is_open_palm(h["landmarks"]) for h in hands[:2]
        )
        if not self.engaged:
            self.count = 0
            self._fired = False
            return False
        self.count += 1
        if self.count >= self.hold_frames and not self._fired:
            self._fired = True
            return True
        return False

    @property
    def progress(self) -> float:
        return min(self.count / max(self.hold_frames, 1), 1.0)


# Windows virtual-key codes
_VK_CONTROL, _VK_MENU, _VK_F12 = 0x11, 0x12, 0x7B


def _windows_key_down(vk: int) -> bool:
    import ctypes
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


class PanicHotkey:
    """Ctrl+Alt+F12 disarms, from any window (polled, Windows only).

    ``key_down`` is injectable so the logic is testable on any OS. Triggers on
    the press edge, not while held.
    """

    def __init__(self, key_down=None) -> None:
        if key_down is None and sys.platform == "win32":
            key_down = _windows_key_down
        self._key_down = key_down
        self._was_down = False

    def poll(self) -> bool:
        if self._key_down is None:
            return False
        try:
            down = (self._key_down(_VK_CONTROL) and self._key_down(_VK_MENU)
                    and self._key_down(_VK_F12))
        except Exception:
            return False
        fired = down and not self._was_down
        self._was_down = down
        return fired
