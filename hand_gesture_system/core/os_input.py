"""
core/os_input.py
----------------
The ONLY module that sends mouse / keyboard events to the operating system.

Every feature controller calls ``os_input.click()``, ``os_input.press()`` etc.
instead of pyautogui directly.  That gives us, in one place:

* **Arming:** while ``ARMING`` is disarmed, actions that *do* something are
  silently dropped.  Releases (key up / mouse up) are never blocked, so
  disarming can never leave a key stuck.
* **No throttling:** pyautogui's default 0.1 s pause after every call is
  switched off (it would cap the vision loop at ~10 fps whenever the mouse
  moves).
* **No corner failsafe:** pyautogui's failsafe raises when the cursor touches a
  screen corner, which an air mouse does constantly and which would kill the
  vision thread.  It is replaced by the arming model plus a panic hotkey and
  panic gesture (see core/arming.py).  Cursor targets are clamped to the
  screen instead.
* **Tracking:** we remember which keys / buttons are held so ``release_all()``
  can free exactly those on shutdown, disarm or a crash.
* **Testability:** pyautogui is imported lazily and can be replaced with a fake.
"""

from __future__ import annotations

from .arming import Arming
from .status import TIMELINE

ARMING = Arming()

_backend = None
_held_keys: set[str] = set()
_mouse_down = False


def set_backend(backend) -> None:
    """Install a fake backend (tests) or None to restore the real one."""
    global _backend
    _backend = backend
    _held_keys.clear()
    reset_tracking()


def reset_tracking() -> None:
    global _mouse_down
    _mouse_down = False
    _held_keys.clear()


def _pg():
    global _backend
    if _backend is None:
        import pyautogui
        pyautogui.PAUSE = 0
        pyautogui.FAILSAFE = False
        _backend = pyautogui
    return _backend


def is_armed() -> bool:
    return ARMING.armed


# ── queries (never gated) ──────────────────────────────────────────────────
def size() -> tuple[int, int]:
    w, h = _pg().size()
    return int(w), int(h)


# ── actions (dropped while disarmed) ───────────────────────────────────────
def moveTo(x: float, y: float) -> None:
    if not ARMING.armed:
        return
    w, h = size()
    _pg().moveTo(min(max(x, 0), w - 1), min(max(y, 0), h - 1))


def click() -> None:
    if ARMING.armed:
        _pg().click()
        TIMELINE.add("action", "Click")
    else:
        TIMELINE.add_blocked("Click")


def rightClick() -> None:
    if ARMING.armed:
        _pg().rightClick()
        TIMELINE.add("action", "Right click")
    else:
        TIMELINE.add_blocked("Right click")


def scroll(clicks: int) -> None:
    if ARMING.armed:
        _pg().scroll(clicks)
        TIMELINE.add_throttled("action", "Scroll up" if clicks > 0 else "Scroll down")
    else:
        TIMELINE.add_blocked("Scroll")


def press(key: str) -> None:
    if ARMING.armed:
        _pg().press(key)
        TIMELINE.add("action", f"Key {key}")
    else:
        TIMELINE.add_blocked(f"Key {key}")


def hotkey(*keys: str) -> None:
    if ARMING.armed:
        _pg().hotkey(*keys)
        TIMELINE.add("action", "Hotkey " + "+".join(keys))
    else:
        TIMELINE.add_blocked("Hotkey " + "+".join(keys))


def mouseDown() -> None:
    global _mouse_down
    if ARMING.armed:
        _pg().mouseDown()
        _mouse_down = True


def keyDown(key: str) -> None:
    if ARMING.armed:
        _pg().keyDown(key)
        _held_keys.add(key)


# ── releases (always allowed, only if we are the ones holding) ─────────────
def mouseUp() -> None:
    global _mouse_down
    if _mouse_down:
        _mouse_down = False
        _pg().mouseUp()


def keyUp(key: str) -> None:
    if key in _held_keys:
        _held_keys.discard(key)
        _pg().keyUp(key)


def release_all() -> None:
    """Free every key and mouse button we are holding. Safe to call any time."""
    for key in list(_held_keys):
        try:
            keyUp(key)
        except Exception:
            _held_keys.discard(key)
    try:
        mouseUp()
    except Exception:
        pass
