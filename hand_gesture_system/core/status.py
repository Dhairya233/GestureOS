"""
core/status.py
--------------
Display-only state shared between the vision thread (writer) and the GUI thread
(reader): a snapshot of what the engine currently sees, and an event timeline.

Nothing here controls the computer.  Writers never block; readers get copies.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace

CAMERA_STATES = ("loading", "connected", "no_hand", "disconnected")


@dataclass
class EngineStatus:
    armed: bool = False
    camera: str = "loading"            # one of CAMERA_STATES
    camera_message: str = ""           # e.g. "Reconnecting (attempt 3)"
    fps: float = 0.0
    frame_size: tuple = (0, 0)
    hands: int = 0
    gestures: dict = field(default_factory=dict)   # {"Left": "Open palm", ...}
    feature: str = ""                  # feature currently reacting, if any
    hold_label: str = ""               # what a held pose will do ("Launch app")
    hold_progress: float = 0.0         # 0..1
    classifier: str = "rules"          # which pose classifier is feeding the display
    classifier_note: str = ""          # why a requested classifier is not in use


class StatusBoard:
    """Thread-safe holder for one EngineStatus."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._s = EngineStatus()

    def update(self, **kw) -> None:
        with self._lock:
            self._s = replace(self._s, **kw)

    def snapshot(self) -> EngineStatus:
        with self._lock:
            return replace(self._s, gestures=dict(self._s.gestures))


@dataclass(frozen=True)
class Event:
    seq: int
    t: float            # time.time()
    kind: str           # "gesture" | "action" | "blocked" | "system"
    text: str


class Timeline:
    """Bounded event log with a sequence counter so the UI can ask
    'what is new since seq N' without copying everything."""

    def __init__(self, maxlen: int = 200, clock=time.time) -> None:
        self._lock = threading.Lock()
        self._events: deque[Event] = deque(maxlen=maxlen)
        self._seq = 0
        self._clock = clock
        self._last_blocked: dict[str, float] = {}

    def add(self, kind: str, text: str) -> None:
        with self._lock:
            self._seq += 1
            self._events.append(Event(self._seq, self._clock(), kind, text))

    def add_throttled(self, kind: str, text: str, min_interval: float = 0.5) -> None:
        """Like add(), but drops repeats of the same text inside min_interval
        (used for per-frame actions such as scrolling)."""
        now = self._clock()
        with self._lock:
            if now - self._last_blocked.get(kind + text, -1e9) < min_interval:
                return
            self._last_blocked[kind + text] = now
            self._seq += 1
            self._events.append(Event(self._seq, now, kind, text))

    def add_blocked(self, text: str, min_interval: float = 1.5) -> None:
        """Blocked-while-disarmed events are throttled per text so a held pose
        does not flood the log."""
        now = self._clock()
        with self._lock:
            if now - self._last_blocked.get(text, -1e9) < min_interval:
                return
            self._last_blocked[text] = now
            self._seq += 1
            self._events.append(Event(self._seq, now, "blocked", text))

    def since(self, seq: int = 0) -> list[Event]:
        with self._lock:
            return [e for e in self._events if e.seq > seq]

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._last_blocked.clear()


STATUS = StatusBoard()
TIMELINE = Timeline()


class GestureAnnouncer:
    """Turns the per-frame pose names into timeline events.

    A pose is announced once it has been stable for ``stable_frames`` frames and
    differs from the last announced pose of that hand, so flicker between two
    poses during a transition does not spam the timeline.  'Unknown' is never
    announced."""

    def __init__(self, timeline: Timeline, stable_frames: int = 8) -> None:
        self._tl = timeline
        self._n = stable_frames
        self._cand: dict[str, tuple[str, int]] = {}
        self._announced: dict[str, str] = {}

    def update(self, gestures: dict) -> None:
        for label in list(self._cand):
            if label not in gestures:           # hand left the frame
                del self._cand[label]
                self._announced.pop(label, None)
        for label, pose in gestures.items():
            prev, count = self._cand.get(label, ("", 0))
            count = count + 1 if pose == prev else 1
            self._cand[label] = (pose, count)
            if (count == self._n and pose != "Unknown"
                    and self._announced.get(label) != pose):
                self._announced[label] = pose
                self._tl.add("gesture", f"{label[0]} · {pose}")
