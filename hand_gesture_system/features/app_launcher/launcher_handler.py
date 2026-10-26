"""
features/app_launcher/launcher_handler.py
------------------------------------------
AppLauncher: Finger-count-based OS quick-launch via the Left hand.

Gesture Map
-----------
  2 fingers (Index + Middle)  → config['app_paths']['2_fingers']
  3 fingers (+ Ring)          → config['app_paths']['3_fingers']
  4 fingers (+ Pinky)         → config['app_paths']['4_fingers']

Finger Detection (core/geometry.py, rotation-invariant)
--------------------------------------------------------
  Index / Middle / Ring / Pinky : tip farther from the wrist than the PIP joint
  Thumb : tip farther from the pinky knuckle than the thumb's IP joint

Temporal Stability Filter
--------------------------
  The detected finger count must remain identical for `target_hold_threshold`
  consecutive frames (default 15) before any launch is triggered.  A single
  frame where the count changes or drops to 0 resets the hold counter to 0,
  so the user must hold the pose cleanly from the beginning again.

Cooldown Guard
--------------
  After each launch a `cooldown_period`-second (default 3 s) window is
  enforced to prevent duplicate application instances from spawning if the
  user holds the pose slightly too long.

Public attributes (read by main.py for OSD)
--------------------------------------------
  finger_count  : int  – active finger count from the last frame (0 if no Left hand)
  hold_frames   : int  – consecutive confirming frames so far
  last_app_name : str  – name of the last application triggered (empty string if none)
"""

import time
import platform

from core import geometry as geo, os_input
from .launch import LaunchError, launch


class AppLauncher:
    """
    Monitors the Left hand's finger count and launches OS applications
    after the count has been held steady for a configurable number of
    consecutive frames.
    """

    # =========================================================================
    # Initialisation
    # =========================================================================

    def __init__(self) -> None:
        # ── State tracking ────────────────────────────────────────────────────
        self.current_count: int   = 0   # finger count confirmed in the *previous* frame
        self.hold_frames:   int   = 0   # consecutive frames the count has been stable
        self.last_app_name: str   = ""  # name of the last triggered application

        # ── Temporal stability ────────────────────────────────────────────────
        self.target_hold_threshold: int = 15   # frames required before triggering

        # ── Cooldown guard ────────────────────────────────────────────────────
        self.last_launch_time: float = 0.0
        self.cooldown_period:  float = 3.0   # seconds between allowed launches

    # =========================================================================
    # Public API
    # =========================================================================

    def update(self, detected_hands, config: dict | None = None) -> None:
        """
        Process one frame of hand data and launch an application if the
        stability and cooldown conditions are both met.

        Parameters
        ----------
        detected_hands : list[dict] | None
            Output of HandTracker.get_landmarks().
            Each dict: { "label": "Right"|"Left",
                         "landmarks": [(id, x_px, y_px, z), ...] }
        config : dict | None
            The live application config dict.  If provided, app_paths values
            are used to resolve launch targets; otherwise defaults are used.
        """
        # ── Locate the Left hand ──────────────────────────────────────────────
        left_hand = self._find_left_hand(detected_hands)

        if left_hand is None:
            # No Left hand visible — reset everything.
            self.current_count = 0
            self.hold_frames   = 0
            return

        lms = left_hand["landmarks"]   # list of (id, x_px, y_px, z)

        # ── Count extended fingers ────────────────────────────────────────────
        active_count = self._count_fingers(lms)

        # The media controller owns the peace sign (play/pause). Without this
        # guard, holding it for ~15 frames also read as "2 fingers" and launched
        # the 2-finger target (Command Prompt by default).
        if (config or {}).get("enable_media", True) and self._is_peace_sign(lms):
            active_count = 0

        # ── Temporal Stability Filter ─────────────────────────────────────────
        if active_count > 0 and active_count == self.current_count:
            # Count is non-zero and unchanged from the previous frame.
            self.hold_frames += 1
        else:
            # Count changed or fell to 0 — restart the stability window.
            self.hold_frames   = 0
            self.current_count = active_count

        # ── OS Quick-Launch Trigger ───────────────────────────────────────────
        if (
            self.hold_frames == self.target_hold_threshold
            and time.time() - self.last_launch_time > self.cooldown_period
        ):
            self._launch(active_count, config or {})

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _find_left_hand(self, detected_hands) -> dict | None:
        """Return the first Left-hand dict from detected_hands, or None."""
        if not detected_hands:
            return None
        return next((h for h in detected_hands if h["label"] == "Left"), None)

    @staticmethod
    def _is_peace_sign(lms) -> bool:
        """Same test as MediaController: index + middle up, ring + pinky curled."""
        return (geo.finger_extended(lms, "index") and geo.finger_extended(lms, "middle")
                and geo.finger_curled_tight(lms, "ring") and geo.finger_curled_tight(lms, "pinky"))

    def _count_fingers(self, lms) -> int:
        """
        Return the number of extended fingers (0-5) on the Left hand.

        Each of index / middle / ring / pinky counts when its tip is farther
        from the wrist than its PIP joint; the thumb counts when its tip is
        farther from the pinky knuckle than its IP joint (swung out, not folded
        across the palm).  All distance-based, so it works at any hand
        rotation and for either hand.
        """
        return geo.count_extended_fingers(lms)

    def _launch(self, count: int, config: dict) -> None:
        """
        Resolve the target for *count* fingers from config['app_paths'], then
        dispatch it: http(s) URLs are opened with webbrowser; everything else is
        started directly without a shell (see launch.py).

        Supported counts: 2, 3, 4.  Any other count is a no-op.
        """
        _key_map = {2: "2_fingers", 3: "3_fingers", 4: "4_fingers"}
        key = _key_map.get(count)
        if key is None:
            self.hold_frames = 0
            return

        app_paths: dict = config.get("app_paths", {})
        _defaults = {
            "2_fingers": "cmd.exe" if platform.system() == "Windows" else "open -a Terminal",
            "3_fingers": "https://www.google.com",
            "4_fingers": "code",
        }
        target: str = app_paths.get(key, _defaults[key]).strip()

        if not target:
            self.hold_frames = 0
            return

        if not os_input.is_armed():
            self.hold_frames = 0      # disarmed: do nothing, no cooldown used
            return

        try:
            app_name = launch(target)
        except LaunchError as exc:
            print(f"[APP LAUNCHER] Not launched ({key}): {exc}")
            self.last_launch_time = time.time()   # don't retry every frame
            self.hold_frames = 0
            return

        print(f"[APP LAUNCHER] Triggered: {app_name}  (fingers={count})")
        self.last_app_name    = app_name
        self.last_launch_time = time.time()
        self.hold_frames      = 0
