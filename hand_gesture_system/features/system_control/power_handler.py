"""
features/system_control/power_handler.py
-----------------------------------------
SystemController: Gesture-based application lifecycle management.

Public API
----------
update(detected_hands) -> bool
    Primary entry point used by the VisionEngine thread.
    Returns True after a geometrically strict "Thumbs Up" is held for
    required_frames (20) consecutive frames, False in all other cases.
    Never calls sys.exit() — the caller is responsible for acting on the
    returned value.

check_for_quit(detected_hands) -> bool
    Backwards-compatible alias for update().

Strict Thumbs Up geometry (MediaPipe 21-point model)
-----------------------------------------------------
Condition 1 — Thumb clearly RAISED:
    Thumb tip  (LM 4) Y  <  Index MCP  (LM 5) Y      ← tip above index knuckle
    Thumb tip  (LM 4) Y  <  Thumb IP   (LM 3) Y      ← tip above its own IP joint

Condition 2 — Four fingers tightly FISTED (tips below their MCP knuckles):
    Index  tip (LM  8) Y  >  Index  MCP (LM  5) Y
    Middle tip (LM 12) Y  >  Middle MCP (LM  9) Y
    Ring   tip (LM 16) Y  >  Ring   MCP (LM 13) Y
    Pinky  tip (LM 20) Y  >  Pinky  MCP (LM 17) Y

Stability filter
----------------
Both conditions must hold for 10 consecutive frames before True is
returned.  A single failed frame resets the counter to 0, requiring
the user to hold the pose cleanly from the beginning again.

Note on Y-axis direction
------------------------
In pixel space Y=0 is at the TOP of the frame, so a smaller Y value
means the point is physically higher on screen.
"""

import math

from core import geometry as geo


class SystemController:
    """
    Monitors the right hand for a stable Thumbs Up gesture.

    Returns True from update() after required_frames (20) consecutive
    frames confirm the pose.  The class never calls sys.exit() itself;
    the VisionEngine thread sets its own shutdown_requested flag based on
    the returned boolean.
    """

    # ── Landmark indices ──────────────────────────────────────────────────────
    _THUMB_TIP = 4
    _THUMB_IP  = 3    # Thumb interphalangeal joint
    _INDEX_MCP = 5    # Index finger MCP (reference for "thumb above fist")


    # Frames the pose must be held before termination is signalled
    required_frames = 20   # overridden from config (kill_frames)

    # =========================================================================
    # Initialisation
    # =========================================================================

    def __init__(self) -> None:
        # Running count of consecutive frames where the gesture was confirmed.
        self.confirmation_frames: int = 0

    # =========================================================================
    # Public API
    # =========================================================================

    def update(self, detected_hands) -> bool:
        """
        Evaluate the current frame for the Thumbs Up kill-switch gesture.

        Parameters
        ----------
        detected_hands : list[dict] | None
            Structured hand data from HandTracker.get_landmarks():
            [ { "label": "Right"|"Left",
                "landmarks": [(id, x_px, y_px, z), ...] }, ... ]

        Returns
        -------
        bool
            True  — the gesture has been held for required_frames (20)
                    consecutive frames; the caller should initiate shutdown.
            False — gesture absent or the stability counter is still
                    accumulating; do nothing.

        Side effects
        ------------
        Prints a stability counter line while the pose is being held.
        Resets confirmation_frames to 0 whenever a frame fails the check.
        """
        right_hand = self._find_right_hand(detected_hands)

        # No right hand in the frame — reset and report safe.
        if right_hand is None:
            self.confirmation_frames = 0
            return False

        lms = right_hand["landmarks"]   # list of (id, x_px, y_px, z)

        # ── Evaluate strict Thumbs Up geometry ───────────────────────────────
        if self._thumb_clearly_raised(lms) and self._fist_is_tight(lms):
            self.confirmation_frames += 1
            if self.confirmation_frames == 1:
                print("[SYS CTRL] Thumbs-up detected; hold to quit.")

            if self.confirmation_frames >= self.required_frames:
                print("[SYS CTRL] Thumbs Up confirmed — requesting shutdown.")
                return True
        else:
            # Even one bad frame resets the sequence.
            self.confirmation_frames = 0

        return False

    def check_for_quit(self, detected_hands) -> bool:
        """Backwards-compatible alias for update()."""
        return self.update(detected_hands)

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _find_right_hand(self, detected_hands) -> dict | None:
        if not detected_hands:
            return None
        return next((h for h in detected_hands if h["label"] == "Right"), None)

    # The thumb must point within 45 degrees of straight up on the screen.
    _MIN_THUMB_UP_COS = 0.707

    def _thumb_clearly_raised(self, lms) -> bool:
        """
        Thumb extended AND pointing (nearly) straight up on screen.

        A thumbs-up is a gravity gesture, so "up" is deliberately tied to the
        screen: the thumb's direction (landmark 2 -> tip) must be within 45
        degrees of vertical, and the tip must be above the thumb's IP joint and
        the index knuckle.  A hand turned sideways with the thumb out does NOT
        count, which makes an accidental quit less likely.  The thumb-extended
        and fist parts use rotation-invariant geometry.
        """
        _, x2, y2, _ = lms[2]
        _, x4, y4, _ = lms[self._THUMB_TIP]
        _, _, ip_y, _ = lms[self._THUMB_IP]
        _, _, mcp_y, _ = lms[self._INDEX_MCP]

        length = math.hypot(x4 - x2, y4 - y2)
        if length == 0:
            return False
        points_up = (y2 - y4) / length >= self._MIN_THUMB_UP_COS
        return (geo.thumb_extended(lms) and points_up
                and y4 < mcp_y and y4 < ip_y)

    def _fist_is_tight(self, lms) -> bool:
        """
        All four fingers tightly curled: each tip folded in closer to the wrist
        than its own base knuckle (rotation-invariant).
        """
        return all(geo.finger_curled_tight(lms, f) for f in geo.FINGER_NAMES)
