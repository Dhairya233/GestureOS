"""
features/modifier_key/modifier_handler.py
-------------------------------------------
ModifierController: Left-hand fist gesture that holds / releases the Ctrl
modifier key via pyautogui, with a presence failsafe that automatically
releases the key if the hand leaves the camera frame.

Gesture Vocabulary (Left hand only)
───────────────────────────────────────────────────────────────────────────────

FIST — Ctrl key held
─────────────────────
  All four fingers are tightly curled so that every fingertip sits strictly
  BELOW its own PIP joint (larger Y = lower on screen in pixel space):

    Index  tip Y (8)  >  Index  PIP Y (6)
    Middle tip Y (12) >  Middle PIP Y (10)
    Ring   tip Y (16) >  Ring   PIP Y (14)
    Pinky  tip Y (20) >  Pinky  PIP Y (18)

  On first frame of detection  → os_input.keyDown('ctrl')
  While sustained              → key remains physically held by the OS.

OPEN / RELAXED — Ctrl key released
────────────────────────────────────
  Any frame where even ONE fingertip is not below its PIP joint breaks the
  fist condition and triggers an immediate os_input.keyUp('ctrl').

PRESENCE FAILSAFE
──────────────────
  If the Left hand disappears from the frame while Ctrl is held, the key is
  force-released before returning, preventing a stuck modifier state.

Public attributes (read by main.py for OSD)
────────────────────────────────────────────
  is_ctrl_held : bool  – True while Ctrl is physically held by pyautogui.
"""

from core import geometry as geo, os_input


class ModifierController:
    """
    Monitors the Left hand for a fist posture and translates it into a
    held / released Ctrl modifier key via pyautogui.

    The key is held for as long as the fist is maintained and released the
    instant the hand opens, relaxes, or leaves the camera frame entirely.
    """


    # =========================================================================
    # Initialisation
    # =========================================================================

    def __init__(self) -> None:
        # True while pyautogui is physically holding Ctrl down.
        self.is_ctrl_held: bool = False

    # =========================================================================
    # Public API
    # =========================================================================

    def update(self, detected_hands) -> None:
        """
        Process one frame of hand data and drive the Ctrl modifier key state.

        Parameters
        ----------
        detected_hands : list[dict] | None
            Output of HandTracker.get_landmarks().
            Each dict: { "label": "Right"|"Left",
                         "landmarks": [(id, x_px, y_px, z), ...] }
        """
        left_hand = self._find_left_hand(detected_hands)

        # ── Presence failsafe ─────────────────────────────────────────────────
        if left_hand is None:
            if self.is_ctrl_held:
                os_input.keyUp("ctrl")
                self.is_ctrl_held = False
                print("--> OS STATE: CTRL KEY FORCED RELEASE (HAND LOST)")
            return

        lms = left_hand["landmarks"]   # list of (id, x_px, y_px, z)

        # ── Fist detection ────────────────────────────────────────────────────
        if self._is_fist(lms):
            if not self.is_ctrl_held and os_input.is_armed():
                os_input.keyDown("ctrl")
                self.is_ctrl_held = True
                print("--> OS STATE: CTRL KEY DOWN")
        else:
            if self.is_ctrl_held:
                os_input.keyUp("ctrl")
                self.is_ctrl_held = False
                print("--> OS STATE: CTRL KEY RELEASED")

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _find_left_hand(self, detected_hands) -> dict | None:
        """Return the first dict whose label is 'Left', or None."""
        if not detected_hands:
            return None
        return next((h for h in detected_hands if h["label"] == "Left"), None)

    def _is_fist(self, lms) -> bool:
        """
        True when ALL four fingers are curled (none extended: each tip is no
        farther from the wrist than its PIP joint).  Rotation-invariant.
        """
        return geo.is_fist(lms)

    def reset(self) -> None:
        """Forget the held-Ctrl state (called when GestureOS is disarmed)."""
        self.is_ctrl_held = False
