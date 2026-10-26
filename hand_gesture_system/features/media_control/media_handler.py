"""
features/media_control/media_handler.py
-----------------------------------------
MediaController: Gesture-based media playback and volume management.

Gesture vocabulary (Left hand only)
-------------------------------------

1. Peace Sign  →  Play / Pause toggle
   ─────────────────────────────────────────────────────────────────────────
   Condition A — Index (LM 8) and Middle (LM 12) fingers EXTENDED:
       Fingertip Y  <  Lower joint Y   (tip is above the PIP joint)
       • Index  tip (8)  above Index  PIP (6)
       • Middle tip (12) above Middle PIP (10)

   Condition B — Ring (LM 16) and Pinky (LM 20) fingers CURLED:
       Fingertip Y  >  MCP base knuckle Y  (tip sits below its own base)
       • Ring  tip (16) below Ring  MCP (13)
       • Pinky tip (20) below Pinky MCP (17)

   Cooldown: gesture fires at most once every 1.5 seconds.

2. Pointing  →  Mute / Unmute toggle
   ─────────────────────────────────────────────────────────────────────────
   Only the Index finger (LM 8) is extended; Middle (LM 12), Ring (LM 16),
   and Pinky (LM 20) are all tightly curled below their MCP base knuckles.

   Cooldown: gesture fires at most once every 1.5 seconds.

3. Pinch + Edge-Hold Zone  →  Volume Up / Down
   ─────────────────────────────────────────────────────────────────────────
   Pinch detection:
       Euclidean pixel distance between Thumb tip (4) and Index tip (8) < 30.

   While pinching, the X-position of the Index tip determines the zone
   (assumes a 640-pixel-wide frame):

       RIGHT ZONE  (X > 480) : Volume Up   — fires repeatedly while held,
                                             with a 0.15 s inter-event gap.
       LEFT  ZONE  (X < 160) : Volume Down — same rate-limiting behaviour.
       CENTER ZONE (160–480) : Dead-zone — hand rests here safely; no event.

   This architecture prevents the user's hand from drifting off-camera,
   as the trigger zones are fixed screen edges rather than relative motion.

Note on Y-axis direction
─────────────────────────
In pixel space Y=0 is at the TOP of the frame, so a smaller Y value means
the landmark is physically higher on screen.
"""

import time

from core import geometry as geo, os_input


class MediaController:
    """
    Monitors the left hand for three distinct gestures and fires the
    corresponding OS media key events via pyautogui.

    Gestures
    --------
    • Peace Sign      → play/pause toggle        (1.5 s cooldown)
    • Pointing        → mute/unmute toggle        (1.5 s cooldown)
    • Pinch + Edge    → volume up / volume down   (0.15 s repeat rate)
    """

    # ── Landmark indices ──────────────────────────────────────────────────────
    _THUMB_TIP   = 4

    _INDEX_TIP   = 8
    _INDEX_PIP   = 6    # Proximal interphalangeal — lower joint for extension check
    _INDEX_MCP   = 5    # Metacarpophalangeal — base knuckle for curl check

    _MIDDLE_TIP  = 12
    _MIDDLE_PIP  = 10
    _MIDDLE_MCP  = 9

    _RING_TIP    = 16
    _RING_MCP    = 13   # Metacarpophalangeal — base knuckle for curl check

    _PINKY_TIP   = 20
    _PINKY_MCP   = 17

    # "Pinch engaged" threshold, as a fraction of hand size (core/geometry.py)
    _PINCH_RATIO: float = geo.PINCH_CLICK

    # Edge zones as fractions of the frame width (so any camera resolution works)
    _RIGHT_ZONE_FRAC: float = 0.75   # X > this * width  →  Volume Up zone
    _LEFT_ZONE_FRAC:  float = 0.25   # X < this * width  →  Volume Down zone

    # Minimum seconds between consecutive volume events while held in a zone
    _VOLUME_REPEAT_SEC: float = 0.15

    # Minimum seconds between play/pause triggers
    _PLAYPAUSE_COOLDOWN_SEC: float = 1.5

    # Minimum seconds between mute triggers
    _MUTE_COOLDOWN_SEC: float = 1.5

    # =========================================================================
    # Initialisation
    # =========================================================================

    def __init__(self) -> None:
        # Timestamp of the last play/pause command (epoch seconds).
        self.last_playpause_time: float = 0

        # Timestamp of the last mute toggle command (epoch seconds).
        self.last_mute_time: float = 0

        # Timestamp of the last volume event (epoch seconds).
        self.last_volume_time: float = 0

        # Mutable volume repeat interval (patched by VisionEngine from config).
        self.volume_repeat_sec: float = self._VOLUME_REPEAT_SEC

    # =========================================================================
    # Public API
    # =========================================================================

    def update(
        self,
        detected_hands,
        *,
        enable_media: bool = True,
        enable_volume: bool = True,
        frame_width: int = 640,
    ) -> None:
        """
        Evaluate the current hand data and fire OS media commands as needed.

        Parameters
        ----------
        detected_hands : list[dict] | None
            Structured hand data from HandTracker.get_landmarks():
            [ { "label": "Right"|"Left",
                "landmarks": [(id, x_px, y_px, z), ...] }, ... ]
        enable_media : bool
            When True, the play/pause gesture is active.
        enable_volume : bool
            When True, the mute toggle and volume-zone gestures are active.
        """
        left_hand = self._find_left_hand(detected_hands)

        if left_hand is None:
            return

        lms = left_hand["landmarks"]    # list of (id, x_px, y_px, z)

        if enable_media:
            self._check_play_pause(lms)

        if enable_volume:
            self._check_mute(lms)
            self._check_volume_zones(lms, frame_width)


    # =========================================================================
    # Private — gesture logic
    # =========================================================================

    def _check_play_pause(self, lms) -> None:
        """
        Fire play/pause if the left hand shows a clear Peace Sign and
        the cooldown has elapsed.
        """
        if self._is_peace_sign(lms):
            now = time.time()
            if now - self.last_playpause_time > self._PLAYPAUSE_COOLDOWN_SEC:
                os_input.press("playpause")
                print("--> OS COMMAND: PLAY/PAUSE TOGGLE")
                self.last_playpause_time = now

    def _check_mute(self, lms) -> None:
        """
        Fire mute/unmute if the left hand shows a clear Pointing gesture
        (only index extended) and the cooldown has elapsed.
        """
        if self._is_pointing(lms):
            now = time.time()
            if now - self.last_mute_time > self._MUTE_COOLDOWN_SEC:
                os_input.press("volumemute")
                print("--> OS COMMAND: MUTE TOGGLE")
                self.last_mute_time = now

    def _check_volume_zones(self, lms, frame_width: int = 640) -> None:
        """
        While a pinch is held, map the Index tip's horizontal screen position
        to volume up / down using fixed edge zones.

        Zone layout (fractions of the frame width):
          [0 ──── 25%]   LEFT ZONE  → Volume Down (repeat every 0.15 s)
          [25% ── 75%]   DEAD ZONE  → no action
          [75% ── 100%]  RIGHT ZONE → Volume Up   (repeat every 0.15 s)
        """
        if not self._is_pinching(lms):
            return

        _, x8, _, _ = lms[self._INDEX_TIP]   # Index tip X in pixels
        now = time.time()
        right_x = self._RIGHT_ZONE_FRAC * frame_width
        left_x = self._LEFT_ZONE_FRAC * frame_width

        if x8 > right_x:
            # ── RIGHT ZONE: Volume Up ─────────────────────────────────────────
            if now - self.last_volume_time > self.volume_repeat_sec:
                os_input.press("volumeup")
                print("--> OS COMMAND: VOLUME UP")
                self.last_volume_time = now

        elif x8 < left_x:
            # ── LEFT ZONE: Volume Down ────────────────────────────────────────
            if now - self.last_volume_time > self.volume_repeat_sec:
                os_input.press("volumedown")
                print("--> OS COMMAND: VOLUME DOWN")
                self.last_volume_time = now

        # ── CENTER DEAD ZONE (160 ≤ x8 ≤ 480): do nothing ────────────────────

    # =========================================================================
    # Private — geometry helpers
    # =========================================================================

    def _find_left_hand(self, detected_hands) -> dict | None:
        """Return the first dict whose label is 'Left', or None."""
        if not detected_hands:
            return None
        return next((h for h in detected_hands if h["label"] == "Left"), None)

    def _is_peace_sign(self, lms) -> bool:
        """
        Index and middle extended; ring and pinky curled tight (tip folded in
        closer to the wrist than the finger's base knuckle).  Rotation-invariant.
        """
        return (geo.finger_extended(lms, "index")
                and geo.finger_extended(lms, "middle")
                and geo.finger_curled_tight(lms, "ring")
                and geo.finger_curled_tight(lms, "pinky"))

    def _is_pointing(self, lms) -> bool:
        """
        ONLY the index finger extended; middle, ring and pinky curled tight.
        Rotation-invariant.
        """
        return (geo.finger_extended(lms, "index")
                and geo.finger_curled_tight(lms, "middle")
                and geo.finger_curled_tight(lms, "ring")
                and geo.finger_curled_tight(lms, "pinky"))

    def _is_pinching(self, lms) -> bool:
        """Thumb tip and index tip closer than _PINCH_RATIO of the hand size."""
        return geo.pinch_ratio(lms, self._THUMB_TIP, self._INDEX_TIP) < self._PINCH_RATIO
