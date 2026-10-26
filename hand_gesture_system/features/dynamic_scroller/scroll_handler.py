"""
features/dynamic_scroller/scroll_handler.py
---------------------------------------------
DynamicScroller: Proportional, zone-based page scrolling via an open-palm
gesture on the Right hand.

Zone Architecture (default 480 px frame height)
-------------------------------------------------
  UPPER Scroll Zone : Y < 190  →  os_input.scroll(+clicks)   (scroll UP)
  CENTER Deadzone   : 190 ≤ Y ≤ 290  →  no action
  LOWER Scroll Zone : Y > 290  →  os_input.scroll(−clicks)   (scroll DOWN)

  The deadzone boundaries derive from:
      upper_boundary = center_y − deadzone_radius  =  240 − 50  = 190
      lower_boundary = center_y + deadzone_radius  =  240 + 50  = 290

Engagement Prerequisite — Open Palm (Right hand only)
-------------------------------------------------------
  The scroller activates ONLY when all four finger tips are clearly extended:
      Tip Y (8, 12, 16, 20) < their respective PIP Y (6, 10, 14, 18)
  A fist, a pinch, or any other non-open posture causes an immediate return
  with no scroll event fired.

Proportional Scroll Strength
------------------------------
  The further the palm centre (LM 9) strays from the deadzone boundary,
  the more scroll clicks are sent per frame:

      distance = |palm_Y − boundary|
      clicks   = int(distance × scroll_speed_modifier × (480 / frame_height))

  scroll_speed_modifier defaults to 0.1, so a palm 30 px above the boundary
  yields 3 clicks per frame.

Note on Y-axis direction
─────────────────────────
  In pixel space Y=0 is at the TOP of the frame, so a smaller Y value means
  the landmark is physically higher on screen.  Scrolling UP therefore
  corresponds to the palm being in the UPPER zone (small Y).
"""

from core import geometry as geo, os_input


# ---------------------------------------------------------------------------
# Landmark indices
# ---------------------------------------------------------------------------
_PALM_LM = 9   # Middle-finger MCP — stable geometric palm centre



class DynamicScroller:
    """
    Monitors the right hand's open-palm position and fires proportional
    pyautogui scroll events based on how far the palm sits above or below
    the central deadzone.

    Parameters
    ----------
    frame_h : int
        Frame height in pixels; the scroll zones are centred on frame_h / 2.
        (Call set_frame_size() if the camera resolution changes.)
    deadzone_frac : float
        Half-height of the neutral band, as a fraction of frame height
        (default 50/480, i.e. 190-290 px on a 480 px frame).  No scrolling
        occurs while the palm is within this band.
    scroll_speed_modifier : float
        Multiplier applied to pixel distance before converting to integer
        scroll-click count.  Reduce for slower scrolling, increase for faster.
    """

    # =========================================================================
    # Initialisation
    # =========================================================================

    def __init__(
        self,
        frame_h: int                = 480,
        deadzone_frac: float        = 50 / 480,
        scroll_speed_modifier: float = 0.1,
    ) -> None:
        self.deadzone_frac         = deadzone_frac
        self.scroll_speed_modifier = scroll_speed_modifier
        self.set_frame_size(frame_h)

    def set_frame_size(self, frame_h: int) -> None:
        """(Re)compute the neutral band from the real frame height.

        The band is centred vertically and is ``deadzone_frac`` of the frame
        height on each side of centre (about 190-290 px on a 480 px frame).
        """
        self.center_y        = frame_h // 2
        self._speed_scale    = 480.0 / max(frame_h, 1)   # same feel at any resolution
        self.deadzone_radius = int(round(self.deadzone_frac * frame_h))
        self._upper_boundary: int = self.center_y - self.deadzone_radius
        self._lower_boundary: int = self.center_y + self.deadzone_radius

    # =========================================================================
    # Public API
    # =========================================================================

    def update(self, detected_hands) -> None:
        """
        Process one frame of hand data and fire scroll events as needed.

        Parameters
        ----------
        detected_hands : list[dict] | None
            Structured hand data from HandTracker.get_landmarks():
            [ { "label": "Right"|"Left",
                "landmarks": [(id, x_px, y_px, z), ...] }, ... ]
        """
        right_hand = self._find_right_hand(detected_hands)

        if right_hand is None:
            return

        lms = right_hand["landmarks"]   # list of (id, x_px, y_px, z)

        # ── Open-palm engagement gate ─────────────────────────────────────────
        # Only scroll when the hand is fully open; reject fists, pinches, etc.
        if not self._is_open_palm(lms):
            return

        # ── Palm Y position ───────────────────────────────────────────────────
        _, _, palm_y, _ = lms[_PALM_LM]

        # ── Zone classification + proportional scroll ─────────────────────────
        if palm_y < self._upper_boundary:
            # Hand raised into upper zone → scroll UP.
            distance = self._upper_boundary - palm_y
            clicks   = int(distance * self._speed_scale * self.scroll_speed_modifier)
            if clicks > 0:
                os_input.scroll(clicks)
                print(f"--> OS COMMAND: SCROLL UP ({clicks})")

        elif palm_y > self._lower_boundary:
            # Hand lowered into lower zone → scroll DOWN.
            distance = palm_y - self._lower_boundary
            clicks   = int(distance * self._speed_scale * self.scroll_speed_modifier)
            if clicks > 0:
                os_input.scroll(-clicks)
                print(f"--> OS COMMAND: SCROLL DOWN ({clicks})")

        # Palm inside deadzone → do nothing.

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _find_right_hand(self, detected_hands) -> dict | None:
        """Return the first dict whose label is 'Right', or None."""
        if not detected_hands:
            return None
        return next((h for h in detected_hands if h["label"] == "Right"), None)

    def _is_open_palm(self, lms) -> bool:
        """
        True only when all four fingers are extended (tip farther from the
        wrist than the PIP joint).  Rejects fists, pinches, pointing, peace
        signs and thumbs-up at any hand rotation.
        """
        return geo.four_fingers_extended(lms)
