"""
features/mouse_control/mouse_handler.py
-----------------------------------------
MouseController: Gesture-based cursor movement, left-click, right-click, and
continuous drag-and-drop via the Right hand.

Unified State Architecture
───────────────────────────────────────────────────────────────────────────────
The controller activates whenever the Pinky finger is curled.  This single
anchor keeps all four gestures online without conflicting gate conditions,
since the pinky stays curled across hover, click, right-click and drag.

All finger and pinch tests come from core/geometry.py: they compare landmark
*distances* (finger extended = tip farther from the wrist than its PIP joint;
pinch = tip-to-tip distance divided by hand size), so they work at any hand
rotation and any distance from the camera.

Gesture Vocabulary (Right hand only)
───────────────────────────────────────────────────────────────────────────────

1. Primary Gate (always required)
   Pinky NOT extended.

2. Hover / Normal Cursor Movement  (only when NOT dragging)
   Index extended and middle NOT extended.  The index tip is mapped to the
   screen via numpy.interp and smoothed with an exponential filter.

3. Thumb-to-Middle Left Click
   Thumb tip to middle tip < click_ratio of hand size fires a click.
   A 0.5-second cooldown prevents spam.

4. Drag-and-Drop  (Thumb-to-Index Pinch)
   • below drag_engage_ratio  -> mouse down on first entry; the cursor tracks
     the pinch midpoint.
   • above drag_release_ratio -> mouse up.  The gap between the two ratios is
     the hysteresis that prevents flicker.

5. Right Click  (Thumb-to-Ring Pinch)
   Thumb tip to ring tip < right_click_ratio of hand size, 0.6-second
   cooldown, and only when NOT dragging.
"""

import time

import numpy
from core import geometry as geo, os_input


class MouseController:
    """
    Drives OS cursor movement, left-click, right-click, and drag-and-drop
    from right-hand gestures detected by the MediaPipe hand landmark model.

    Parameters
    ----------
    frame_r : int
        Margin (pixels) inset from every webcam edge to form the active
        tracking rectangle.  Larger → more aggressive mapping.
    smoothening : float
        Divisor for the exponential smoothing formula.  Range: 3–10.
    click_ratio : float
        Thumb-to-Middle distance, as a fraction of hand size (see
        core/geometry.py), below which a left-click fires.
    click_cooldown_s : float
        Minimum seconds between successive left-click events.
    drag_engage_ratio : float
        Thumb-to-Index distance (fraction of hand size) below which a drag begins.
    drag_release_ratio : float
        Thumb-to-Index distance (fraction of hand size) above which a drag ends
        (the gap to the engage value is the hysteresis).
    right_click_ratio : float
        Thumb-to-Ring distance (fraction of hand size) that triggers a right-click.
    right_click_cooldown_s : float
        Minimum seconds between successive right-click events.
    """

    # ── Landmark indices ──────────────────────────────────────────────────────
    _THUMB_TIP  = 4

    _INDEX_TIP  = 8
    _INDEX_PIP  = 6

    _MIDDLE_TIP = 12
    _MIDDLE_PIP = 10

    _RING_TIP   = 16
    _RING_PIP   = 14

    _PINKY_TIP  = 20
    _PINKY_PIP  = 18

    # =========================================================================
    # Initialisation
    # =========================================================================

    def __init__(
        self,
        frame_r: int                  = 100,
        smoothening: float            = 5,
        click_ratio: float            = geo.PINCH_CLICK,
        click_cooldown_s: float       = 0.5,
        drag_engage_ratio: float      = geo.PINCH_ENGAGE,
        drag_release_ratio: float     = geo.PINCH_RELEASE,
        right_click_ratio: float      = geo.PINCH_CLICK,
        right_click_cooldown_s: float = 0.6,
    ) -> None:
        # Screen dimensions (pixels) — queried once at startup.
        self.screen_w, self.screen_h = os_input.size()

        # Active tracking region margin (pixels from each webcam edge).
        self.frame_r = frame_r

        # Smoothing divisor — higher is smoother but slower.
        self.smoothening = smoothening

        # Previous and current smoothed cursor positions.
        self.prev_x: float = 0.0
        self.prev_y: float = 0.0
        self.curr_x: float = 0.0
        self.curr_y: float = 0.0

        # ── Left-click state ──────────────────────────────────────────────────
        self.last_click_time: float = 0.0

        # ── Drag-and-drop state ───────────────────────────────────────────────
        self.is_dragging: bool = False

        # ── Right-click state ─────────────────────────────────────────────────
        self.last_right_click_time: float = 0.0

        # ── Internal thresholds (exposed so callers can tune without subclassing)
        self._click_ratio             = click_ratio
        self._click_cooldown_s        = click_cooldown_s
        self._drag_engage_ratio       = drag_engage_ratio
        self._drag_release_ratio      = drag_release_ratio
        self._right_click_ratio       = right_click_ratio
        self._right_click_cooldown_s  = right_click_cooldown_s

    # =========================================================================
    # Public API
    # =========================================================================

    def update(
        self,
        detected_hands,
        frame_width:  int = 640,
        frame_height: int = 480,
        *,
        enable_click:  bool = True,
        enable_rclick: bool = True,
        enable_drag:   bool = True,
    ) -> None:
        """
        Process one frame of hand data and drive the OS cursor / input events.

        Decision flow per frame
        -----------------------
        1. Locate the Right hand.  Bail if absent.
        2. Bail if the Pinky is NOT curled (primary gate).
        3. Evaluate drag state (Thumb-Index pinch) — highest priority.
           • If dragging: move cursor to pinch midpoint and skip hover logic.
           • If not dragging: check for drag engage.
        4. If NOT dragging, evaluate normal hover movement (Index up + Middle curled).
        5. Evaluate left-click (Thumb-Middle pinch) — only when NOT dragging.
        6. Evaluate right-click (Thumb-Ring pinch) — only when NOT dragging.

        Parameters
        ----------
        detected_hands : list[dict] | None
            Output of HandTracker.get_landmarks().
            Each dict: { "label": "Right"|"Left",
                         "landmarks": [(id, x_px, y_px, z), ...] }
        frame_width : int
            Width of the video frame in pixels.
        frame_height : int
            Height of the video frame in pixels.
        enable_click : bool
            When True, the Thumb-Middle left-click gesture is active.
        enable_rclick : bool
            When True, the Thumb-Ring right-click gesture is active.
        enable_drag : bool
            When True, the Thumb-Index drag-and-drop gesture is active.
        """
        right_hand = self._find_right_hand(detected_hands)

        if right_hand is None:
            # Hand left the frame — release any active drag cleanly.
            if self.is_dragging:
                os_input.mouseUp()
                self.is_dragging = False
                print("--> OS STATE: MOUSE UP (DRAG RELEASED — hand lost)")
            return

        lms = right_hand["landmarks"]   # list of (id, x_px, y_px, z)

        # ── Primary gate: Pinky must be curled ───────────────────────────────
        if not self._pinky_curled(lms):
            # Pinky is up — release drag if active, then disengage entirely.
            if self.is_dragging:
                os_input.mouseUp()
                self.is_dragging = False
                print("--> OS STATE: MOUSE UP (DRAG RELEASED — gate lost)")
            return

        # ── Drag-and-Drop (Thumb ↔ Index pinch) — evaluated first ────────────
        if enable_drag:
            self._handle_drag(lms, frame_width, frame_height)
        elif self.is_dragging:
            # Drag was disabled while active — release cleanly.
            os_input.mouseUp()
            self.is_dragging = False
            print("--> OS STATE: MOUSE UP (DRAG RELEASED — feature disabled)")

        # ── Normal hover cursor movement (only when not dragging) ─────────────
        if not self.is_dragging:
            self._handle_hover(lms, frame_width, frame_height)

        # ── Left click (Thumb ↔ Middle pinch) — only when not dragging ────────
        if not self.is_dragging and enable_click:
            self._check_left_click(lms)

        # ── Right click (Thumb ↔ Ring pinch) — only when not dragging ─────────
        if not self.is_dragging and enable_rclick:
            self._check_right_click(lms)


    # =========================================================================
    # Private — gesture handlers
    # =========================================================================

    def _handle_drag(self, lms, frame_width: int, frame_height: int) -> None:
        """
        Manage drag-and-drop state based on Thumb-to-Index tip distance.

        • Distance < drag_engage_px  → start or sustain a drag.
        • Distance > drag_release_px → end a drag (hysteresis gap prevents flicker).

        While dragging, the cursor tracks the midpoint between Thumb and Index
        tips so the user has an intuitive anchor for the dragged object.
        """
        _, x4, y4, _   = lms[self._THUMB_TIP]
        _, x8, y8, _   = lms[self._INDEX_TIP]

        distance = geo.pinch_ratio(lms, self._THUMB_TIP, self._INDEX_TIP)

        if distance < self._drag_engage_ratio:
            # ── Pinch is active ───────────────────────────────────────────────
            if not self.is_dragging:
                if not os_input.is_armed():
                    return          # disarmed: never start a drag
                os_input.mouseDown()
                self.is_dragging = True
                print("--> OS STATE: MOUSE DOWN (DRAG STARTED)")

            # Move cursor to the pinch midpoint (smoothed).
            mid_x = (x4 + x8) / 2
            mid_y = (y4 + y8) / 2
            self._move_cursor(mid_x, mid_y, frame_width, frame_height)

        elif distance > self._drag_release_ratio and self.is_dragging:
            # ── Pinch released beyond hysteresis threshold ────────────────────
            os_input.mouseUp()
            self.is_dragging = False
            print("--> OS STATE: MOUSE UP (DRAG RELEASED)")

    def _handle_hover(self, lms, frame_width: int, frame_height: int) -> None:
        """
        Move the cursor smoothly using the Index finger tip position when the
        hand is in the standard pointing posture (Index extended, Middle curled).
        """
        if not self._is_pointing(lms):
            return

        _, x_tip, y_tip, _ = lms[self._INDEX_TIP]
        self._move_cursor(x_tip, y_tip, frame_width, frame_height)

    def _check_left_click(self, lms) -> None:
        """
        Fire a left-click when the Thumb tip (4) and Middle tip (12) pinch
        to within click_threshold_px, subject to the cooldown guard.
        """

        distance = geo.pinch_ratio(lms, self._THUMB_TIP, self._MIDDLE_TIP)

        if distance < self._click_ratio:
            now = time.time()
            if now - self.last_click_time > self._click_cooldown_s:
                os_input.click()
                print("--> OS COMMAND: LEFT CLICK")
                self.last_click_time = now

    def _check_right_click(self, lms) -> None:
        """
        Fire a right-click when the Thumb tip (4) and Ring tip (16) pinch
        to within right_click_threshold_px, subject to the cooldown guard.
        """

        distance = geo.pinch_ratio(lms, self._THUMB_TIP, self._RING_TIP)

        if distance < self._right_click_ratio:
            now = time.time()
            if now - self.last_right_click_time > self._right_click_cooldown_s:
                os_input.rightClick()
                print("--> OS COMMAND: RIGHT CLICK")
                self.last_right_click_time = now

    # =========================================================================
    # Private — cursor movement helper
    # =========================================================================

    def _move_cursor(
        self, cam_x: float, cam_y: float, frame_width: int, frame_height: int
    ) -> None:
        """
        Map a camera-space (cam_x, cam_y) coordinate to screen space and apply
        exponential smoothing before calling os_input.moveTo().
        """
        r = self.frame_r
        mapped_x = numpy.interp(cam_x, [r, frame_width  - r], [0, self.screen_w])
        mapped_y = numpy.interp(cam_y, [r, frame_height - r], [0, self.screen_h])

        self.curr_x = self.prev_x + (mapped_x - self.prev_x) / self.smoothening
        self.curr_y = self.prev_y + (mapped_y - self.prev_y) / self.smoothening

        os_input.moveTo(self.curr_x, self.curr_y)

        self.prev_x = self.curr_x
        self.prev_y = self.curr_y

    # =========================================================================
    # Private — geometry helpers
    # =========================================================================

    def _find_right_hand(self, detected_hands) -> dict | None:
        """Return the first dict whose label is 'Right', or None."""
        if not detected_hands:
            return None
        return next((h for h in detected_hands if h["label"] == "Right"), None)

    def _pinky_curled(self, lms) -> bool:
        """
        Return True when the Pinky tip (20) sits BELOW the Pinky PIP (18),
        i.e. tip Y > PIP Y in pixel space.  This is the primary activation
        gate that is common to all four mouse gestures.
        """
        return not geo.finger_extended(lms, "pinky")

    def _is_pointing(self, lms) -> bool:
        """
        Return True when the hand is in the standard hover/pointing posture:
          • Index tip (8)  ABOVE Index PIP (6)   — finger extended.
          • Middle tip (12) BELOW Middle PIP (10) — finger curled.

        Ring and Pinky are not re-checked here; Pinky is already confirmed by
        the primary gate, and Ring is allowed to be free for right-click.
        """
        return (geo.finger_extended(lms, "index")
                and not geo.finger_extended(lms, "middle"))

    def reset(self) -> None:
        """Forget drag state (called when GestureOS is disarmed)."""
        self.is_dragging = False
