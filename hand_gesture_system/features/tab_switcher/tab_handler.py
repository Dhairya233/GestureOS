"""
features/tab_switcher/tab_handler.py
--------------------------------------
EdgeJoystick: Zone-based OS window navigation via palm position.

Zone Architecture (default, 640 px frame width)
-------------------------------------------------
  LEFT  Zone  :    0 – 160 px  → Alt+Shift+Tab  (previous window)
  CENTER Deadzone : 161 – 479 px  → Idle / system ready
  RIGHT Zone  :  480 – 640 px  → Alt+Tab        (next window)

Hold logic
----------
  • On entering an action zone: fire the hotkey immediately, start timer.
  • If the palm stays inside that zone for > INITIAL_HOLD_S seconds:
    continue firing every CYCLE_INTERVAL_S seconds until the palm
    returns to the Center Deadzone.
  • On leaving any action zone: reset cycle count and timer.

Engagement Prerequisite — Open Palm
------------------------------------
  Zone checks only activate when the right hand is in an open-palm posture.
  "Open" means the tips of all four fingers (LM 8, 12, 16, 20) are clearly
  higher on screen (smaller Y) than their PIP joints (LM 6, 10, 14, 18).
  A fist, a pinch, a peace sign, or any other non-open posture immediately
  resets the state to IDLE and skips zone detection entirely, preventing
  accidental window switches.

Public attributes (read by main.py for OSD)
--------------------------------------------
  state         : str   – "IDLE" | "CENTER" | "CYCLING_RIGHT" | "CYCLING_LEFT"
  cycle_count   : int   – windows switched in the current continuous cycle
  palm_x        : int   – last known palm X pixel (None when no hand visible)
"""

import time
import cv2
from core import geometry as geo, os_input
from core import tokens as T


# ---------------------------------------------------------------------------
# Landmark indices
# ---------------------------------------------------------------------------
# Landmark 9 = middle-finger MCP (base of middle finger).
# It sits in the geometric centre of the palm and is the most stable
# single landmark for position tracking.
_PALM_LM = 9

# A finger is "extended" when tip Y < PIP Y (tip is higher on screen).


class EdgeJoystick:
    """
    Divides the frame width into Left / Center / Right zones and uses
    the right hand's palm position to trigger OS window-switching hotkeys.

    Parameters
    ----------
    frame_w : int
        Width of the video frame in pixels.
    left_zone_max : int
        Rightmost pixel still inside the Left zone.
    right_zone_min : int
        Leftmost pixel inside the Right zone.
    initial_hold_s : float
        Seconds the palm must remain in a zone before continuous cycling
        begins.
    cycle_interval_s : float
        Seconds between each repeated hotkey fire during continuous
        cycling.
    """

    def __init__(
        self,
        frame_w: int            = 640,
        left_zone_frac: float   = 0.25,   # palm in the left 25% of the frame
        right_zone_frac: float  = 0.75,   # palm in the right 25% of the frame
        initial_hold_s: float   = 1.5,   # hand must stay in zone 1.5 s before cycling
        cycle_interval_s: float = 1.2,   # subsequent switches fire every 1.2 s
    ) -> None:

        self._left_zone_frac   = left_zone_frac
        self._right_zone_frac  = right_zone_frac
        self.set_frame_size(frame_w)
        self._initial_hold_s   = initial_hold_s
        self._cycle_interval_s = cycle_interval_s

        # ── Public state ─────────────────────────────────────────────────────
        self.state:       str      = "IDLE"   # IDLE / CENTER / CYCLING_RIGHT / CYCLING_LEFT
        self.cycle_count: int      = 0
        self.palm_x:      int|None = None

        # ── Private timing ────────────────────────────────────────────────────
        self._active_zone:       str|None   = None   # "LEFT" / "CENTER" / "RIGHT" / None
        self._zone_entry_time:   float|None = None
        self._last_trigger_time: float|None = None

    # =========================================================================
    # Public API
    # =========================================================================

    def set_frame_size(self, frame_w: int) -> None:
        """(Re)compute the zone edges from the real frame width."""
        self._frame_w        = frame_w
        self._left_zone_max  = int(round(self._left_zone_frac * frame_w))
        self._right_zone_min = int(round(self._right_zone_frac * frame_w))

    def update(self, detected_hands) -> None:
        """
        Process one frame of hand data and fire hotkeys as needed.

        Parameters
        ----------
        detected_hands : list[dict] | None
            Output of HandTracker.get_landmarks().
            Each dict: { "label": "Right"|"Left",
                         "landmarks": [(id, x_px, y_px, z), ...] }
        """
        now = time.monotonic()

        # ── Locate right hand ─────────────────────────────────────────────────
        right_hand = self._find_right_hand(detected_hands)

        if right_hand is None:
            self._reset()
            return

        # ── Palm X position ───────────────────────────────────────────────────
        lms = right_hand["landmarks"]
        _, x_px, _, _ = lms[_PALM_LM]
        self.palm_x = x_px

        # ── Open-palm engagement gate ─────────────────────────────────────────
        # Zone logic is only permitted when all four fingers are clearly open.
        # Any other posture (fist, pinch, peace sign, etc.) forces IDLE so that
        # MediaController and other modules can operate without interference.
        if not self._is_open_palm(lms):
            self._reset_to_idle()
            return

        # ── Zone classification ───────────────────────────────────────────────
        if x_px <= self._left_zone_max:
            zone = "LEFT"
        elif x_px >= self._right_zone_min:
            zone = "RIGHT"
        else:
            zone = "CENTER"

        # ── Zone transition handling ──────────────────────────────────────────
        if zone != self._active_zone:
            self._active_zone = zone

            if zone == "CENTER":
                # Hand returned to idle — reset without firing.
                self.state       = "CENTER"
                self.cycle_count = 0
                self._zone_entry_time   = None
                self._last_trigger_time = None

            else:
                # Entered an action zone — fire immediately on entry.
                self._zone_entry_time   = now
                self._last_trigger_time = now
                self.cycle_count        = 0
                self._fire(zone, now)

        # ── Sustained hold — continue cycling while inside an action zone ─────
        elif zone in ("LEFT", "RIGHT") and self._zone_entry_time is not None:
            held_s      = now - self._zone_entry_time
            since_last  = now - (self._last_trigger_time or now)

            if held_s >= self._initial_hold_s and since_last >= self._cycle_interval_s:
                self._fire(zone, now)

    def draw_osd(self, frame) -> None:
        """
        Render the Edge-Joystick heads-up display onto *frame* in-place.

        Draws:
          • Faint vertical zone-boundary lines.
          • Translucent zone tint (left / right) when active.
          • Top-of-frame action banner when cycling.
          • Subtle centre-zone ready notice.
          • Palm position indicator dot.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame to annotate (modified in-place).
        """
        fh, fw = frame.shape[:2]
        lx = self._left_zone_max    # 160
        rx = self._right_zone_min   # 480

        # ── Zone tint overlays ────────────────────────────────────────────────
        overlay = frame.copy()

        if self.state == "CYCLING_LEFT":
            # Blue tint over left zone
            cv2.rectangle(overlay, (0, 0), (lx, fh), T.hex_to_bgr(T.TEXT), -1)
            cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

        elif self.state == "CYCLING_RIGHT":
            # Orange tint over right zone
            overlay2 = frame.copy()
            cv2.rectangle(overlay2, (rx, 0), (fw, fh), T.hex_to_bgr(T.TEXT), -1)
            cv2.addWeighted(overlay2, 0.12, frame, 0.88, 0, frame)

        # ── Zone boundary lines ───────────────────────────────────────────────
        line_colour = T.hex_to_bgr(T.FAINT)
        cv2.line(frame, (lx, 0), (lx, fh), line_colour, 1, cv2.LINE_AA)
        cv2.line(frame, (rx, 0), (rx, fh), line_colour, 1, cv2.LINE_AA)

        # Tiny zone labels near the top
        for label, pos in (("LEFT", 6), ("CENTER", lx + 8), ("RIGHT", rx + 6)):
            cv2.putText(frame, label, (pos, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, T.hex_to_bgr(T.MUTED), 1, cv2.LINE_AA)

        # ── Action banner (top of frame) ──────────────────────────────────────
        if self.state in ("CYCLING_RIGHT", "CYCLING_LEFT"):
            if self.state == "CYCLING_RIGHT":
                banner_text = f"NEXT WINDOW  +{self.cycle_count}"
            else:
                banner_text = f"PREVIOUS WINDOW  -{self.cycle_count}"

            # Semi-transparent banner background strip
            banner_h = 44
            banner_overlay = frame.copy()
            cv2.rectangle(banner_overlay, (0, 0), (fw, banner_h), T.hex_to_bgr(T.BG), -1)
            cv2.addWeighted(banner_overlay, 0.75, frame, 0.25, 0, frame)
            cv2.putText(frame, banner_text, (12, 29),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, T.hex_to_bgr(T.TEXT), 1, cv2.LINE_AA)

        # ── Palm position dot ─────────────────────────────────────────────────
        if self.palm_x is not None:
            # Find the right hand's palm-centre Y from stored data is not
            # accessible here, so draw a vertical indicator line instead.
            dot_colour = T.hex_to_bgr(T.TEXT if self.state in ("CYCLING_RIGHT", "CYCLING_LEFT") else T.FAINT)

            cv2.line(frame, (self.palm_x, fh - 8), (self.palm_x, fh),
                     dot_colour, 3, cv2.LINE_AA)
            cv2.circle(frame, (self.palm_x, fh - 12), 5, dot_colour, -1, cv2.LINE_AA)

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _find_right_hand(self, detected_hands) -> dict | None:
        if not detected_hands:
            return None
        return next((h for h in detected_hands if h["label"] == "Right"), None)

    def _is_open_palm(self, lms) -> bool:
        """True only when all four fingers are extended, at any rotation."""
        return geo.four_fingers_extended(lms)

    def _reset_to_idle(self) -> None:
        """
        Soft reset: clears state and timers but keeps palm_x so the OSD
        position indicator stays visible while the hand is in frame.
        The OSD banner is suppressed automatically because state == "IDLE"
        has no corresponding draw branch in draw_osd().
        """
        self.state              = "IDLE"
        self.cycle_count        = 0
        self._active_zone       = None
        self._zone_entry_time   = None
        self._last_trigger_time = None

    def _fire(self, zone: str, now: float) -> None:
        """Execute the hotkey and update state + counter."""
        if zone == "RIGHT":
            self.state = "CYCLING_RIGHT"
            print(f"--> EXECUTING HOTKEY: ALT+TAB  (palm_x={self.palm_x}px)")
            os_input.hotkey("alt", "tab")
        else:
            self.state = "CYCLING_LEFT"
            print(f"--> EXECUTING HOTKEY: ALT+SHIFT+TAB  (palm_x={self.palm_x}px)")
            os_input.hotkey("alt", "shift", "tab")

        self.cycle_count        += 1
        self._last_trigger_time  = now
        print(f"[EDGE JOYSTICK]  state={self.state:<15}  count={self.cycle_count}")

    def _reset(self) -> None:
        """Called when the right hand leaves the frame entirely."""
        self.state              = "IDLE"
        self.cycle_count        = 0
        self.palm_x             = None
        self._active_zone       = None
        self._zone_entry_time   = None
        self._last_trigger_time = None
