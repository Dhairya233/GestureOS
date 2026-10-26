"""
features/air_transformer/transformer_handler.py
------------------------------------------------
Implements the 'Air Transformer' gesture:

  - Hold both hands in frame and pinch (thumb tip ↔ index tip) with BOTH hands.
  - While pinching:
      • Move hands apart / together  → Zoom the virtual box in / out.
      • Rotate the line between the two index tips → Rotate the virtual box.
      • The box centre tracks the midpoint between the two index tips.
  - Release either pinch → box locks in place (grey).

Landmark indices used (MediaPipe 21-point hand model):
  4  = Thumb tip
  8  = Index finger tip
"""

import math
import cv2
import numpy as np

from core import geometry as geo
from core import tokens as T


class AirTransformer:
    """
    Maintains a virtual bounding box that responds to a two-hand
    pinch-zoom-rotate gesture.

    Parameters
    ----------
    frame_w : int
        Width of the video frame in pixels.  Used to centre the box on init.
    frame_h : int
        Height of the video frame in pixels.
    init_size : int
        Initial side length (px) of the virtual box.
    pinch_ratio : float
        Thumb-to-index distance, as a fraction of hand size, below which a hand
        counts as pinching (core/geometry.py).  Larger = easier to trigger.
    """

    # MediaPipe landmark indices
    _INDEX_TIP = 8
    _THUMB_TIP = 4

    def __init__(
        self,
        frame_w: int = 640,
        frame_h: int = 480,
        init_size: int = 200,
        pinch_ratio: float = geo.PINCH_TRANSFORM,
    ) -> None:

        # ── Virtual box state ────────────────────────────────────────────────
        self.center_x: float = frame_w / 2.0
        self.center_y: float = frame_h / 2.0
        self.width: float    = float(init_size)
        self.height: float   = float(init_size)
        self.rotation_angle: float = 0.0   # degrees, CCW-positive

        # ── Config ───────────────────────────────────────────────────────────
        self._pinch_ratio = pinch_ratio

        # ── Inter-frame state (reset when pinch is lost) ─────────────────────
        self._prev_dist:  float | None = None   # px distance between index tips
        self._prev_angle: float | None = None   # degrees of the connector line

        # ── Draw state (set by update, consumed by draw) ──────────────────────
        self._tracking_active: bool = False
        self._left_tip:  tuple[int, int] | None = None
        self._right_tip: tuple[int, int] | None = None

    # =========================================================================
    # Public API
    # =========================================================================

    def update(self, all_hands) -> None:
        """
        Consume the structured hand data from ``HandTracker.get_landmarks()``
        and update the virtual box state for this frame.

        Parameters
        ----------
        all_hands : list[dict] | None
            Format: [ { "label": "Right"|"Left",
                         "landmarks": [(id, x_px, y_px, z), ...] }, ... ]
        """
        # ── Guard: need at least two hands ───────────────────────────────────
        if not all_hands or len(all_hands) < 2:
            self._reset_tracking()
            return

        # ── Separate by label ────────────────────────────────────────────────
        left_hand  = next((h for h in all_hands if h["label"] == "Left"),  None)
        right_hand = next((h for h in all_hands if h["label"] == "Right"), None)

        if left_hand is None or right_hand is None:
            self._reset_tracking()
            return

        # ── Pinch check ──────────────────────────────────────────────────────
        left_pinching,  left_tip  = self._get_pinch(left_hand)
        right_pinching, right_tip = self._get_pinch(right_hand)

        if not (left_pinching and right_pinching):
            self._reset_tracking()
            return

        # ── Both hands pinching — compute deltas ─────────────────────────────
        lx, ly = left_tip
        rx, ry = right_tip

        # Box centre tracks the midpoint of the two index tips
        self.center_x = (lx + rx) / 2.0
        self.center_y = (ly + ry) / 2.0

        # Euclidean distance between the two tracking points
        curr_dist = math.hypot(rx - lx, ry - ly)

        # Angle (degrees) of the line from left tip → right tip
        curr_angle = math.degrees(math.atan2(ry - ly, rx - lx))

        if self._prev_dist is not None and self._prev_dist > 1e-6:
            # ── Zoom: scale box by the ratio of distances ─────────────────
            scale = curr_dist / self._prev_dist
            self.width  = max(20.0, self.width  * scale)
            self.height = max(20.0, self.height * scale)

            # ── Rotate: accumulate angular change ─────────────────────────
            # Wrap delta to [-180, 180] to avoid discontinuity at ±180°.
            delta = curr_angle - self._prev_angle
            if delta >  180.0:
                delta -= 360.0
            elif delta < -180.0:
                delta += 360.0

            self.rotation_angle += delta

        # Cache for next frame
        self._prev_dist  = curr_dist
        self._prev_angle = curr_angle

        # Store draw-time state
        self._tracking_active = True
        self._left_tip        = (lx, ly)
        self._right_tip       = (rx, ry)

    def draw(self, frame) -> None:
        """
        Render the virtual box onto *frame* in-place.

        Visual cues:
          • Green border + yellow connector line  → actively tracking (pinching).
          • Grey border, no connector line        → gesture lost / box locked.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame to draw on (modified in-place).
        """
        # ── Colour scheme ────────────────────────────────────────────────────
        if self._tracking_active:
            box_colour  = T.hex_to_bgr(T.ARMED)
            line_colour = T.hex_to_bgr(T.TEXT)
            fill_alpha  = 0.12
        else:
            box_colour  = T.hex_to_bgr(T.FAINT)
            line_colour = None
            fill_alpha  = 0.05

        # ── Build rotated rectangle corner points ────────────────────────────
        rect = (
            (self.center_x, self.center_y),
            (self.width, self.height),
            self.rotation_angle,
        )
        box_pts = cv2.boxPoints(rect).astype(np.int32)

        # ── Semi-transparent fill ────────────────────────────────────────────
        overlay = frame.copy()
        cv2.fillPoly(overlay, [box_pts], box_colour)
        cv2.addWeighted(overlay, fill_alpha, frame, 1.0 - fill_alpha, 0, frame)

        # ── Solid border ─────────────────────────────────────────────────────
        cv2.drawContours(frame, [box_pts], 0, box_colour, 2, cv2.LINE_AA)

        # ── Corner accent squares ────────────────────────────────────────────
        for pt in box_pts:
            cv2.rectangle(
                frame,
                (pt[0] - 4, pt[1] - 4),
                (pt[0] + 4, pt[1] + 4),
                box_colour, -1,
            )

        # ── Connector line + pinch-point circles ─────────────────────────────
        if self._tracking_active and line_colour and self._left_tip and self._right_tip:
            cv2.line(frame, self._left_tip, self._right_tip,
                     line_colour, 2, cv2.LINE_AA)

            for pt in (self._left_tip, self._right_tip):
                cv2.circle(frame, pt, 9,  line_colour,    -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 9,  T.hex_to_bgr(T.BG), 1, cv2.LINE_AA)

        # ── Status label ─────────────────────────────────────────────────────
        status     = "tracking" if self._tracking_active else "locked"
        label_text = f"Transformer {status}"
        label_pos  = (12, frame.shape[0] - 40)

        # Shadow
        cv2.putText(frame, label_text, label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        # Foreground
        cv2.putText(frame, label_text, label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_colour, 1, cv2.LINE_AA)

        # ── Box dimensions HUD ───────────────────────────────────────────────
        hud_text = (
            f"W:{int(self.width)}  H:{int(self.height)}"
            f"  Rot:{int(self.rotation_angle % 360):3d}deg"
        )
        hud_pos = (int(self.center_x) - 80, int(self.center_y) - int(self.height / 2) - 10)
        # Clamp to frame bounds
        hud_pos = (max(0, hud_pos[0]), max(14, hud_pos[1]))

        cv2.putText(frame, hud_text, hud_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, hud_text, hud_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_colour, 1, cv2.LINE_AA)

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _get_pinch(self, hand: dict) -> tuple[bool, tuple[int, int]]:
        """
        Check whether a hand is performing a pinch gesture.

        Returns
        -------
        (is_pinching, (index_tip_x, index_tip_y))
        """
        lms = hand["landmarks"]            # list of (id, x_px, y_px, z)
        _, ix, iy, _ = lms[self._INDEX_TIP]
        _, tx, ty, _ = lms[self._THUMB_TIP]

        ratio = geo.pinch_ratio(lms, self._THUMB_TIP, self._INDEX_TIP)
        return ratio < self._pinch_ratio, (ix, iy)

    def _reset_tracking(self) -> None:
        """Called when the pinch gesture is lost — clears inter-frame deltas."""
        self._prev_dist        = None
        self._prev_angle       = None
        self._tracking_active  = False
        self._left_tip         = None
        self._right_tip        = None
