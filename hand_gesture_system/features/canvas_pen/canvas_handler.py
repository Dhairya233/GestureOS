"""
features/canvas_pen/canvas_handler.py
---------------------------------------
CanvasController: Air-Canvas drawing on a persistent ink layer via the
Right hand, with a gesture-driven color picker and board eraser.

Gesture Vocabulary (Right hand only)
───────────────────────────────────────────────────────────────────────────────

STATE 1 — HOVER & SELECT  (Index + Middle extended; Ring + Pinky curled)
─────────────────────────────────────────────────────────────────────────
  The controller is in a non-drawing "hover" mode.
  • self.prev_x/y are reset to 0 so the next drawn line starts fresh from
    wherever the fingertip lands, avoiding a long streak across the canvas.
  • If the Index tip enters the top header strip (the top 70/480 of the frame
    height) the X coordinate picks one of five equal-width zones:

    Zone (fraction of width) │ Action
    ─────────────────────────┼──────────────────────────────────────
    0 – 20 %                 │ Set color → Blue   (255, 0, 0)  BGR
    20 – 40 %                │ Set color → Green  (0, 255, 0)
    40 – 60 %                │ Set color → Red    (0, 0, 255)
    60 – 80 %                │ Set color → Yellow (0, 255, 255)
    80 – 100 %               │ Clear canvas (reset to black)

STATE 2 — DRAW MODE  (ONLY Index extended; Middle + Ring + Pinky curled)
────────────────────────────────────────────────────────────────────────
  • A thick line is drawn on the persistent canvas layer from the previous
    Index tip position to the current one (7 px, active draw_color).
  • prev_x/y are seeded from the current position on the first frame of a
    new stroke so no artefact from the previous hover position appears.

Finger extension test
─────────────────────
  Tip Y < PIP Y  →  extended  (tip is higher on screen in pixel space).

Public interface
────────────────
  update(detected_hands, frame) → np.ndarray
      Call once per frame.  Returns the persistent canvas array
      (same shape as frame) so the caller can blend it onto the live feed.
"""

import cv2
import numpy as np

from core import geometry as geo


class CanvasController:
    """
    Maintains a persistent ink layer and updates it frame-by-frame based
    on Right-hand gestures.

    Parameters
    ----------
    brush_thickness : int
        Pixel width of drawn strokes (default 7).
    """

    # ── Landmark indices ──────────────────────────────────────────────────────
    _INDEX_TIP  = 8
    _INDEX_PIP  = 6
    _MIDDLE_TIP = 12
    _MIDDLE_PIP = 10
    _RING_TIP   = 16
    _RING_PIP   = 14
    _PINKY_TIP  = 20
    _PINKY_PIP  = 18

    # ── Colour-picker strip ───────────────────────────────────────────────────
    # Five equal zones across the frame width (Blue, Green, Red, Yellow, Clear),
    # in a header whose height is a fraction of the frame height (70/480).
    PALETTE_LABELS = ("BLUE", "GREEN", "RED", "YELLOW", "CLEAR")
    _HEADER_FRAC = 70 / 480

    # =========================================================================
    # Initialisation
    # =========================================================================

    def __init__(
        self,
        brush_thickness: int = 7,
    ) -> None:
        # Persistent ink layer — initialised lazily on the first frame so we
        # don't need to know the frame dimensions at construction time.
        self.canvas: np.ndarray | None = None

        # Previous stroke endpoint (reset between strokes to prevent streaks).
        self.prev_x: int = 0
        self.prev_y: int = 0

        # Active drawing color (BGR).  Defaults to Blue.
        self.draw_color: tuple = (255, 0, 0)

        # Available palette colors in zone order: Blue, Green, Red, Yellow.
        self.colors: list = [
            (255,   0,   0),   # Blue
            (  0, 255,   0),   # Green
            (  0,   0, 255),   # Red
            (  0, 255, 255),   # Yellow
        ]

        # Configuration.
        self._frame_w = 640
        self._frame_h = 480
        self._brush_thickness = brush_thickness

    # =========================================================================
    # Public API
    # =========================================================================

    @staticmethod
    def zone_edges(frame_w: int) -> list:
        """X pixel edges of the five colour zones for a frame of this width."""
        return [round(i * frame_w / 5) for i in range(6)]

    @classmethod
    def header_height(cls, frame_h: int) -> int:
        """Height in pixels of the colour strip for a frame of this height."""
        return max(1, round(cls._HEADER_FRAC * frame_h))

    def palette_layout(self, frame_w: int, frame_h: int):
        """[(x1, x2, bgr_colour, label)] for drawing the strip on the overlay."""
        edges = self.zone_edges(frame_w)
        colours = self.colors + [(100, 100, 100)]
        return [(edges[i], edges[i + 1], colours[i], self.PALETTE_LABELS[i])
                for i in range(5)]

    def update(self, detected_hands, frame: np.ndarray) -> np.ndarray:
        """
        Process one frame and update the persistent canvas layer.

        Parameters
        ----------
        detected_hands : list[dict] | None
            Output of HandTracker.get_landmarks().
            Each dict: { "label": "Right"|"Left",
                         "landmarks": [(id, x_px, y_px, z), ...] }
        frame : np.ndarray
            The current BGR webcam frame (used only to derive dimensions on
            the first call — it is NOT modified here).

        Returns
        -------
        np.ndarray
            The persistent canvas array (same H×W×3 as frame).  The caller
            should blend this onto the live feed with cv2.addWeighted.
        """
        # ── Lazy canvas initialisation ────────────────────────────────────────
        if self.canvas is None or self.canvas.shape != frame.shape:
            self.canvas = np.zeros_like(frame)   # first frame / resolution change
        self._frame_h, self._frame_w = frame.shape[:2]

        # ── Locate the Right hand ─────────────────────────────────────────────
        right_hand = self._find_right_hand(detected_hands)

        if right_hand is None:
            # No Right hand visible — reset prev so next stroke starts fresh.
            self.prev_x = 0
            self.prev_y = 0
            return self.canvas

        lms = right_hand["landmarks"]   # list of (id, x_px, y_px, z)

        # ── Finger extension flags ────────────────────────────────────────────
        index_up  = self._is_extended(lms, self._INDEX_TIP,  self._INDEX_PIP)
        middle_up = self._is_extended(lms, self._MIDDLE_TIP, self._MIDDLE_PIP)
        ring_up   = self._is_extended(lms, self._RING_TIP,   self._RING_PIP)
        pinky_up  = self._is_extended(lms, self._PINKY_TIP,  self._PINKY_PIP)

        # Current Index tip coordinates.
        _, cx, cy, _ = lms[self._INDEX_TIP]

        # ── STATE 1: HOVER & SELECT ───────────────────────────────────────────
        # Index + Middle extended; Ring + Pinky curled.
        if index_up and middle_up and not ring_up and not pinky_up:
            self._handle_hover(cx, cy, frame)

        # ── STATE 2: DRAW MODE ────────────────────────────────────────────────
        # ONLY Index extended; Middle + Ring + Pinky curled.
        elif index_up and not middle_up and not ring_up and not pinky_up:
            self._handle_draw(cx, cy)

        else:
            # Any other posture — lift the virtual pen (reset prev).
            self.prev_x = 0
            self.prev_y = 0

        return self.canvas

    # =========================================================================
    # Private — state handlers
    # =========================================================================

    def _handle_hover(self, cx: int, cy: int, frame: np.ndarray) -> None:
        """
        Hover / color-select state.

        Resets the previous stroke endpoint so lines don't bleed across the
        canvas when transitioning back to draw mode.  If the fingertip is
        inside the top header strip, the X position selects a palette color
        or triggers a canvas clear.
        """
        # Lift pen — next draw stroke starts fresh.
        self.prev_x = 0
        self.prev_y = 0

        # Only act on deliberate header entry.
        if cy >= self.header_height(self._frame_h):
            return

        zone = min(int(cx * 5 / max(self._frame_w, 1)), 4)
        if zone < 4:
            self.draw_color = self.colors[zone]
            print(f"[CANVAS] Color selected: {self.PALETTE_LABELS[zone].title()}")
        else:
            # Clear zone (last fifth of the width).
            self.canvas = np.zeros_like(frame)
            print("[CANVAS] Board cleared")

    def _handle_draw(self, cx: int, cy: int) -> None:
        """
        Draw mode: paint a continuous stroke from the previous Index tip
        position to the current one on the persistent canvas layer.
        """
        # Seed the first point of a new stroke without drawing a stray line.
        if self.prev_x == 0 and self.prev_y == 0:
            self.prev_x = cx
            self.prev_y = cy

        cv2.line(
            self.canvas,
            (self.prev_x, self.prev_y),
            (cx, cy),
            self.draw_color,
            self._brush_thickness,
        )

        self.prev_x = cx
        self.prev_y = cy

    # =========================================================================
    # Private — geometry helpers
    # =========================================================================

    def _find_right_hand(self, detected_hands) -> dict | None:
        """Return the first dict whose label is 'Right', or None."""
        if not detected_hands:
            return None
        return next((h for h in detected_hands if h["label"] == "Right"), None)

    def _is_extended(self, lms, tip_idx: int, pip_idx: int) -> bool:
        """True when the finger is extended (tip farther from the wrist than
        its PIP joint; rotation-invariant, see core/geometry.py)."""
        return geo.dist(lms, tip_idx, geo.WRIST) > geo.dist(lms, pip_idx, geo.WRIST)
