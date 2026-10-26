"""
core/hand_tracker.py
--------------------
Responsible SOLELY for:
  1. Initialising the MediaPipe HandLandmarker (Tasks API, v0.10+).
  2. Detecting hand presence in an OpenCV BGR frame.
  3. Extracting the 21 normalised (x, y, z) landmarks for detected hands.

Requires the bundled `hand_landmarker.task` model file (float16, v1).
Download once:
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

This module is intentionally unaware of:
  - Any feature logic (scrolling, clicking, gestures, etc.)
  - Any system-level automation or keyboard/mouse simulation.
  - The `features/` package or `main.py` orchestration.
"""

import os
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as mp_base_options

# ---------------------------------------------------------------------------
# MediaPipe drawing helpers (still available under tasks.python.vision).
# ---------------------------------------------------------------------------

# Hand skeleton connections (21 pairs) — defined in the Tasks vision module.
_HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS

# Default model path (same directory as this file's parent package).
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),  # → hand_gesture_system/
    "hand_landmarker.task",
)


class HandTracker:
    """
    A self-contained hand-detection and landmark-extraction wrapper
    built on top of the MediaPipe Tasks HandLandmarker (v0.10+ API).

    Parameters
    ----------
    model_path : str
        Path to the `hand_landmarker.task` model asset file.
    detection_confidence : float
        Minimum confidence for the palm-detection model. Range [0.0, 1.0].
        Default 0.5 (lowered from 0.7 to reduce landmark drops on occluded
        or closed-fist transitions).
    tracking_confidence : float
        Minimum confidence for landmark tracking. Range [0.0, 1.0].
        Default 0.5 (lowered from 0.7 for the same reason).
    presence_confidence : float
        Minimum confidence that a detected object is a hand. Range [0.0, 1.0].
        Default 0.5.
    max_hands : int
        Maximum number of hands to detect simultaneously. Default 2.
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL_PATH,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        presence_confidence: float = 0.5,
        max_hands: int = 2,
        swap_handedness: bool = True,
    ) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"MediaPipe model not found at: {model_path}\n"
                "Download it from:\n"
                "  https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )

        options = vision.HandLandmarkerOptions(
            base_options=mp_base_options.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,   # Stateless per-frame mode.
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence,
        )

        # See get_landmarks(): True swaps MediaPipe's Left/Right label.  If the
        # on-screen L/R letters are backwards for your hands, set
        # "swap_handedness" to false in config.json (no code change needed).
        self.swap_handedness = swap_handedness

        self._landmarker = vision.HandLandmarker.create_from_options(options)

        # Cache the last processed result for draw_landmarks().
        self._last_result: vision.HandLandmarkerResult | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_hands(self, frame):
        """
        Process a single BGR frame through the MediaPipe HandLandmarker.

        MediaPipe Tasks requires an `mp.Image` object in RGB format, so
        colour-channel conversion happens here — callers always work in
        OpenCV's native BGR space.

        Parameters
        ----------
        frame : np.ndarray
            A BGR image as returned by cv2.VideoCapture.read().

        Returns
        -------
        result : mediapipe.tasks.python.vision.HandLandmarkerResult
            Raw MediaPipe result. Prefer `get_landmarks()` for structured data.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self._last_result = self._landmarker.detect(mp_image)
        return self._last_result

    def get_landmarks(self, frame):
        """
        Extract a fully structured dataset for every detected hand in the frame.

        Calls `find_hands()` internally — no need to call it separately.

        Parameters
        ----------
        frame : np.ndarray
            A BGR image as returned by cv2.VideoCapture.read().
            IMPORTANT: pass the frame *after* cv2.flip() has been applied so
            that pixel coordinates and handedness labels match what the user
            sees on screen.

        Returns
        -------
        list[dict] | None
            A list with one dict per detected hand, ordered as MediaPipe
            returns them (typically dominant hand first).  Each dict has:

            {
                "label"     : str,               # "Right" or "Left"
                "landmarks" : list[tuple],        # 21 entries
            }

            Each landmark tuple is:
                (id, x_px, y_px, z)
                  - id    : int   landmark index 0–20
                  - x_px  : int   pixel column  (0 ... frame width  - 1)
                  - y_px  : int   pixel row     (0 ... frame height - 1)
                  - z     : float depth relative to the wrist (negative = closer)

            Returns None if no hand is detected.

        Handedness note
        ---------------
        Because the caller flips the frame horizontally before this method
        runs, MediaPipe processes the already-mirrored image.  In that
        coordinate system "Right" == the user's physical right hand and
        "Left" == the user's physical left hand — no label inversion needed.

        Example
        -------
        all_hands = tracker.get_landmarks(frame)
        if all_hands:
            wrist = all_hands[0]["landmarks"][0]  # (0, 312, 540, 0.0)
            idx, x_px, y_px, z = wrist
        """
        result = self.find_hands(frame)

        if not result.hand_landmarks:
            return None

        # Derive pixel dimensions from the frame once.
        h, w = frame.shape[:2]

        all_hands = []
        for i, hand_landmarks in enumerate(result.hand_landmarks):

            # Handedness label —————————————————————————————————————————————
            # MediaPipe classifies from the camera's perspective on the raw
            # pixel data it receives.  Because we flip the frame horizontally
            # before detection, the classifier sees a mirror image and its
            # labels are inverted relative to the user's physical hands.
            # Correction: swap Left ↔ Right so downstream code sees the true
            # physical hand.
            raw_label = result.handedness[i][0].category_name
            if self.swap_handedness:
                label = "Right" if raw_label == "Left" else "Left"
            else:
                label = raw_label

            # Landmark list — normalised floats → pixel integers ————————————
            landmark_list = [
                (idx, int(lm.x * w), int(lm.y * h), lm.z)
                for idx, lm in enumerate(hand_landmarks)
            ]

            all_hands.append({
                "label":     label,
                "landmarks": landmark_list,
            })

        return all_hands

    def draw_landmarks(self, frame):
        """
        Draw the 21 landmark dots and their skeletal connections onto the
        provided BGR frame **in-place**, using the cached last result.

        Parameters
        ----------
        frame : np.ndarray
            The BGR frame to draw on (modified in-place).

        Returns
        -------
        frame : np.ndarray
            The same frame with annotations drawn.
        """
        if self._last_result and self._last_result.hand_landmarks:
            h, w = frame.shape[:2]
            for hand_landmarks in self._last_result.hand_landmarks:
                # Draw connections (skeleton lines).
                # Connection objects expose .start and .end landmark indices.
                for connection in _HAND_CONNECTIONS:
                    start = hand_landmarks[connection.start]
                    end   = hand_landmarks[connection.end]
                    sx, sy = int(start.x * w), int(start.y * h)
                    ex, ey = int(end.x * w),   int(end.y * h)
                    cv2.line(frame, (sx, sy), (ex, ey), (255, 255, 255), 2, cv2.LINE_AA)

                # Draw landmark dots.
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 215, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, (cx, cy), 5, (255, 255, 255),  1, cv2.LINE_AA)

        return frame

    def release(self) -> None:
        """Release the MediaPipe HandLandmarker resources gracefully."""
        self._landmarker.close()
