"""Record labelled hand landmarks from the webcam into a CSV (for tools/train_classifier.py).

    python tools/record_gestures.py --session desk_evening_lamp
    python tools/record_gestures.py --session window_morning --out data/window_morning.csv

Keys (the preview window must have focus)
    1-9      choose the gesture to record   (see the list drawn on the window)
    SPACE    start / stop recording the chosen gesture (a short countdown first)
    q / ESC  save and quit

Only frames with exactly ONE hand are saved.  Name each sitting with --session:
the evaluation splits by session, so use a new name for every sitting (another
day, room, lighting, or after moving the camera).  Recording guide: docs/ML.md.

The CSV contains your hand landmarks (coordinates only, no images).  It is
personal data: data/ is git-ignored, keep it out of the repository.
"""
import argparse
import datetime
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "hand_gesture_system"))

import cv2                                               # noqa: E402

import config_manager                                    # noqa: E402
from core import tokens as T                             # noqa: E402
from core.hand_tracker import HandTracker                # noqa: E402
from ml.dataset import CsvWriter                         # noqa: E402
from ml.labels import CLASSES                            # noqa: E402
from ml.recording import Recorder                        # noqa: E402

HINTS = {
    "open_palm": "palm to camera, fingers spread", "fist": "closed fist",
    "point": "index finger up, others curled", "two_fingers": "index + middle up",
    "three_fingers": "index, middle, ring up", "four_fingers": "four fingers up, thumb tucked",
    "thumbs_up": "thumb straight up, fingers curled", "pinch": "thumb touching index tip",
    "none": "relaxed or in-between hand, anything that is NOT one of the others",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default=None, help="name of this sitting (default: timestamp)")
    ap.add_argument("--out", default=None, help="CSV path (default data/<session>.csv)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--countdown", type=float, default=3.0, help="seconds before recording starts")
    a = ap.parse_args()

    session = a.session or datetime.datetime.now().strftime("session_%Y%m%d_%H%M")
    if session.startswith("synthetic_"):
        sys.exit("session names starting with 'synthetic_' are reserved for generated data")
    path = a.out or os.path.join(ROOT, "data", f"{session}.csv")
    model = os.path.join(ROOT, "hand_gesture_system", "hand_landmarker.task")
    if not os.path.exists(model):
        sys.exit("Model file missing. Run: python tools/download_model.py")

    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {a.camera}")
    tracker = HandTracker(model_path=model, max_hands=2)     # 2 so we can SEE and skip two-hand frames
    tracker.swap_handedness = bool(config_manager.load_config().get("swap_handedness", True))
    rec = Recorder(CsvWriter(path), session)
    print(f"Recording session {session!r} -> {path}")
    t0, start_at = time.time(), None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera read failed"); break
            frame = cv2.flip(frame, 1)                      # the same mirroring as the app
            hands = tracker.get_landmarks(frame)
            h, w = frame.shape[:2]
            now = time.time()
            if start_at is not None and now >= start_at and not rec.recording:
                rec.toggle()
            status = rec.feed(hands, now - t0, (w, h))
            tracker.draw_landmarks(frame)
            _draw(frame, rec, status, start_at, now)
            cv2.imshow("record_gestures", frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if ord("1") <= k <= ord("9") and not rec.recording:
                rec.set_label(CLASSES[k - ord("1")])
            if k == 32:
                if rec.recording:
                    rec.toggle(); start_at = None
                else:
                    start_at = now + a.countdown
    finally:
        rec.writer.close(); cap.release(); tracker.release(); cv2.destroyAllWindows()
    print("saved per class:", dict(rec.saved), "| skipped:", dict(rec.skipped))
    print(f"wrote {path}")
    return 0


def _draw(frame, rec, status, start_at, now) -> None:
    def put(text, y, color=T.hex_to_bgr(T.TEXT), s=0.55):
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, s, color, 1, cv2.LINE_AA)

    state = "RECORDING" if rec.recording else ("starting in %.0f" % (start_at - now) if start_at else "paused")
    put(f"{rec.label}: {HINTS[rec.label]}", 24)
    put(f"{state}   [{status}]", 48, T.hex_to_bgr(T.ERROR if rec.recording else T.MUTED))
    for i, c in enumerate(CLASSES):
        mark = ">" if c == rec.label else " "
        put(f"{mark}{i + 1} {c:14s} {rec.saved[c]:5d}", 80 + i * 20, s=0.45)


if __name__ == "__main__":
    sys.exit(main())
