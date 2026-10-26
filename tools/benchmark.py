"""Per-stage latency and FPS of the real vision engine, with and without the ML classifier.

    python tools/benchmark.py --source camera          # real camera (what you should publish)
    python tools/benchmark.py --source synthetic       # no camera: a code check, see below
    python tools/benchmark.py --source camera --model hand_gesture_system/models/gesture_model.npz

Stages (from core/timing.py, recorded by the engine itself):
    capture      cv2 VideoCapture.read()
    mediapipe    HandTracker.get_landmarks (palm detector + landmark model)
    classify     pose naming for the display: rules or ML
    controllers  every gesture controller (mouse, scroll, ...) plus kill switch, with a FAKE
                 pyautogui, so real operating-system call time is NOT included
    overlay      drawing the skeleton, HUD and zone guides
    frame        the whole per-frame processing after capture

Reported as mean and p95 in ms.  FPS is the measured loop rate; with a real camera it is
capped by the camera (usually 30), so 'headroom fps' = 1000 / mean(frame) is also given.

--source synthetic has no camera or hand: MediaPipe runs on an EMPTY frame (palm detector
only, a lower bound) and its result is replaced by two synthetic hands so the classifier,
controllers and overlay do real work.  Without --model it times a randomly initialised
network of the deployed shape (cost does not depend on the weights).  Those numbers are
labelled as such and written to out/bench_demo, never docs/media.
"""
import argparse
import json
import os
import platform
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _shown(path):
    """Path relative to the repo for display; the absolute path when it is on another drive (Windows)."""
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return path
sys.path.insert(0, os.path.join(ROOT, "hand_gesture_system"))

import numpy as np                                      # noqa: E402

import config_manager                                   # noqa: E402
import main as app_main                                 # noqa: E402
from core import os_input                               # noqa: E402
from core.hand_tracker import HandTracker               # noqa: E402
from ml import features as F                            # noqa: E402
from ml.classifier import DEFAULT_MODEL_PATH, MLClassifier, RuleClassifier, make_classifier   # noqa: E402
from ml.model import NumpyModel                         # noqa: E402
from ml.labels import CLASSES                           # noqa: E402

STAGES = ("capture", "mediapipe", "classify", "controllers", "overlay", "frame")


class FakePG:
    """Swallows every OS call: the benchmark must never move your real mouse."""
    def size(self): return (1920, 1080)
    def __getattr__(self, name): return lambda *a, **k: None


class FakeCap:
    def __init__(self, frame): self._f = frame
    def read(self): return True, self._f.copy()
    def isOpened(self): return True
    def release(self): pass


def _blank_frame(fw=640, fh=480):
    return (np.random.default_rng(1).random((fh, fw, 3)) * 40 + 100).astype(np.uint8)


class SyntheticHands:
    """Real MediaPipe call (timed), then return fixed synthetic hands."""
    def __init__(self, real):
        from core.poses import pose_points
        self.real, self.swap_handedness = real, True
        def hand(label, pose, ox, **kw):
            pts = pose_points(pose, **kw)
            sg = -1 if label == "Left" else 1
            return {"label": label, "landmarks": [(i, int(ox + sg * pts[i][0] * 90), int(330 + pts[i][1] * 90), 0.0)
                                                  for i in range(21)]}
        self.hands = [hand("Left", "three", 170), hand("Right", "point", 450, pinch_to=8, pinch_ratio=0.1)]
    def get_landmarks(self, frame):
        self.real.get_landmarks(frame)
        return self.hands
    def draw_landmarks(self, frame): return frame
    def release(self): self.real.release()


def random_model() -> NumpyModel:
    rng = np.random.default_rng(0)
    return NumpyModel("mlp", np.zeros(F.N_FEATURES), np.ones(F.N_FEATURES),
                      [rng.normal(0, .3, (F.N_FEATURES, 32)), rng.normal(0, .3, (32, len(CLASSES)))],
                      [np.zeros(32), np.zeros(len(CLASSES))], CLASSES, {"rotate": True})


def run_one(name, source, camera, frames, warmup, model_path):
    cfg = config_manager.load_config()
    cfg = dict(cfg, classifier=name)
    config_manager.load_config = lambda: cfg                       # engine re-reads config; keep ours
    os_input.set_backend(FakePG())
    os_input.ARMING.set_armed(True)                                # controllers run their full path
    app_main.WEBCAM_INDEX = camera
    if source == "synthetic":
        # never touch a real capture device (probing a missing camera can stall for a long time)
        app_main.cv2.VideoCapture = lambda *a, **k: FakeCap(_blank_frame())
    model = os.path.join(ROOT, "hand_gesture_system", "hand_landmarker.task")
    tracker = HandTracker(model_path=model, max_hands=2)
    if source == "synthetic":
        tracker = SyntheticHands(tracker)
        fw, fh = 640, 480
    else:
        import cv2
        c = cv2.VideoCapture(camera)
        ok, f = c.read(); c.release()
        if not ok:
            sys.exit(f"Could not read from camera {camera}")
        fh, fw = f.shape[:2]
    eng = app_main.VisionEngine(tracker, app_main.build_controllers(cfg, fw, fh), fw, fh)
    if name == "ml":
        if source == "synthetic" and not model_path:
            eng._classifier = MLClassifier(random_model())
        else:
            eng._classifier, note = make_classifier("ml", model_path or DEFAULT_MODEL_PATH)
            if note:
                sys.exit(note + "\nTrain one first (tools/train_classifier.py) or pass --model.")
    else:
        eng._classifier = RuleClassifier()
    eng._cls_requested = name
    eng.start()
    t_start = None
    deadline = time.time() + 300
    while eng._frame_count < warmup + frames and time.time() < deadline and eng.is_alive():
        if t_start is None and eng._frame_count >= warmup:
            t_start, n0 = time.perf_counter(), eng._frame_count
        time.sleep(0.01)
    t_end, n1 = time.perf_counter(), eng._frame_count
    eng.stop(); eng.join(timeout=5)
    try:
        tracker.release()
    except Exception:
        pass
    os_input.ARMING.set_armed(False); os_input.set_backend(None)
    if t_start is None or n1 <= n0:
        sys.exit("engine produced too few frames")
    summ = eng.timings.summary(skip=warmup)
    fps = (n1 - n0) / (t_end - t_start)
    return {"stages": summ, "fps": fps, "frames": n1 - n0,
            "headroom_fps": 1000.0 / summ["frame"]["mean_ms"]}


def table(res) -> str:
    lines = ["| Stage | rules mean | rules p95 | ml mean | ml p95 |", "|---|---|---|---|---|"]
    for st in STAGES:
        r, m = res["rules"]["stages"].get(st), res["ml"]["stages"].get(st)
        cell = lambda x, k: f"{x[k]:.2f} ms" if x else "-"
        lines.append(f"| {st} | {cell(r, 'mean_ms')} | {cell(r, 'p95_ms')} | {cell(m, 'mean_ms')} | {cell(m, 'p95_ms')} |")
    lines.append(f"| loop FPS | {res['rules']['fps']:.1f} | | {res['ml']['fps']:.1f} | |")
    lines.append(f"| headroom FPS (1000 / mean frame) | {res['rules']['headroom_fps']:.0f} | | {res['ml']['headroom_fps']:.0f} | |")
    return "\n".join(lines)


def plot(res, path, note):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.2))
    stages = [s for s in STAGES if s != "frame" and s in res["rules"]["stages"]]
    h = 0.36
    for k, (name, col) in enumerate((("rules", "#6B727B"), ("ml", "#E15759"))):
        y = np.arange(len(stages)) + (k - 0.5) * h
        mean = [res[name]["stages"][s]["mean_ms"] for s in stages]
        p95 = [res[name]["stages"][s]["p95_ms"] for s in stages]
        ax.barh(y, mean, h, color=col, label=f"classifier = {name}")
        ax.scatter(p95, y, marker="|", color="black", s=120, zorder=3, label="p95" if k == 0 else None)
    ax.set_yticks(range(len(stages)), stages); ax.invert_yaxis()
    ax.set_xlabel("ms per frame (bar = mean, tick = p95)"); ax.legend(fontsize=8)
    ax.set_title("Per-stage latency", fontsize=11)
    fig.text(0.5, -0.04, note, ha="center", fontsize=8, wrap=True)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["camera", "synthetic"], required=True)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--model", default=None, help="trained .npz for the ml run")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    synthetic = a.source == "synthetic"
    out = a.out or os.path.join(ROOT, "out", "bench_demo" if synthetic else os.path.join("docs", "media"))
    os.makedirs(out, exist_ok=True)

    res = {name: run_one(name, a.source, a.camera, a.frames, a.warmup, a.model) for name in ("rules", "ml")}
    env = {"platform": platform.platform(), "processor": platform.processor() or platform.machine(),
           "python": platform.python_version(), "cpu_count": os.cpu_count(), "source": a.source,
           "frames": a.frames, "warmup": a.warmup}
    if synthetic:
        env["note"] = ("SYNTHETIC INPUT: no camera; MediaPipe ran on an empty frame (lower bound) and was "
                       "replaced by synthetic hands; ml weights random unless --model; fake pyautogui")
    else:
        env["note"] = "real camera; controllers ran against a fake pyautogui (OS call time not included)"
    with open(os.path.join(out, "latency.json"), "w") as f:
        json.dump({"env": env, "results": res}, f, indent=2)
    md = f"Measured on: {env['platform']}, {env['processor']}, Python {env['python']}, {env['cpu_count']} cores.\n" \
         f"{env['note']}\n\n" + table(res) + "\n"
    with open(os.path.join(out, "latency.md"), "w") as f:
        f.write(md)
    plot(res, os.path.join(out, "latency.png"), env["note"])
    print(md)
    print("wrote", _shown(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
