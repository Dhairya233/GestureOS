"""Record 1080p footage of the real app driven by scripted landmarks (used by the demo video).

    xvfb-run -a -s "-screen 0 1920x1080x24" python tools/capture_footage.py TIMELINE.json OUT.mkv

TIMELINE.json: {"scenes":[{"i":0,"start":0,"dur":11.3},...],"total":104}
Like tools/make_demo.py there is NO webcam: generated hand landmarks replace MediaPipe and
OS input is faked. Everything else (engine, controllers, overlay, UI) is the real code.
"""
import json
import math
import os
import subprocess
import sys
import time
import tkinter as tk
import webbrowser

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "hand_gesture_system"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import benchmark as B                                  # noqa: E402
import main as app_main                                # noqa: E402
import config_manager                                  # noqa: E402
from core import os_input                              # noqa: E402
from core.poses import THUMB_OUT, pose_points          # noqa: E402

TL = json.load(open(sys.argv[1]))
OUTFILE = sys.argv[2]
SC = {s["i"]: s for s in TL["scenes"]}
webbrowser.open = lambda *a, **k: True


def hand(label, pose, wx, wy, px=100, pinch_to=None, ratio=0.1, thumb_out=False):
    pts = pose_points(pose, pinch_to=pinch_to, pinch_ratio=ratio)
    if thumb_out:
        for lm, xy in zip((1, 2, 3, 4), THUMB_OUT):
            if not (pinch_to is not None and lm in (3, 4)):
                pts[lm] = xy
    sg = -1 if label == "Left" else 1
    return {"label": label, "landmarks": [(i, int(wx + sg * pts[i][0] * px), int(wy + pts[i][1] * px), 0.0)
                                          for i in range(21)]}


def sweep(d, cx=320, amp=120, sp=1.3):
    return cx + amp * math.sin(d * sp)


def P(label, pose, x, y, **k):
    return hand(label, pose, x, y, **k)


# scene scripts: fn(d, u) -> hands; d = seconds into the scene, u = 0..1
def s0(d, u): return [] if d < 0.4 else [P("Right", "point", sweep(d, amp=90, sp=1.0), 430, thumb_out=True)]
def s1(d, u): return [P("Left", "four", 170 + 12 * math.sin(d), 440 + 8 * math.sin(d * 1.3)),
                      P("Right", "open", 430 + 15 * math.sin(d * 0.9), 430 + 10 * math.sin(d * 1.1))]
def s2(d, u):
    if d < 2.4:
        return [P("Right", "point", sweep(d, amp=60), 430, thumb_out=True)]
    return [P("Left", "open", 170, 440), P("Right", "open", 430, 440)]
def s3(d, u):
    if u < 0.5:
        return [P("Right", "point", sweep(d, amp=130, sp=1.7), 410 + 25 * math.sin(d * 2.1), thumb_out=True)]
    pin = (0.55 < u < 0.68) or (0.78 < u < 0.9)
    return [P("Right", "point", 330, 410, pinch_to=12 if pin else None, ratio=0.05, thumb_out=True)]
def s4(d, u): return [P("Right", "open", 330, 450 if u < 0.5 else 250)]
def s5(d, u):
    if 1.5 < d < 1.9:
        return [P("Left", "peace", 170, 440)]
    if 4.6 < d < 8.6:
        return [P("Left", "three", 170, 440)]
    return []
def s6(d, u):
    dur = SC[6]["dur"]
    if 0.6 < d < 5.0:
        a = (d - 0.6) * 2.6
        return [P("Right", "point", 330 + 100 * math.sin(a) * math.cos(a * 0.5), 390 - 50 * math.cos(a), thumb_out=True)]
    if 5.8 < d < dur - 0.4:
        return [P("Right", "open", 565, 340)]
    return []
def s7(d, u):
    if 1.2 < d < 4.5:
        return [P("Left", "open", 170, 440), P("Right", "open", 430, 440)]
    return [P("Right", "point", sweep(d, amp=60), 430, thumb_out=True)] if d < 1.2 else []
def s8(d, u): return [P("Right", "point", sweep(d, amp=100, sp=1.2), 430, thumb_out=True)]
def s9(d, u): return [P("Right", "point", sweep(d, amp=90, sp=0.9), 430, thumb_out=True)]
FN = {0: s0, 1: s1, 2: s2, 3: s3, 4: s4, 5: s5, 6: s6, 7: s7, 8: s8, 9: s9}
CANVAS_ON = (lambda t: SC[6]["start"] + 0.5 < t < SC[6]["start"] + 5.3)
TABS = [(SC[8]["start"], "Gestures"), (SC[8]["start"] + 4.0, "Settings"), (SC[8]["start"] + 8.0, "Diagnostics"),
        (SC[9]["start"], "Live")]

os_input.set_backend(B.FakePG())


class PacedCap(B.FakeCap):
    def read(self):
        time.sleep(1 / 30)
        return super().read()


app_main.cv2.VideoCapture = lambda *a, **k: PacedCap(B._blank_frame())
cfg = dict(config_manager.load_config(), classifier="rules", enable_canvas=False)
config_manager.load_config = lambda: cfg


class Scripted:
    swap_handedness = True
    t0 = 1e18
    def get_landmarks(self, frame):
        t = time.time() - self.t0
        for s in TL["scenes"]:
            if s["start"] <= t < s["start"] + s["dur"]:
                return FN[s["i"]](t - s["start"], (t - s["start"]) / s["dur"])
        return []
    def draw_landmarks(self, frame): return frame
    def release(self): pass


tracker = Scripted()
eng = app_main.VisionEngine(tracker, app_main.build_controllers(cfg, 640, 480), 640, 480)
from ui.app_window import GestureApp                   # noqa: E402
app = GestureApp(vision_engine=eng)
eng.set_app(app)
app.geometry("1920x1080+0+0")
os_input.ARMING.set_armed(False)


def pump(sec):
    end = time.time() + sec
    while time.time() < end:
        app.update_idletasks(); app.update(); time.sleep(0.003)


def find_switch(widget, label):
    from ui.widgets import SwitchRow
    for child in widget.winfo_children():
        if isinstance(child, SwitchRow):
            for c in child.winfo_children():
                if c.winfo_class() == "Label" and label in str(c.cget("text")):
                    return child
        r = find_switch(child, label)
        if r:
            return r
    return None


def set_canvas(on):
    row = find_switch(app, "Air canvas")
    if row is not None and bool(row.var.get()) != on:
        row.switch.toggle()


pump(1.5)
eng.start(); pump(1.5)
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "x11grab", "-draw_mouse", "0", "-framerate", "30",
                       "-video_size", "1920x1080", "-i", os.environ["DISPLAY"], "-c:v", "libx264",
                       "-preset", "ultrafast", "-crf", "12", "-pix_fmt", "yuv420p", OUTFILE],
                      stdin=subprocess.PIPE)
pump(1.0)
marker = tk.Toplevel(app); marker.overrideredirect(True); marker.geometry("24x24+0+0")
marker.configure(bg="#ff0000"); marker.attributes("-topmost", True); pump(0.6)
marker.destroy(); app.update()
tracker.t0 = time.time()
tabs = list(TABS); canvas_state = False
while True:
    t = time.time() - tracker.t0
    if t >= TL["total"] - 1.0:
        break
    while tabs and t >= tabs[0][0]:
        app._navigate(tabs.pop(0)[1])
    want = CANVAS_ON(t)
    if want != canvas_state:
        set_canvas(want); canvas_state = want
    app.update_idletasks(); app.update(); time.sleep(0.003)
ff.communicate(b"q")
eng.stop(); eng.join(3); app.destroy()
print("captured", os.path.getsize(OUTFILE) // 1024, "KB")
