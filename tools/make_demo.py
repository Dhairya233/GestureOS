"""Render the README screenshots and demo.mp4 from the REAL application.

    python tools/make_demo.py docs/media          (Linux: xvfb-run -a ...; needs ffmpeg)

What is real: the VisionEngine thread, every controller, the safety/arming logic,
the overlay, the whole CustomTkinter UI and its event timeline.
What is NOT real: there is no webcam. A script feeds generated hand landmarks to
the engine in place of MediaPipe, and OS input goes to a fake pyautogui so nothing
moves your cursor. The video says so on every frame. For a webcam recording, use
screen capture on Windows (see docs/MEDIA.md).
"""
import math
import os
import subprocess
import sys
import tempfile
import time
import webbrowser

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "hand_gesture_system"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from PIL import Image, ImageDraw, ImageFont, ImageGrab  # noqa: E402

import benchmark as B                                 # noqa: E402
import main as app_main                               # noqa: E402
import config_manager                                 # noqa: E402
from core import os_input                             # noqa: E402
from core.poses import THUMB_OUT, pose_points                    # noqa: E402
from core.status import TIMELINE                      # noqa: E402

OUT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "media"))
os.makedirs(OUT, exist_ok=True)
FPS, W, H, BAR = 15, 1200, 740, 64
NOTE = "Scripted landmarks, not a webcam recording"
webbrowser.open = lambda *a, **k: True               # the launcher must not open a real browser


def hand(label, pose, wx, wy, px=90, pinch_to=None, ratio=0.1, thumb_out=False):
    pts = pose_points(pose, pinch_to=pinch_to, pinch_ratio=ratio)
    if thumb_out:                      # thumb held away from the other fingers
        for lm, xy in zip((1, 2, 3, 4), THUMB_OUT):
            if not (pinch_to is not None and lm in (3, 4)):
                pts[lm] = xy
    sg = -1 if label == "Left" else 1
    return {"label": label, "landmarks": [(i, int(wx + sg * pts[i][0] * px), int(wy + pts[i][1] * px), 0.0)
                                          for i in range(21)]}


def sweep(t, cx=320, amp=110, speed=1.6):
    return cx + amp * math.sin(t * speed)


# (seconds, caption, hands(dt) -> list, canvas on?)   dt = seconds since the scene began
_RAW = [
    (3.5, "1  Starts disarmed: it sees your hands, but sends nothing to the PC",
     lambda d: [hand("Right", "point", sweep(d), 420, thumb_out=True)], False),
    (3.0, "2  Show both open palms for about a second to arm",
     lambda d: [hand("Left", "open", 170, 430), hand("Right", "open", 470, 430)], False),
    (4.5, "3  Air mouse: the index finger moves the cursor",
     lambda d: [hand("Right", "point", sweep(d), 400 + 25 * math.sin(d * 2.3), thumb_out=True)], False),
    (2.2, "4  Thumb-to-middle pinch clicks",
     lambda d: [hand("Right", "point", 330, 400, pinch_to=12, ratio=0.05, thumb_out=True)], False),
    (2.5, "5  Open palm low in the frame scrolls down",
     lambda d: [hand("Right", "open", 330, 445)], False),
    (2.5, "6  ...and high in the frame scrolls up",
     lambda d: [hand("Right", "open", 330, 255)], False),
    (2.5, "7  Left hand peace sign: play / pause",
     lambda d: [hand("Left", "peace", 170, 430)] if d < 0.5 else [], False),
    (3.5, "8  Hold three fingers up: launch your app (held, so it never fires by accident)",
     lambda d: [hand("Left", "three", 170, 430)], False),
    (4.0, "9  Air canvas: draw with your index finger",
     lambda d: [hand("Right", "point", 330 + 90 * math.cos(d * 2.4), 380 + 45 * math.sin(d * 2.4), thumb_out=True)], True),
    (2.5, "10  Open palm at the edge of the frame: switch window",
     lambda d: [hand("Right", "open", 560, 340)], False),
    (3.3, "11  Both palms again to disarm. Ctrl+Alt+F12 also disarms from any window",
     lambda d: [hand("Left", "open", 170, 430), hand("Right", "open", 470, 430)], False),
    (3.0, "Every gesture is listed in the app, generated from the controller code",
     lambda d: [], False),
]
SCENES, _t = [], 0.0
for _dur, _cap, _fn, _canvas in _RAW:
    SCENES.append((_t, _t + _dur, _cap, _fn, _canvas)); _t += _dur
TOTAL = _t


def start_of(prefix):
    return next(s[0] for s in SCENES if s[2].startswith(prefix))


SHOTS = {  # seconds -> window screenshot for the README
    start_of("1 ") + 1.8: "live_disarmed", start_of("2 ") + 2.3: "live_arming",
    start_of("3 ") + 2.0: "live_mouse", start_of("4 ") + 1.5: "live_click",
    start_of("5 ") + 1.5: "live_scroll", start_of("8 ") + 1.9: "live_launcher",
    start_of("9 ") + 3.6: "live_canvas",
}

os_input.set_backend(B.FakePG())
app_main.cv2.VideoCapture = lambda *a, **k: PacedCap(B._blank_frame())
cfg = config_manager.load_config()
cfg = dict(cfg, classifier="rules", enable_canvas=False)
config_manager.load_config = lambda: cfg


class PacedCap(B.FakeCap):
    """30 fps like a real webcam, so hold times (frames) match real use."""
    def read(self):
        time.sleep(1 / 30)
        return super().read()


class Scripted:
    swap_handedness = True
    t0 = 0.0
    def get_landmarks(self, frame):
        t = time.time() - self.t0
        for a, b, _c, fn, _canvas in SCENES:
            if a <= t < b:
                return fn(t - a)
        return []
    def draw_landmarks(self, frame): return frame
    def release(self): pass


tracker = Scripted()
eng = app_main.VisionEngine(tracker, app_main.build_controllers(cfg, 640, 480), 640, 480)
from ui.app_window import GestureApp                  # noqa: E402
app = GestureApp(vision_engine=eng)
eng.set_app(app)
app.geometry(f"{W}x{H}+0+0")
os_input.ARMING.set_armed(False)


def pump(sec):
    end = time.time() + sec
    while time.time() < end:
        app.update_idletasks(); app.update(); time.sleep(0.004)


def find_switch(widget, label):
    """The real 'Air canvas' style switch on the Live tab, found by its label."""
    from ui.widgets import SwitchRow
    for child in widget.winfo_children():
        if isinstance(child, SwitchRow) and any(
                getattr(c, "cget", None) and c.winfo_class() == "Label" and label in str(c.cget("text"))
                for c in _descend(child)):
            return child
        found = find_switch(child, label)
        if found:
            return found
    return None


def _descend(w):
    for c in w.winfo_children():
        yield c
        yield from _descend(c)
        yield from ()


def set_canvas(on):
    row = find_switch(app, "Air canvas")
    if row is not None and bool(row.var.get()) != on:
        row.switch.toggle()      # same path as a click: updates the config the engine reads


def font(size, bold=False):
    for p in ("/usr/share/fonts/opentype/inter/Inter-%s.otf" % ("Bold" if bold else "Regular"),
              "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else "")):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


BG, TEXT, MUTED, GREEN = (14, 16, 19), (230, 232, 235), (154, 161, 170), (61, 220, 151)


def card(lines, name):
    im = Image.new("RGB", (W, H + BAR), BG); d = ImageDraw.Draw(im)
    y = 250
    for text, size, bold, col in lines:
        f = font(size, bold); w = d.textlength(text, font=f)
        d.text(((W - w) / 2, y), text, font=f, fill=col); y += size + 22
    return im


def framed(shot, caption):
    im = Image.new("RGB", (W, H + BAR), BG); d = ImageDraw.Draw(im)
    d.text((24, 17), caption, font=font(21, True), fill=TEXT)
    f = font(13); w = d.textlength(NOTE, font=f)
    d.text((W - w - 24, 6), NOTE, font=f, fill=MUTED)
    im.paste(shot, (0, BAR))
    return im


frames_dir = tempfile.mkdtemp(prefix="gos_demo_")
n = 0
def save(im, repeat=1):
    global n
    for _ in range(repeat):
        im.save(os.path.join(frames_dir, f"f{n:05d}.png")); n += 1

save(card([("GestureOS", 64, True, TEXT), ("Control a Windows PC with your hands", 28, False, MUTED),
           ("", 20, False, MUTED), (NOTE + ". Real engine, overlay and UI;", 18, False, GREEN),
           ("generated hands, fake OS input.", 18, False, GREEN)], "title"), FPS * 3)

pump(1.0)
eng.start()
pump(0.8)
tracker.t0 = time.time()
shots = dict(SHOTS)
k = 0
tab = "Live"
while True:
    t = k / FPS
    if t >= TOTAL:
        break
    while time.time() - tracker.t0 < t:
        pump(0.003)
    cur = next((s for s in SCENES if s[0] <= t < s[1]), SCENES[-1])
    set_canvas(cur[4])
    if cur is SCENES[-1] and tab != "Gestures":
        app._navigate("Gestures"); tab = "Gestures"; pump(0.3)
    app.update_idletasks(); app.update()
    grab = ImageGrab.grab(bbox=(0, 0, W, H), xdisplay=os.environ["DISPLAY"]).convert("RGB")
    save(framed(grab, cur[2]))
    for st in [s for s in shots if s <= t]:
        grab.save(os.path.join(OUT, shots.pop(st) + ".png"))
    k += 1

# extra still captures: other tabs
os_input.ARMING.set_armed(False)
for name, fn in (("Settings", "settings"), ("Diagnostics", "diagnostics"), ("Whiteboard", "whiteboard")):
    app._navigate(name); pump(1.2)
    ImageGrab.grab(bbox=(0, 0, W, H), xdisplay=os.environ["DISPLAY"]).save(os.path.join(OUT, fn + ".png"))
app._navigate("Gestures"); pump(0.8)
ImageGrab.grab(bbox=(0, 0, W, H), xdisplay=os.environ["DISPLAY"]).save(os.path.join(OUT, "gestures.png"))
events = [e.text for e in TIMELINE.since(0)]
eng.stop(); eng.join(3); app.destroy()

save(card([("Safe by design", 48, True, TEXT), ("Starts disarmed. Armed only by you. Ctrl+Alt+F12 disarms from any window.", 22, False, MUTED),
           ("", 20, False, MUTED), ("github.com/Dhairya233/GestureOS", 24, True, GREEN)], "end"), FPS * 4)

mp4 = os.path.join(OUT, "scripted_demo.mp4")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i",
                os.path.join(frames_dir, "f%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "23", "-movflags", "+faststart", mp4], check=True)
# hero GIF: the first 14 s from the middle scenes, small and light
gif = os.path.join(OUT, "scripted_hero.gif")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "9", "-t", "16", "-i", mp4,
                "-vf", "fps=10,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse",
                gif], check=True)
print("events seen:", len(events))
for e in events[:60]:
    print("  ", e)
print("wrote", mp4, os.path.getsize(mp4) // 1024, "KB;", gif, os.path.getsize(gif) // 1024, "KB")
