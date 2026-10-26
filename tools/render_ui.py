"""Render every tab of the UI to PNG files using a fake camera engine.

    python tools/render_ui.py <out_dir>

Useful for before/after comparisons when refactoring the UI.  On a headless
Linux box run it as:  xvfb-run -a -s "-screen 0 1280x800x24" python tools/render_ui.py out
For README screenshots use the real app instead (real camera frame, armed state).
"""
import os, sys, time
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hand_gesture_system"))
from PIL import ImageGrab
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_overlay import backdrop, hand            # noqa: E402
from core import overlay, os_input                   # noqa: E402
from core.recognizer import describe_hands           # noqa: E402
from core.status import STATUS, TIMELINE             # noqa: E402
import numpy as np                                   # noqa: E402
import cv2                                           # noqa: E402
from ui.app_window import GestureApp                 # noqa: E402


def sample_frame(armed, camera=False):
    # Live tab = skeleton on the dark backdrop (design C); Diagnostics = over the camera picture
    if camera:
        f = backdrop(640, 360)
    else:
        f = np.empty((360, 640, 3), np.uint8); f[:] = (19, 16, 14)
    hands = [hand("Left", "three", (170, 330), px=80), hand("Right", "point", (450, 330), px=80, pinch=0.1)]
    overlay.draw(f, overlay.OverlayContext(armed=armed, hands=hands, gestures=describe_hands(hands),
                                           fps=29.6, mouse_rect=100, scroll_lines=(40, 320),
                                           hold_label="Launch: hold" if armed else "",
                                           hold_progress=0.6))
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB), describe_hands(hands)

class Eng:
    shutdown_requested = False
    running = True
    latest_frame = sample_frame(False)[0]
    latest_camera_frame = sample_frame(False, camera=True)[0]
    want_camera_view = False
    def stop(self): pass
    def join(self, timeout=None): pass

app = GestureApp(vision_engine=Eng())
app.geometry("1200x740+0+0")
def pump(sec):
    end = time.time() + sec
    while time.time() < end:
        app.update_idletasks(); app.update(); time.sleep(0.01)
os_input.ARMING.set_armed(False)
pump(1.0)
names = {"Live": "live", "Gestures": "gestures", "Settings": "settings", "Whiteboard": "whiteboard", "Diagnostics": "diagnostics"}
for tab, fn in names.items():
    app._navigate(tab); pump(1.2)
    # feed deterministic log lines
    if tab == "Diagnostics":
        app.append_log("hello from test"); pump(0.5)
    ImageGrab.grab(bbox=(0, 0, 1200, 740), xdisplay=os.environ["DISPLAY"]).save(f"{out}/{fn}.png")
    print("shot", fn)
# armed state, with hands and events
app.vision_engine.latest_frame, gestures = sample_frame(True)
app._arm_var.set(True); app._on_arm_switch()
STATUS.update(camera="connected", fps=29.6, frame_size=(640, 360), hands=2, gestures=gestures,
              hold_label="Launch: hold", hold_progress=0.6)
TIMELINE.add("gesture", "L · Three fingers"); TIMELINE.add("gesture", "R · Point")
TIMELINE.add("action", "Click"); TIMELINE.add_throttled("action", "Scroll down")
TIMELINE.add("system", "Frame size is now 640x360; zones rescaled.")
app._navigate("Live"); pump(0.8)
ImageGrab.grab(bbox=(0, 0, 1200, 740), xdisplay=os.environ["DISPLAY"]).save(f"{out}/live_armed.png")
os_input.ARMING.set_armed(False)
app.destroy()
print("done")
