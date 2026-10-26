import json, math, subprocess, sys
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 30
TL = json.load(open("timeline.json")); SC = {s["i"]: s for s in TL["scenes"]}; TOTAL = TL["total"]
OFF = 47 / 30            # footage frame where scene time 0 begins (marker sync)
ACCENT = (151, 220, 61)  # BGR of #3DDC97
WHITE = (235, 232, 230); MUTED = (170, 161, 154); BGC = (19, 16, 14)

def font(sz, bold=False):
    return ImageFont.truetype("/usr/share/fonts/opentype/inter/Inter-%s.otf" % ("Bold" if bold else "Regular"), sz)
def smooth(x): x = min(max(x, 0.0), 1.0); return x * x * x * (x * (6 * x - 15) + 10)
def lerp(a, b, k): return a + (b - a) * k
def S(i, rel=0.0): return SC[i]["start"] + rel

# ── text patches (RGBA numpy, premultiplied-free) ──────────────────────────────────
_cache = {}
def text_patch(text, size, bold=False, color=WHITE, tracking=0, shadow=True):
    key = (text, size, bold, color, tracking, shadow)
    if key in _cache: return _cache[key]
    f = font(size, bold)
    w = int(sum(f.getlength(c) + tracking for c in text)) + 24; h = int(size * 1.5)
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    x = 12
    for c in text:
        if shadow: d.text((x + 2, 8), c, font=f, fill=(0, 0, 0, 140))
        d.text((x, 6), c, font=f, fill=color[::-1] + (255,)); x += f.getlength(c) + tracking
    a = np.array(im); a = np.concatenate([a[..., 2::-1], a[..., 3:]], 2)   # RGBA->BGRA
    _cache[key] = a; return a

def blit(dst, patch, x, y, alpha=1.0):
    x, y = int(x), int(y); ph, pw = patch.shape[:2]
    x0, y0 = max(x, 0), max(y, 0); x1, y1 = min(x + pw, W), min(y + ph, H)
    if x1 <= x0 or y1 <= y0 or alpha <= 0: return
    p = patch[y0 - y:y1 - y, x0 - x:x1 - x]
    a = (p[..., 3:4].astype(np.float32) / 255.0) * alpha
    dst[y0:y1, x0:x1] = (dst[y0:y1, x0:x1] * (1 - a) + p[..., :3] * a).astype(np.uint8)

def rrect(shape, x, y, w, h, r, val=255):
    m = np.zeros(shape, np.uint8)
    r = int(max(min(r, w // 2, h // 2), 0))
    cv2.rectangle(m, (x + r, y), (x + w - r, y + h), val, -1); cv2.rectangle(m, (x, y + r), (x + w, y + h - r), val, -1)
    for cx, cy in ((x + r, y + r), (x + w - r, y + r), (x + r, y + h - r), (x + w - r, y + h - r)):
        cv2.circle(m, (cx, cy), r, val, -1, cv2.LINE_AA)
    return m

def key_patch(label, w, h=88):
    im = np.zeros((h + 20, w + 20, 4), np.uint8)
    m = rrect(im.shape[:2], 10, 6, w, h, 16)
    im[..., :3] = (60, 54, 50); im[..., 3] = m
    top = rrect(im.shape[:2], 10, 6, w, h - 8, 16)
    im[top > 0, :3] = (92, 84, 78)
    pil = Image.fromarray(im[..., [2, 1, 0, 3]]); d = ImageDraw.Draw(pil); f = font(34, True)
    tw = d.textlength(label, font=f); d.text((10 + (w - tw) / 2, 6 + (h - 8) / 2 - 24), label, font=f, fill=(235, 232, 230, 255))
    a = np.array(pil); return np.concatenate([a[..., 2::-1], a[..., 3:]], 2)

def pill_patch(text, color, size=30, filled=True):
    f = font(size, True); w = int(f.getlength(text)) + 64; h = size + 34
    im = np.zeros((h + 16, w + 16, 4), np.uint8)
    m = rrect(im.shape[:2], 8, 8, w, h, h // 2)
    im[..., :3] = (28, 24, 22); im[..., 3] = (m * 0.92).astype(np.uint8)
    edge = m.copy(); inner = rrect(im.shape[:2], 10, 10, w - 4, h - 4, h // 2 - 2)
    ring = (edge > 0) & (inner == 0); im[ring, :3] = color; im[ring, 3] = 255
    pil = Image.fromarray(im[..., [2, 1, 0, 3]]); d = ImageDraw.Draw(pil)
    d.ellipse((8 + 24, 8 + h / 2 - 8, 8 + 40, 8 + h / 2 + 8), fill=color[::-1] + (255,) if filled else None, outline=color[::-1] + (255,), width=3)
    d.text((8 + 52, 8 + (h - size * 1.2) / 2 - 2), text, font=f, fill=color[::-1] + (255,))
    a = np.array(pil); return np.concatenate([a[..., 2::-1], a[..., 3:]], 2)

# ── camera and layout ───────────────────────────────────────────────────────────────
FEED = (850, 430)
CAMS = {  # scene -> (cx, cy, zoom) in footage pixels
    0: (860, 440, 1.55), 1: (960, 540, 1.0), 2: (860, 420, 1.4), 3: (880, 450, 1.5), 4: (880, 460, 1.4),
    5: (700, 640, 1.3), 6: (900, 470, 1.4), 7: (860, 500, 1.2), 8: (960, 540, 1.0), 9: (960, 540, 1.0)}
LAY = {  # scene -> window box (x, y, w, h, radius)
    0: (0, 0, 1920, 1080, 0), 9: (90, 250, 1000, 562, 14)}
BOX = (160, 30, 1600, 900, 18)
def box_for(i): return LAY.get(i, BOX)
def eased_scene_param(t, getter):
    cur = 0
    for s in TL["scenes"]:
        if t >= s["start"] - 0.3: cur = s["i"]
    # transition into scene cur over [start-0.3, start+0.8]
    s = SC[cur]; k = smooth((t - (s["start"] - 0.3)) / 1.1)
    if cur == 0: return getter(0)
    a, b = getter(cur - 1), getter(cur)
    return tuple(lerp(x, y, k) for x, y in zip(a, b))

# ── events (filled while rendering; armed state detected from the footage pixels) ───
events = {"armed": [], "disarmed": []}
armed_prev = None
def detect_armed(F):
    r = F[8:44, 190:300].reshape(-1, 3)              # arm pill: green text/icon when armed (BGR)
    return int(((r[:, 1] > 170) & (r[:, 2] < 120) & (r[:, 0] < 200)).sum()) > 12

def captions_at(t, armed):
    """list of (chapter, title, sub, t_in) active at time t."""
    C = []
    def add(i, rel, ch, title, sub, until=None): C.append((S(i, rel), until, ch, title, sub))
    add(1, 1.2, "01  ·  SEE WHAT IT SEES", "21 landmarks per hand", "Skeleton, pose names and zones, drawn live")
    add(2, 0.5, "02  ·  SAFE BY DEFAULT", "Starts disarmed", "It watches. It sends nothing to your PC.", until=("armed",))
    add(2, 0.5, "02  ·  SAFE BY DEFAULT", "Armed", "Both open palms, held for a second", until=None)
    add(3, 0.6, "03  ·  AIR MOUSE", "Point to move", "Your index finger steers the cursor")
    add(3, S(3, 0) + 0.55 * SC[3]["dur"] - S(3, 0) - 0.3, "03  ·  AIR MOUSE", "Pinch to click", "Thumb to middle finger")
    add(4, 0.5, "04  ·  SCROLL", "Open palm low: scroll down", "Zones are drawn on screen")
    add(4, 0.5 * SC[4]["dur"], "04  ·  SCROLL", "Open palm high: scroll up", "Hold position, scroll speed follows distance")
    add(5, 1.0, "05  ·  LEFT HAND", "Peace sign: play / pause", "Media keys, with a cooldown")
    add(5, 4.2, "05  ·  LEFT HAND", "Three fingers, held: launch", "Held on purpose, so it never fires by accident")
    add(6, 0.5, "06  ·  CANVAS AND WINDOWS", "Draw in the air", "Index finger paints, colours live at the top")
    add(6, 5.6, "06  ·  CANVAS AND WINDOWS", "Palm at the edge: switch window", "Alt+Tab, hands-free")
    add(7, 0.6, "07  ·  SAFETY FIRST", "Two palms disarm it instantly", "Gestures are shown, never sent")
    add(7, 4.8, "07  ·  SAFETY FIRST", "Ctrl + Alt + F12", "Disarms from any window")
    add(8, 0.4, "08  ·  EVERYTHING VISIBLE", "A guide for every gesture", "Diagrams generated from the controller code")
    add(8, 4.4, "08  ·  EVERYTHING VISIBLE", "Settings you can tune", "Cursor speed, zones, launch targets")
    add(8, 8.4, "08  ·  EVERYTHING VISIBLE", "Diagnostics", "Camera view with overlay, CPU, RAM and logs")
    # choose the latest caption whose start <= t and which has not been superseded
    live = [c for c in C if c[0] <= t]
    if not live: return None
    # within a scene captions replace each other; scene 2 handles armed state
    cur = live[-1]
    if cur[3] == "Starts disarmed" and armed: cur = [c for c in C if c[3] == "Armed"][0]
    if cur[3] == "Armed" and not armed: cur = [c for c in C if c[3] == "Starts disarmed"][0]
    # end of scene: fade handled by scene end check
    return cur

def draw_caption(canvas, cap, t, scene_end):
    t_in, _u, ch, title, sub = cap
    # time since this caption became current (use its start, but for armed-switch use detection time)
    k = smooth((t - t_in) / 0.45)
    if title in ("Armed",) and events["armed"]: k = smooth((t - max(t_in, events["armed"][-1])) / 0.45)
    out = smooth((scene_end - t) / 0.4)
    a = min(k, out) if t < scene_end else 0
    if a <= 0: return
    yoff = (1 - a) * 18
    blit(canvas, text_patch(ch, 22, True, ACCENT, tracking=3), 150, 936 + yoff, a)
    blit(canvas, text_patch(title, 50, True, WHITE), 150, 962 + yoff, a)
    blit(canvas, text_patch(sub, 26, False, MUTED, shadow=False), 152, 1022 + yoff, a * 0.95)

def scene_at(t):
    for s in TL["scenes"]:
        if s["start"] <= t < s["start"] + s["dur"]: return s
    return TL["scenes"][-1]

# ── background ────────────────────────────────────────────────────────────────────────
yy, xx = np.mgrid[0:H // 4, 0:W // 4].astype(np.float32)
def background(F, t):
    small = cv2.resize(F, (96, 54), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.GaussianBlur(small, (0, 0), 2.0), (W, H), interpolation=cv2.INTER_CUBIC)
    bg = (bg.astype(np.float32) * 0.22)
    gx = W / 8 + (W / 4) * (0.5 + 0.5 * math.sin(t * 0.21)); gy = H / 8
    g = np.exp(-(((xx - gx) / (W / 5)) ** 2 + ((yy - gy) / (H / 4)) ** 2))
    g = cv2.resize(g, (W, H))[..., None]
    bg += g * np.array([40, 30, 18], np.float32) * 0.9      # soft teal glow (BGR)
    return bg

VIG = None
def vignette():
    global VIG
    if VIG is None:
        y, x = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt(((x - W / 2) / (W / 2)) ** 2 + ((y - H / 2) / (H / 2)) ** 2)
        VIG = np.clip(1.06 - 0.34 * d ** 2, 0.5, 1.0)[..., None].astype(np.float32)
    return VIG

def main(out_path):
    rd = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{OFF:.4f}", "-i", "footage.mkv", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                          stdout=subprocess.PIPE)
    wr = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                           "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", out_path], stdin=subprocess.PIPE)
    n = int(TOTAL * FPS); fb = W * H * 3
    last = None; armed_prev = None; armed = False
    for k in range(n):
        t = k / FPS
        buf = rd.stdout.read(fb)
        if len(buf) == fb: last = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        F = last if last is not None else np.zeros((H, W, 3), np.uint8)
        armed_now = bool(detect_armed(F))
        if armed_prev is not None and armed_now != armed_prev:
            events["armed" if armed_now else "disarmed"].append(t)
        armed_prev = armed_now; armed = armed_now
        sc = scene_at(t)
        bg = background(F, t)
        # window box (animated between layouts)
        def boxget(i): return box_for(i)
        bx, by, bw, bh, br = eased_scene_param(t, boxget)
        cx, cy, z = eased_scene_param(t, lambda i: CAMS[i])
        z *= 1 + 0.012 * (0.5 + 0.5 * math.sin(t * 0.6))                           # gentle breathing
        cw, ch = W / z, H / z
        if bw / bh != W / H: cw = ch * bw / bh if bw / bh < W / H else cw
        x0 = min(max(cx - cw / 2, 0), W - cw); y0 = min(max(cy - ch / 2, 0), H - ch)
        crop = F[int(y0):int(y0 + ch), int(x0):int(x0 + cw)]
        win = cv2.resize(crop, (int(bw), int(bh)), interpolation=cv2.INTER_CUBIC if z > 1 else cv2.INTER_AREA)
        # soft bloom on the brightest strokes (skeleton, ring markers): a grade, not new content
        lo = cv2.resize(win, (int(bw) // 4, int(bh) // 4), interpolation=cv2.INTER_AREA).astype(np.float32)
        lo = np.clip(lo - 150, 0, 255) * 1.6
        bloom = cv2.resize(cv2.GaussianBlur(lo, (0, 0), 5), (int(bw), int(bh)), interpolation=cv2.INTER_LINEAR)
        win = np.clip(win.astype(np.float32) + bloom * 0.55, 0, 255).astype(np.uint8)
        bx, by = int(bx), int(by)
        canvas = bg
        # shadow
        if br > 0 or bw < W:
            sm = np.zeros((H // 8, W // 8), np.uint8)
            sm = rrect(sm.shape, bx // 8, by // 8 + 6, int(bw) // 8, int(bh) // 8, max(br // 8, 1))
            sm = cv2.GaussianBlur(sm, (0, 0), 5); sh = cv2.resize(sm, (W, H))[..., None].astype(np.float32) / 255.0
            canvas = canvas * (1 - 0.65 * sh)
        canvas = canvas.astype(np.uint8)
        x1, y1 = min(bx + int(bw), W), min(by + int(bh), H)
        roi_w, roi_h = x1 - bx, y1 - by
        if roi_w > 0 and roi_h > 0:
            mask = rrect((int(bh), int(bw)), 0, 0, int(bw) - 1, int(bh) - 1, br)[:roi_h, :roi_w] if br > 0 else np.full((roi_h, roi_w), 255, np.uint8)
            m = (cv2.GaussianBlur(mask, (0, 0), 0.8).astype(np.float32) / 255.0)[..., None] if br > 0 else 1.0
            canvas[by:y1, bx:x1] = (canvas[by:y1, bx:x1] * (1 - m) + win[:roi_h, :roi_w] * m).astype(np.uint8)
            if br > 0:   # hairline border
                inner = rrect((int(bh), int(bw)), 2, 2, int(bw) - 5, int(bh) - 5, br - 2)[:roi_h, :roi_w]
                ring = ((mask > 0) & (inner == 0))
                canvas[by:y1, bx:x1][ring] = (84, 76, 70)
        canvas = (canvas.astype(np.float32) * vignette()).astype(np.uint8)
        # intro darkening so the title reads
        if sc["i"] == 0 or t < S(1, 0.9):
            dark = 0.52 * (1 - smooth((t - S(1, -0.3)) / 1.3))
            canvas = (canvas * (1 - dark)).astype(np.uint8)
        # intro title
        if t < S(1, 0.9):
            a = smooth((t - 1.0) / 0.9) * (1 - smooth((t - S(1, -0.6)) / 0.6))
            if a > 0:
                tp = text_patch("GestureOS", 150, True, WHITE, tracking=6)
                blit(canvas, tp, (W - tp.shape[1]) / 2, 330 + (1 - a) * 24, a)
                sp = text_patch("Control your PC with your hands", 40, False, MUTED, tracking=2)
                blit(canvas, sp, (W - sp.shape[1]) / 2, 510 + (1 - a) * 24, a)
        # state badge (from scene 2)
        if t >= S(2, 0.0) and t < S(9, 0.0) + 0.3:
            ba = smooth((t - S(2, 0.0)) / 0.5) * (1 - smooth((t - S(9, -0.2)) / 0.5))
            pp = pill_patch("ARMED" if armed else "DISARMED", ACCENT if armed else (150, 140, 135), filled=armed)
            blit(canvas, pp, W - pp.shape[1] - 150, 936, ba)
        # captions
        cap = captions_at(t, armed)
        if cap and sc["i"] != 9:
            draw_caption(canvas, cap, t, sc["start"] + sc["dur"] + (0.0 if sc["i"] != 8 else 0.0))
        # keycaps (scene 7)
        if S(7, 4.8) <= t < S(7) + SC[7]["dur"] + 0.2:
            a = smooth((t - S(7, 4.8)) / 0.5) * (1 - smooth((t - S(8, -0.4)) / 0.4))
            x = 960 - 330
            for j, (lab, wk) in enumerate((("Ctrl", 190), ("Alt", 170), ("F12", 170))):
                kp = key_patch(lab, wk)
                blit(canvas, kp, x, 330 + (1 - a) * 40 + (6 if (math.sin(t * 5 + j) > 0.95) else 0), a)
                if j < 2: blit(canvas, text_patch("+", 50, True, MUTED), x + wk + 14, 340 + (1 - a) * 40, a)
                x += wk + 78
        # outro
        if sc["i"] == 9:
            r = t - sc["start"]
            def item(patch, y, t_in, x=1140):
                a = smooth((r - t_in) / 0.6); blit(canvas, patch, x, y + (1 - a) * 22, a)
            item(text_patch("GestureOS", 108, True, WHITE, tracking=3), 300, 0.7)
            item(text_patch("Control your PC with your hands.", 34, False, MUTED), 450, 1.1)
            chips = ["Python", "OpenCV", "MediaPipe", "CustomTkinter"]
            cxp = 1150
            for j, c in enumerate(chips):
                pp = pill_patch(c, (170, 161, 154), size=22, filled=False)
                a = smooth((r - (1.8 + 0.18 * j)) / 0.5); blit(canvas, pp, cxp, 540 + (1 - a) * 16, a); cxp += pp.shape[1] - 6
            item(text_patch("Armed only by you  ·  600+ tests  ·  MIT", 28, False, MUTED), 640, 3.0)
            item(text_patch("github.com/Dhairya233/GestureOS", 36, True, ACCENT), 740, 4.0)
            item(text_patch("Demo rendered from the real engine with scripted hand input", 22, False, (120, 112, 106), shadow=False), 1020, 4.8, x=100)
        elif t > S(1, 1.0):
            a = 0.55
            blit(canvas, text_patch("Scripted hand input · real GestureOS engine, overlay and UI", 21, False, (150, 142, 136), shadow=False),
                 W - 700, 1030, a)
        # grade: vignette + fades
        o = canvas.astype(np.float32)
        f_in = smooth(t / 1.0); f_out = smooth((TOTAL - t) / 1.6)
        o *= min(f_in, f_out)
        wr.stdin.write(np.clip(o, 0, 255).astype(np.uint8).tobytes())
        if k % 150 == 0: print("frame", k, "/", n, flush=True)
    wr.stdin.close(); wr.wait(); rd.kill()
    json.dump(events, open("events.json", "w"))
    print("events", events)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "silent.mp4")
