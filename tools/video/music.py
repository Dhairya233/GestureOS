"""Original generative score (numpy only): pads, bass, plucks, soft drums, risers, reverb."""
import json, numpy as np, soundfile as sf
from scipy.signal import butter, sosfilt, fftconvolve
SR = 44100
tl = json.load(open("prod/timeline.json")); TOTAL = tl["total"]; S = tl["scenes"]
N = int((TOTAL + 1) * SR)
rng = np.random.default_rng(7)
BPM = 96; beat = 60 / BPM
def f(m): return 440 * 2 ** ((m - 69) / 12)
def lp(x, fc, o=2): return sosfilt(butter(o, fc, "low", fs=SR, output="sos"), x)
def hp(x, fc, o=2): return sosfilt(butter(o, fc, "high", fs=SR, output="sos"), x)
def env(n, a, r):
    e = np.ones(n); na = int(a * SR); nr = int(r * SR)
    if na: e[:na] = np.linspace(0, 1, na)
    if nr: e[-nr:] *= np.linspace(1, 0, nr)
    return e
L = np.zeros(N); R = np.zeros(N)
def add(sig, t0, pan=0.0, gain=1.0):
    i = int(t0 * SR)
    if i >= N: return
    sig = sig[: N - i]
    L[i:i + len(sig)] += sig * gain * (1 - max(pan, 0)); R[i:i + len(sig)] += sig * gain * (1 + min(pan, 0))
# chords (midi), 2 bars each (8 beats)
CH = [(57, [57, 60, 64]), (53, [53, 57, 60]), (48, [55, 60, 64]), (55, [55, 59, 62])]
bar = 4 * beat; sec_len = 2 * bar
# intensity per scene start (0..1)
def intensity(t):
    for s in S:
        if s["start"] <= t < s["start"] + s["dur"]:
            return {0: .25, 1: .5, 2: .55, 3: .75, 4: .75, 5: .85, 6: .9, 7: .7, 8: .8, 9: 1.0}[s["i"]]
    return .6
k = 0; t = 0.0
while t < TOTAL + 1:
    root, notes = CH[k % 4]; inten = intensity(t)
    n = int(sec_len * SR); tt = np.arange(n) / SR
    pad = np.zeros(n)
    for m in notes:
        for det in (-0.12, 0.0, 0.12):
            fr = f(m) * 2 ** (det / 12)
            pad += (2 * ((tt * fr) % 1) - 1) * 0.5 + np.sin(2 * np.pi * fr * tt) * 0.5
    pad = lp(pad / 9, 700 + 1400 * inten) * env(n, 1.2, 1.4) * (0.5 + 0.35 * inten)
    add(pad, t, 0.0, 0.9)
    bn = int(beat * 2 * SR); bt = np.arange(bn) / SR
    for j in range(4):
        b = np.sin(2 * np.pi * f(root - 12) * bt) * np.exp(-bt * 1.6) * 0.6
        add(b, t + j * 2 * beat, 0, 0.5 * (0.5 + inten * 0.5))
    # plucks (8ths), from scene 1
    if t >= S[1]["start"] - 0.1:
        seq = [0, 1, 2, 1, 2, 1, 0, 1] if k % 2 == 0 else [2, 1, 0, 1, 2, 3, 2, 1]
        arp = notes + [notes[0] + 12]
        for j in range(16):
            m = arp[seq[j % 8] % 4] + 12
            nn = int(0.45 * SR); tn = np.arange(nn) / SR
            p = (np.sin(2 * np.pi * f(m) * tn) + 0.35 * np.sin(2 * np.pi * 2 * f(m) * tn)) * np.exp(-tn * 7)
            add(p, t + j * beat / 2, (-0.5 if j % 2 else 0.5), 0.10 * (0.4 + inten))
    # drums from scene 2
    if t >= S[2]["start"] - 0.1:
        for j in range(8):
            tb = t + j * beat
            kn = int(0.3 * SR); kt = np.arange(kn) / SR
            kick = np.sin(2 * np.pi * (50 + 90 * np.exp(-kt * 28)) * kt) * np.exp(-kt * 10)
            add(kick, tb, 0, 0.55 * inten)
            for h in (0.5,):
                hn = int(0.05 * SR)
                hat = hp(rng.normal(size=hn), 7000) * np.exp(-np.arange(hn) / SR * 90)
                add(hat, tb + h * beat, 0.3, 0.07 * inten)
            if j % 2 == 1:
                sn = int(0.18 * SR); st = np.arange(sn) / SR
                snare = hp(rng.normal(size=sn), 1800) * np.exp(-st * 25)
                add(snare, tb, 0, 0.12 * inten)
    k += 1; t += sec_len
# risers into scene changes
for s in S[1:]:
    rn = int(1.6 * SR); rt = np.arange(rn) / SR
    ns = rng.normal(size=rn); ns = lp(ns, 6000) * (rt / 1.6) ** 2
    add(hp(ns, 800), s["start"] - 1.6, 0, 0.05)
    add(np.sin(2 * np.pi * 70 * np.arange(int(0.9 * SR)) / SR) * np.exp(-np.arange(int(0.9 * SR)) / SR * 4), s["start"], 0, 0.35)
# reverb
irn = int(2.2 * SR); itt = np.arange(irn) / SR
ir = [lp(rng.normal(size=irn), 3500) * np.exp(-itt * 2.4) for _ in range(2)]
wl = fftconvolve(L, ir[0])[:N] * 0.0 + 0
Lw = fftconvolve(L, ir[0])[:N] * 0.06; Rw = fftconvolve(R, ir[1])[:N] * 0.06
L = L + Lw; R = R + Rw
out = np.stack([L, R], 1)
fade = np.ones(N); fi = int(2 * SR); fo = int(3.2 * SR)
fade[:fi] = np.linspace(0, 1, fi); fade[int(TOTAL * SR) - fo:int(TOTAL * SR)] = np.linspace(1, 0, fo); fade[int(TOTAL * SR):] = 0
out *= fade[:, None]
out = out / np.percentile(np.abs(out), 99.5) * 0.6; out = np.tanh(out) * 0.9
sf.write("prod/music.wav", out.astype(np.float32), SR)
print("music", out.shape[0] / SR)
