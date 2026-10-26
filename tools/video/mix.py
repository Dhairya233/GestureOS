import json, subprocess, numpy as np, soundfile as sf
TL = json.load(open("timeline.json")); EV = json.load(open("events.json")); SC = {s["i"]: s for s in TL["scenes"]}
SR = 44100; N = int((TL["total"] + 1) * SR)
sfx = np.zeros((N, 2), np.float32)
def put(sig, t, g=1.0):
    i = int(t * SR); sig = sig[: N - i]
    sfx[i:i + len(sig), 0] += sig * g; sfx[i:i + len(sig), 1] += sig * g
def chime():
    t = np.arange(int(1.2 * SR)) / SR
    return sum(np.sin(2 * np.pi * f * t) * np.exp(-t * d) for f, d in ((880, 3.5), (1320, 4.5), (1760, 6))) / 3
def tick():
    t = np.arange(int(0.12 * SR)) / SR
    return np.sin(2 * np.pi * 1800 * t) * np.exp(-t * 60) * 0.8 + np.sin(2 * np.pi * 900 * t) * np.exp(-t * 40) * 0.4
def down():
    t = np.arange(int(0.9 * SR)) / SR
    return np.sin(2 * np.pi * (520 - 220 * t) * t) * np.exp(-t * 4)
for t in EV["armed"]: put(chime(), t, 0.45)
for t in EV["disarmed"]: put(down(), t, 0.45)
s3 = SC[3]
for u in (0.55, 0.78): put(tick(), s3["start"] + u * s3["dur"] + 0.1, 0.5)
put(tick(), SC[5]["start"] + 5.3, 0.4)
sf.write("sfx.wav", sfx, SR)
inputs = ["-i", "music.wav", "-i", "sfx.wav"]
fc = []; labels = []
for k, s in enumerate(TL["scenes"]):
    inputs += ["-i", f"vo/{s['i']}.wav"]
    ms = int(s["vo_at"] * 1000)
    fc.append(f"[{k+2}:a]aformat=sample_rates=44100:channel_layouts=stereo,highpass=f=70,equalizer=f=3500:t=q:w=1.2:g=2,"
              f"acompressor=threshold=-20dB:ratio=3:attack=8:release=120,volume=1.9,adelay={ms}|{ms}[v{k}]")
    labels.append(f"[v{k}]")
fc.append("".join(labels) + f"amix=inputs={len(labels)}:normalize=0,asplit=2[vo][vosc]")
fc.append("[0:a]volume=0.55[m]")
fc.append("[m][vosc]sidechaincompress=threshold=0.02:ratio=9:attack=15:release=450:makeup=1[md]")
fc.append("[1:a]volume=0.9[s]")
fc.append("[md][vo][s]amix=inputs=3:normalize=0,alimiter=limit=0.95,loudnorm=I=-16:TP=-1.5:LRA=9[out]")
subprocess.run(["ffmpeg", "-y", "-v", "error"] + inputs + ["-filter_complex", ";".join(fc), "-map", "[out]", "-t", str(TL["total"]),
               "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "audio.m4a"], check=True)
print("audio ok")
