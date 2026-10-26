import soundfile as sf, json, sys
from kokoro_onnx import Kokoro
K = Kokoro("tts/kokoro.onnx", "tts/voices.bin")
LINES = [
 "Your hands are already the best input device you own. GestureOS turns an ordinary webcam into a way to control your Windows PC.",
 "A hand-tracking model finds twenty-one points on each hand, many times every second. GestureOS turns those points into gestures, and shows you exactly what it sees.",
 "It starts disarmed. It watches, but sends nothing. Hold both palms open for a second, and it arms.",
 "Point to move the cursor. Pinch your thumb to your middle finger to click.",
 "Hold an open palm high or low in the frame, and the page scrolls.",
 "Your left hand runs media and apps. A peace sign toggles play and pause. Hold three fingers up, and your favourite site launches.",
 "Draw in the air with your index finger. Sweep your palm to the edge of the frame, and it switches windows.",
 "Safety comes first. Two open palms disarm it instantly, and Control Alt F twelve stops everything, from any window.",
 "Every gesture has a visual guide. Every action lands in a live event log. And a diagnostics view shows the camera and the system's health.",
 "Built with Python, OpenCV, MediaPipe and CustomTkinter, and backed by over six hundred automated tests. Everything you saw ran on the real engine, driven here by scripted hands. GestureOS. Control your PC with your hands.",
]
voice = sys.argv[1] if len(sys.argv) > 1 else "af_heart"
meta = []
for i, t in enumerate(LINES):
    a, sr = K.create(t, voice=voice, speed=0.98, lang="en-us")
    sf.write(f"prod/vo/{i}.wav", a, sr)
    meta.append({"i": i, "dur": len(a) / sr, "text": t})
    print(i, round(len(a) / sr, 2))
json.dump(meta, open("prod/vo/meta.json", "w"), indent=1)
