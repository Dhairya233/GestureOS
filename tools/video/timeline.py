import json
meta = json.load(open("prod/vo/meta.json"))
LEAD, TAIL = 0.5, 0.8
MIN = {3: 6.8, 4: 5.8, 2: 8.0, 6: 9.0, 5: 10.5, 7: 9.5, 8: 12.0}
INTRO_PAD = 1.2          # extra silence-with-music before first word
scenes, t = [], 0.0
for m in meta:
    i = m["i"]
    d = max(m["dur"] + LEAD + TAIL + (INTRO_PAD if i == 0 else 0), MIN.get(i, 0))
    scenes.append({"i": i, "start": t, "dur": d, "vo_at": t + LEAD + (INTRO_PAD if i == 0 else 0), "vo_dur": m["dur"]})
    t += d
TOTAL = t + 2.0          # outro hold with music tail
json.dump({"scenes": scenes, "total": TOTAL}, open("prod/timeline.json", "w"), indent=1)
if __name__ == "__main__":
    for s in scenes: print(s["i"], round(s["start"], 1), round(s["dur"], 1))
    print("total", round(TOTAL, 1))
