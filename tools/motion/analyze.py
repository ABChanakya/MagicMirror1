import csv, os, collections, json, statistics as st

base = os.path.expanduser("~/Coding/MagicMirror1/unity/GestureMotionPOC/EnvTests/capture")
rows = list(csv.DictReader(open(os.path.join(base, "metrics.csv"))))
clips = collections.OrderedDict()
for r in rows:
    clips.setdefault(r["clip"], []).append({k: (float(v) if k not in ("clip",) else v) for k, v in r.items()})

def cls(name):
    if "swipe_right" in name: return "swipe_right"
    if "swipe_left" in name: return "swipe_left"
    if "null" in name: return "null"
    return "other"

out = []
for name, fr in clips.items():
    if len(fr) < 8: continue
    hips_y = st.mean(f["hips_y"] for f in fr)
    def stats(pref):
        hy = [f[pref + "_y"] for f in fr]; hx = [f[pref + "_x"] for f in fr]; hz = [f[pref + "_z"] for f in fr]
        vx = [f[pref + "_vx"] for f in fr]; vy = [f[pref + "_vy"] for f in fr]
        return {"peak_h": max(hy), "rise": max(hy) - min(hy), "path": sum(((hx[i+1]-hx[i])**2 + (hy[i+1]-hy[i])**2 + (hz[i+1]-hz[i])**2) ** .5 for i in range(len(hx)-1)),
                "x_range": max(hx) - min(hx), "vx_range": max(vx) - min(vx), "in_frame": sum(1 for a, b in zip(vx, vy) if 0.02 < a < 0.98 and 0.02 < b < 0.98) / len(vx),
                "vx_start": vx[0], "vx_end": vx[-1], "peak_vx": vx[max(range(len(hy)), key=lambda i: hy[i])]}
    R, L = stats("rh"), stats("lh")
    drift = ((fr[-1]["hips_x"] - fr[0]["hips_x"]) ** 2 + (fr[-1]["hips_z"] - fr[0]["hips_z"]) ** 2) ** .5
    hips_range = max(f["hips_y"] for f in fr) - min(f["hips_y"] for f in fr)
    c = cls(name)
    active = R if R["path"] > L["path"] else L
    active_name = "right" if active is R else "left"
    flags = []
    if c in ("swipe_right", "swipe_left"):
        want = "right" if c == "swipe_right" else "left"
        if active_name != want: flags.append("wrong-hand-moves(%s active)" % active_name)
        if active["peak_h"] < hips_y + 0.15: flags.append("hand-never-above-hips+15cm")
        if active["path"] < 0.5: flags.append("small-motion(path %.2fm)" % active["path"])
        if active["in_frame"] < 0.75: flags.append("hand-leaves-dataset-frame(%d%%)" % (100 * active["in_frame"]))
    if c == "null" and max(R["path"], L["path"]) > 0.8: flags.append("null-but-big-motion(%.2fm)" % max(R["path"], L["path"]))
    if drift > 0.25: flags.append("hips-drift %.2fm" % drift)
    if hips_range > 0.25: flags.append("hips-bob %.2fm" % hips_range)
    out.append({"clip": name, "cls": c, "active": active_name, "peak_h": round(active["peak_h"], 2), "peak_above_hips": round(active["peak_h"] - hips_y, 2),
                "path": round(active["path"], 2), "vx_range": round(active["vx_range"], 2), "in_frame": round(active["in_frame"], 2),
                "vx_start": round(active["vx_start"], 2), "vx_end": round(active["vx_end"], 2), "peak_vx": round(active["peak_vx"], 2), "flags": flags})

json.dump(out, open(os.path.join(base, "summary.json"), "w"), indent=1)
by = collections.defaultdict(list)
for o in out: by[o["cls"]].append(o)
for c, lst in by.items():
    n = len(lst); fl = [o for o in lst if o["flags"]]
    print(f"== {c}: {n} clips, {len(fl)} flagged")
    for key in ("peak_above_hips", "path", "vx_range", "in_frame"):
        vals = sorted(o[key] for o in lst)
        print(f"   {key}: min {vals[0]}  median {vals[len(vals)//2]}  max {vals[-1]}")
    cnt = collections.Counter(f.split("(")[0].split(" ")[0] for o in fl for f in o["flags"])
    print("   flag types:", dict(cnt))
    print("   active hand:", dict(collections.Counter(o["active"] for o in lst)))
