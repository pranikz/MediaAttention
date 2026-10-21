"""Turn study results from the browser kit (docs/collect/) into a dataset and per-ad scores.

    mediaattention collect results/*.json --out study/

Writes:
  fixations.csv  participant, ad, x, y (0-1 in the ad), t_ms, dur_ms: ground truth for benchmarks/run.py
  scores.csv     one row per ad: implicit scores (engagement, trust, avoidance, intent: 0-100) and memory
                 (recognition hit rate minus false alarms). Its `file` column matches `bench --kpi`.
  quality.csv    per participant: calibration error and how much data was kept
"""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

RT_MIN, RT_MAX = 300, 2500  # ms; faster is a slip, slower is deliberation rather than a gut reaction
MAX_CALIB_ERROR = 0.12      # fraction of the screen diagonal; worse calibrations are excluded


def fixations(gaze, dispersion=0.03, min_dur=100):
    """I-DT fixation detection on [(t_ms, x, y)] in 0-1 ad coordinates -> [(x, y, t_ms, dur_ms)]."""
    out, i, n = [], 0, len(gaze)
    while i < n:
        j = i
        while j + 1 < n and gaze[j + 1][0] - gaze[i][0] < min_dur:
            j += 1
        win = gaze[i:j + 1]
        if j + 1 >= n or _spread(win) > dispersion:
            i += 1
            continue
        while j + 1 < n and _spread(gaze[i:j + 2]) <= dispersion:
            j += 1
        win = gaze[i:j + 1]
        x, y = sum(p[1] for p in win) / len(win), sum(p[2] for p in win) / len(win)
        if 0 <= x <= 1 and 0 <= y <= 1:  # fixations off the ad are not attention on it
            out.append((round(x, 4), round(y, 4), win[0][0], win[-1][0] - win[0][0]))
        i = j + 1
    return out


def _spread(win):
    xs, ys = [p[1] for p in win], [p[2] for p in win]
    return (max(xs) - min(xs)) + (max(ys) - min(ys))


def implicit_scores(answers_by_participant):
    """{participant: [(ad, scale, yes, rt)]} -> {ad: {scale: 0-100}}.

    Each answer counts +1 (yes) or -1 (no), weighted by how fast it was for that person: reaction times
    are z-scored per participant on a log scale, and weight = 1 / (1 + e^z), so a fast answer weighs
    more than a hesitant one. Answers outside 300-2500 ms are dropped. Per ad and scale, the weighted
    answers are averaged across words and participants and mapped from [-1, 1] to 0-100.
    """
    acc = defaultdict(lambda: defaultdict(list))
    for answers in answers_by_participant.values():
        kept = [a for a in answers if RT_MIN <= a[3] <= RT_MAX]
        if len(kept) < 3:
            continue
        logs = [math.log(a[3]) for a in kept]
        mu = sum(logs) / len(logs)
        sd = (sum((v - mu) ** 2 for v in logs) / len(logs)) ** 0.5 or 1.0
        for (ad, scale, yes, rt), lg in zip(kept, logs):
            w = 1 / (1 + math.exp((lg - mu) / sd))
            acc[ad][scale].append((1 if yes else -1) * w * 2)  # w is 0.5 on average; x2 keeps the scale near [-1, 1]
    return {ad: {s: round(50 * (max(-1, min(1, sum(v) / len(v))) + 1), 1) for s, v in scales.items()}
            for ad, scales in acc.items()}


def import_results(paths, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fix_rows, quality, answers, recog, srcs = [], [], {}, defaultdict(list), {}
    for p in paths:
        r = json.loads(Path(p).read_text())
        if r.get("format") != "mediaattention-study/1":
            raise ValueError(f"{p}: not a MediaAttention study result")
        err = (r.get("calibration") or {}).get("mean_error_frac_diag", 1.0)
        ok = r.get("sim") or err <= MAX_CALIB_ERROR
        n_fix = 0
        if ok:
            for t in r["trials"]:
                if "gaze" not in t:  # the ad failed to load for this participant
                    continue
                srcs[t["ad"]] = t["src"]
                for x, y, t_ms, dur in fixations(t["gaze"]):
                    fix_rows.append({"participant": r["participant"], "ad": t["ad"], "x": x, "y": y, "t_ms": t_ms, "dur_ms": dur})
                    n_fix += 1
            answers[r["participant"]] = [(t["ad"], a["scale"], a["yes"], a["rt"]) for t in r["trials"] for a in t.get("answers", [])]
            for item in r.get("recognition", []):
                recog[item["ad"]].append((item["old"], item["said_seen"]))
        quality.append({"participant": r["participant"], "calibration_error": err, "kept": ok, "fixations": n_fix,
                        "simulated": bool(r.get("sim"))})
    scores = implicit_scores(answers)
    false_alarm = [said for items in recog.values() for old, said in items if not old]
    fa = sum(false_alarm) / len(false_alarm) if false_alarm else None
    rows = []
    for ad, src in sorted(srcs.items()):
        row = {"file": Path(src).name, "ad": ad, "participants": len({f["participant"] for f in fix_rows if f["ad"] == ad}),
               **scores.get(ad, {})}
        hits = [said for old, said in recog.get(ad, []) if old]
        if hits:
            row["memory_hit_rate"] = round(100 * sum(hits) / len(hits), 1)
            if fa is not None:
                row["memory"] = round(100 * max(0.0, sum(hits) / len(hits) - fa), 1)
        rows.append(row)
    _write(out / "fixations.csv", fix_rows, ["participant", "ad", "x", "y", "t_ms", "dur_ms"])
    _write(out / "scores.csv", rows, ["file", "ad", "participants", "engagement", "trust", "avoidance", "intent",
                                      "memory_hit_rate", "memory"])
    _write(out / "quality.csv", quality, ["participant", "calibration_error", "kept", "fixations", "simulated"])
    (out / "ads.json").write_text(json.dumps(srcs, indent=2))
    return rows, quality


def _write(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
