"""Recommended ranges from your own corpus: score a folder, take the top band of each metric."""
import json
import sys
from pathlib import Path

import numpy as np

from . import score_path
from .video import VIDEO_EXT

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}
TWO_SIDED = {"cognitive_demand"}  # moderate is best: too simple bores, too complex overloads
UNSCORED = {"body_attention"}  # more attention on body copy doesn't predict results, so it stays out of the Impact Score


def ranges_from(metrics, min_n=5):
    """metrics: one {name: value} dict per asset -> {name: [lo, hi]} recommended ranges.

    Higher-is-better metrics: the top 20% (Neurons' method). Two-sided metrics: the middle 50%.
    Metrics seen in fewer than `min_n` assets are left out.
    ponytail: ranks by the metric itself; rank by real campaign KPIs (CTR, recall) once you have them.
    """
    values = {}
    for m in metrics:
        for k, v in m.items():
            if v is not None:
                values.setdefault(k, []).append(v)
    out = {}
    for k, v in values.items():
        if len(v) >= min_n and k not in UNSCORED:
            lo, hi = np.percentile(v, [25, 75]) if k in TWO_SIDED else (np.percentile(v, 80), 100)
            out[k] = [round(float(lo), 1), round(float(hi), 1)]
    return out


def bench(folder, out, aois=None, product=None, model="auto", fps=2.0, min_n=5):
    """Scores every image/video under `folder`, writes out/assets.jsonl and out/benchmarks.json."""
    files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXT | VIDEO_EXT)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(out / "assets.jsonl", "w") as f:
        for i, p in enumerate(files, 1):
            print(f"[{i}/{len(files)}] {p}", file=sys.stderr)
            try:
                report, _, _ = score_path(p, aois, product, model, None, fps)
            except Exception as e:  # one broken file shouldn't sink the corpus
                print(f"  skipped: {e}", file=sys.stderr)
                continue
            rows.append(report["metrics"])
            f.write(json.dumps({"file": str(p), "metrics": report["metrics"]}) + "\n")
    ranges = ranges_from(rows, min_n)
    (out / "benchmarks.json").write_text(json.dumps(ranges, indent=2))
    if len(rows) < 30:
        print(f"warning: only {len(rows)} assets scored; ranges get stable around 30+ per category", file=sys.stderr)
    return ranges, len(rows)
