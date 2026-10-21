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
    Without campaign results this ranks assets by the metric itself; see ranges_from_kpi for the KPI version.
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


def read_kpis(path, column=None):
    """CSV with a `file` column (filename or path relative to the folder) and KPI columns -> ({file: kpi}, column)."""
    import csv
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "file" not in rows[0]:
        raise ValueError(f"{path}: needs a 'file' column and at least one KPI column")
    column = column or next(c for c in rows[0] if c != "file")
    return {r["file"].strip(): float(r[column]) for r in rows if (r.get(column) or "").strip()}, column


def ranges_from_kpi(by_file, kpis, lower_better=False, top=0.2, min_n=5):
    """Targets from what your best-performing ads scored, plus which metrics track the KPI.

    by_file: {file: metrics}; kpis: {file: value}. The winners are the top `top` share by KPI. Each
    metric's range is the middle 50% of the winners' scores. Returns (ranges, report): the report gives
    every metric's Spearman correlation with the KPI across all matched assets.
    """
    from scipy.stats import spearmanr
    files = [f for f in by_file if f in kpis]
    sign = -1 if lower_better else 1
    k = np.array([kpis[f] for f in files]) * sign
    if len(files) < min_n / top:
        raise ValueError(f"only {len(files)} assets have a KPI; need {int(min_n / top)}+ to have {min_n} winners")
    winners = [f for f, v in zip(files, k) if v >= np.quantile(k, 1 - top)]
    names = sorted({m for f in files for m in by_file[f]} - UNSCORED)
    ranges, corr = {}, {}
    for m in names:
        win = [by_file[f][m] for f in winners if by_file[f].get(m) is not None]
        if len(win) >= min_n:
            lo, hi = np.percentile(win, [25, 75])
            ranges[m] = [round(float(lo), 1), round(float(hi), 1)]
        pairs = [(by_file[f][m], kpis[f] * sign) for f in files if by_file[f].get(m) is not None]
        if len(pairs) >= min_n:
            rho, pval = spearmanr(*zip(*pairs))
            corr[m] = {"spearman_rho": round(float(rho), 3), "p": round(float(pval), 4), "n": len(pairs)}
    report = {"assets_with_kpi": len(files), "winners": len(winners), "lower_is_better": lower_better,
              "correlation_with_kpi": dict(sorted(corr.items(), key=lambda kv: -abs(kv[1]["spearman_rho"])))}
    return ranges, report


def _kpi_key(path, folder, kpis):
    """Match an asset to its KPI row by path relative to the folder, else by filename."""
    if folder:
        rel = str(Path(path).relative_to(folder)) if Path(path).is_relative_to(folder) else None
        if rel in kpis:
            return rel
    return Path(path).name if Path(path).name in kpis else None


def bench(folder, out, **kw):
    """Scores every image/video under `folder`. See bench_files."""
    return bench_files(sorted(Path(folder).rglob("*")), out, folder=folder, **kw)


def bench_files(files, out, aois=None, product=None, model="auto", fps=2.0, min_n=5, on_file=None,
                kpi=None, kpi_column=None, kpi_lower_better=False, folder=None):
    """Scores the image/video files among `files` (others are ignored), writes out/assets.jsonl and
    out/benchmarks.json. on_file(i, total, path) is called before each one. Returns (ranges, scored, skipped).

    kpi: CSV of campaign results (see read_kpis). With it, targets come from the top 20% by KPI and
    out/kpi_report.json says which metrics track the KPI.
    """
    kpis, kpi_column = read_kpis(kpi, kpi_column) if kpi else ({}, None)
    files = [Path(p) for p in files if Path(p).suffix.lower() in IMAGE_EXT | VIDEO_EXT]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows, skipped, by_file = [], [], {}
    with open(out / "assets.jsonl", "w") as f:
        for i, p in enumerate(files, 1):
            print(f"[{i}/{len(files)}] {p}", file=sys.stderr)
            if on_file:
                on_file(i, len(files), p)
            try:
                report, _, _ = score_path(p, aois, product, model, None, fps)
            except Exception as e:  # one broken file shouldn't sink the corpus
                print(f"  skipped: {e}", file=sys.stderr)
                skipped.append(p.name)
                continue
            rows.append(report["metrics"])
            key = _kpi_key(p, folder, kpis) if kpis else None
            if key:
                by_file[key] = report["metrics"]
            f.write(json.dumps({"file": str(p), "metrics": report["metrics"]}) + "\n")
    if kpis:
        ranges, report = ranges_from_kpi(by_file, kpis, kpi_lower_better, min_n=min_n)
        report["kpi"] = kpi_column
        (out / "kpi_report.json").write_text(json.dumps(report, indent=2))
    else:
        ranges = ranges_from(rows, min_n)
    (out / "benchmarks.json").write_text(json.dumps(ranges, indent=2))
    if len(rows) < 30:
        print(f"warning: only {len(rows)} assets scored; ranges get stable around 30+ per category", file=sys.stderr)
    return ranges, len(rows), skipped
