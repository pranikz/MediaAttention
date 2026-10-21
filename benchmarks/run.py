"""Benchmark attention models against human eye tracking.

    python benchmarks/run.py fiwi  --models center,spectral,deepgaze_iie,deepgaze_msdb,transalnet,unisal
    python benchmarks/run.py ueyes --models ...

Datasets and third-party model repos live outside this repo, under $MA_BENCH
(default ~/Documents/Personal/MediaAttention-bench): data/<dataset>/ and models/<repo>/.
Writes benchmarks/results/<dataset>.json (summary) and <dataset>_per_image.csv.

Metrics (Bylinskii et al. 2019; Kuemmerer et al. 2018), computed per image with pysaliency
and averaged: AUC (all pixels as negatives), sAUC (other images' fixations as negatives),
NSS, CC, KLD (MIT/Tuebingen regularisation) and SIM. CC, KLD and SIM compare against
fixations blurred with the dataset's own Gaussian (see each loader).
"""
import argparse
import contextlib
import csv
import json
import os
import sys
import time
from functools import cache
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mediaattention.saliency import _resize, center_bias, deepgaze, device, spectral_residual, unisal  # noqa: E402

BENCH = Path(os.environ.get("MA_BENCH", "~/Documents/Personal/MediaAttention-bench")).expanduser()
OUT = Path(__file__).parent / "results"


# ---------------------------------------------------------------- datasets
# Each loader returns (items, meta). item: path, category, xs, ys (0-based pixels in the image).
# meta: ppd (pixels per degree of visual angle, in image pixels), sigma (ground-truth blur, px).

def load_fiwi():
    """FiWI (Shen & Zhao, ECCV 2014): 149 webpages, 1360x768, 11 viewers, 5 s free viewing."""
    from scipy.io import loadmat
    root = BENCH / "data/fiwi/dataset"
    cat = {p.name: c for c in ("mixed", "pictorial", "text") for p in (root / "stimuli" / c).glob("*.png")}
    items = []
    for e in loadmat(root / "fixations.mat", squeeze_me=True, struct_as_record=False)["fixations"]:
        xs, ys = [], []
        for s in np.atleast_1d(e.subjects):
            if hasattr(s, "fix_x"):  # some viewers have no data for some pages
                xs += list(np.atleast_1d(s.fix_x) - 1)  # MATLAB is 1-based
                ys += list(np.atleast_1d(s.fix_y) - 1)
        items.append({"path": root / "stimuli" / e.img, "category": cat[e.img], "xs": xs, "ys": ys})
    return items, {"ppd": 50.0, "sigma": 32.0}  # 1 deg ~ 50 px (paper); sigma from the dataset's constants.m


def load_ueyes(split="test"):
    """UEyes (Jiang et al., CHI 2023): webpages, desktop/mobile UIs and posters; Gazepoint GP3, 7 s free viewing.

    Test split from image_types.csv. One point per fixation (FPOGID, valid FPOGV), mapped from normalised
    screen coordinates to image pixels exactly as the dataset's generate_heatmaps.py does (image
    letterboxed on a 1920x1200 screen).
    """
    import pandas as pd
    root = BENCH / "data/ueyes/UEyes_dataset"
    info = pd.read_csv(root / "image_types.csv", sep=";")
    if split:
        info = info[info["Train/Test"].str.lower() == split]
    fix = {}
    for f in sorted((root / "eyetracker_logs").glob("*_fixations.csv")):
        d = pd.read_csv(f, usecols=["MEDIA_NAME", "FPOGX", "FPOGY", "FPOGID", "FPOGV"])
        d = d[d["FPOGV"] == 1].groupby(["MEDIA_NAME", "FPOGID"], as_index=False).last()  # final estimate per fixation
        for name, g in d.groupby("MEDIA_NAME"):
            fix.setdefault(name, []).append(g[["FPOGX", "FPOGY"]].to_numpy())
    items = []
    for _, row in info.iterrows():
        path = root / "images" / row["Image Name"]
        pts = fix.get(row["Image Name"])
        if pts is None or not path.exists():
            continue
        w, h = Image.open(path).size
        r = 1920 / 1200
        sx, sy = (h * r, h) if r >= w / h else (w, w / r)
        p = np.concatenate(pts)
        x, y = p[:, 0] * sx - (sx - w) / 2, p[:, 1] * sy - (sy - h) / 2
        items.append({"path": path, "category": row["Category"], "xs": list(x), "ys": list(y)})
    # 1 deg ~ 40 screen px (GP3 setup); converted to image px per image in per_image_scale()
    return items, {"ppd_screen": 40.0, "screen": (1920, 1200)}


# UEyes: no benchmarked model was trained on any of it, so the whole set (1,980 images) is fair;
# ueyes_test is the dataset's own 108-image test split, for comparison with papers that train on UEyes.
DATASETS = {"fiwi": load_fiwi, "ueyes": lambda: load_ueyes(split=None), "ueyes_test": load_ueyes}


def per_image_scale(item, meta):
    """Returns (ppd, sigma) in this image's pixels."""
    if "ppd" in meta:
        return meta["ppd"], meta["sigma"]
    w, h = Image.open(item["path"]).size
    k = 1 / min(meta["screen"][0] / w, meta["screen"][1] / h)  # image px per screen px
    return meta["ppd_screen"] * k, meta["ppd_screen"] * k


# ---------------------------------------------------------------- models
# Each takes (PIL RGB image, ppd) and returns an (H, W) non-negative map at the image size.

def m_center(img, ppd):
    return np.exp(center_bias((img.height, img.width)))


def m_spectral(img, ppd):
    return spectral_residual(img)


def m_unisal(img, ppd):
    """UNISAL, SALICON weights, as shipped in mediaattention (matches the original repo's CLI, CC > 0.9999)."""
    return unisal(img)


def m_deepgaze_iie(img, ppd):
    return deepgaze(img)


@cache
def _msdb():
    from deepgaze_pytorch import DeepGazeMSDB
    with contextlib.redirect_stdout(sys.stderr):
        return DeepGazeMSDB(pretrained=True).float().to(device()).eval()


def m_deepgaze_msdb(img, ppd):
    import torch
    s = min(1.0, 1024 / max(img.size))
    x = np.asarray(img.resize((round(img.width * s), round(img.height * s))), dtype=np.float32)
    with torch.no_grad():
        logd = _msdb()(torch.tensor(x.transpose(2, 0, 1)[None], device=device()),
                       torch.tensor(center_bias(x.shape[:2])[None], device=device()),
                       pixel_per_dva=ppd * s)  # averaged (dataset=None) parameters
    return _resize(np.exp(logd.float().cpu().numpy()[0]), img.size)


@cache
def _transalnet():
    import torch
    repo = BENCH / "models/TranSalNet"
    sys.path.insert(0, str(repo))
    cwd = os.getcwd()
    os.chdir(repo)  # it loads its backbone from a relative path
    try:
        from TranSalNet_Dense import TranSalNet
        m = TranSalNet()
        m.load_state_dict(torch.load(repo / "pretrained_models/TranSalNet_Dense.pth", map_location="cpu"))
    finally:
        os.chdir(cwd)
    return m.to(device()).eval()


def m_transalnet(img, ppd):
    """TranSalNet-Dense, SALICON weights; aspect-preserving resize + pad to 384x288, as in its repo."""
    import torch
    W, H = 384, 288
    s = min(W / img.width, H / img.height)
    nw, nh = round(img.width * s), round(img.height * s)
    canvas = Image.new("RGB", (W, H))
    ox, oy = (W - nw) // 2, (H - nh) // 2
    canvas.paste(img.resize((nw, nh)), (ox, oy))
    x = torch.tensor(np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255, device=device())
    with torch.no_grad():
        pred = _transalnet()(x)[0, 0].float().cpu().numpy()
    return _resize(pred[oy:oy + nh, ox:ox + nw], img.size)


MODELS = {"center": m_center, "spectral": m_spectral, "deepgaze_iie": m_deepgaze_iie,
          "deepgaze_msdb": m_deepgaze_msdb, "transalnet": m_transalnet, "unisal": m_unisal}


# ---------------------------------------------------------------- metrics

def fixation_map(shape, xs, ys, sigma):
    m = np.zeros(shape)
    np.add.at(m, (np.asarray(ys, int), np.asarray(xs, int)), 1)
    return gaussian_filter(m, sigma, mode="nearest")


def metrics(pred, xs, ys, gt, shuffled):
    from pysaliency import metrics as M
    from pysaliency.roc import general_roc
    pred = pred.astype(np.float64)
    pos = pred[ys, xs]
    sx = np.clip(shuffled[:, 0], 0, pred.shape[1] - 1).astype(int)
    sy = np.clip(shuffled[:, 1], 0, pred.shape[0] - 1).astype(int)
    return {"AUC": general_roc(pos, pred.ravel())[0], "sAUC": general_roc(pos, pred[sy, sx])[0],
            "NSS": float(np.mean(M.NSS(pred, xs, ys))), "CC": M.CC(pred, gt),
            "KLD": M.MIT_KLDiv(pred, gt), "SIM": M.SIM(pred, gt)}


def evaluate(dataset, model_names, limit=None):
    items, meta = DATASETS[dataset]()
    items = items[:limit] if limit else items
    # clean fixations: inside the image only, as integer pixels
    for it in items:
        w, h = Image.open(it["path"]).size
        x, y = np.asarray(it["xs"], float), np.asarray(it["ys"], float)
        ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        it["x"], it["y"], it["size"] = x[ok].astype(int), y[ok].astype(int), (w, h)
    items = [it for it in items if len(it["x"])]
    # shuffled-AUC negatives: other images' fixations, rescaled to relative coordinates
    rel = [np.stack([it["x"] / it["size"][0], it["y"] / it["size"][1]], 1) for it in items]
    cache_dir = BENCH / "cache" / dataset
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in model_names:
        t_total = 0.0
        for i, it in enumerate(items):
            img = Image.open(it["path"]).convert("RGB")
            ppd, sigma = per_image_scale(it, meta)
            t = time.perf_counter()
            pred = MODELS[name](img, ppd)
            t_total += time.perf_counter() - t
            gt = fixation_map((img.height, img.width), it["x"], it["y"], sigma)
            others = np.concatenate([r for j, r in enumerate(rel) if j != i]) * [img.width, img.height]
            m = metrics(pred, it["x"], it["y"], gt, others)
            rows.append({"model": name, "image": it["path"].name, "category": it["category"], **m})
            print(f"\r{name}: {i + 1}/{len(items)}", end="", file=sys.stderr)
        print(f"\r{name}: done, {t_total / len(items):.2f} s/image", file=sys.stderr)
        rows.append({"model": name, "image": "__time__", "category": "", "s_per_image": t_total / len(items)})
    return items, rows


def summarise(rows):
    keys = ["AUC", "sAUC", "NSS", "CC", "KLD", "SIM"]
    out = {}
    for name in dict.fromkeys(r["model"] for r in rows):
        rs = [r for r in rows if r["model"] == name and r["image"] != "__time__"]
        cats = sorted({r["category"] for r in rs})
        out[name] = {
            "n": len(rs),
            "s_per_image": next(r["s_per_image"] for r in rows if r["model"] == name and r["image"] == "__time__"),
            "all": {k: [round(float(np.mean([r[k] for r in rs])), 4), round(float(np.std([r[k] for r in rs]) / np.sqrt(len(rs))), 4)] for k in keys},
            "by_category": {c: {k: round(float(np.mean([r[k] for r in rs if r["category"] == c])), 4) for k in keys} for c in cats},
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=DATASETS)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--limit", type=int, help="first N images only (smoke test)")
    a = ap.parse_args()
    names = a.models.split(",")
    _, rows = evaluate(a.dataset, names, a.limit)
    OUT.mkdir(exist_ok=True)
    suffix = f"_n{a.limit}" if a.limit else ""
    summary = summarise(rows)
    old = OUT / f"{a.dataset}{suffix}.json"
    merged = {**(json.loads(old.read_text()) if old.exists() else {}), **summary}  # re-running one model keeps the rest
    old.write_text(json.dumps(merged, indent=2))
    with open(OUT / f"{a.dataset}{suffix}_per_image_{'-'.join(names)}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({k: v["all"] for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
