"""Scores derived from a predicted attention heatmap. Pure numpy, no models."""
import numpy as np

# ponytail: placeholder ranges (metric -> [lo, hi], all on 0-100). Replace with the
# top-20% band of a real scored corpus per industry/use case, as Neurons does.
DEFAULT_RANGES = {
    "focus": [60, 100],
    "cognitive_demand": [25, 55],
    "brand_attention": [10, 100],
    "product_attention": [15, 100],
    "cta_attention": [5, 100],
}


def normalize(heat):
    h = np.clip(np.asarray(heat, dtype=np.float64), 0, None)
    s = h.sum()
    return h / s if s > 0 else np.full(h.shape, 1 / h.size)


def focus(heat, mass=0.8):
    """0-100. 100 = attention lands on a tiny area, 0 = spread evenly.

    Attention dispersion: the share of the image needed to hold `mass` of all attention.
    """
    p = np.sort(normalize(heat).ravel())[::-1]
    k = np.searchsorted(np.cumsum(p), mass) + 1
    return float(100 * max(0.0, 1 - (k / p.size) / mass))


def cognitive_demand(gray, block=16, levels=32):
    """0-100 visual complexity: mean Shannon entropy of local blocks of the grayscale image."""
    g = np.asarray(gray, dtype=np.float64)
    h, w = (g.shape[0] // block) * block, (g.shape[1] // block) * block
    if not h or not w:
        return 0.0
    q = np.minimum((g[:h, :w] / 256 * levels).astype(int), levels - 1)
    blocks = q.reshape(h // block, block, w // block, block).transpose(0, 2, 1, 3).reshape(-1, block * block)
    ids = np.arange(len(blocks))[:, None] * levels + blocks
    p = np.bincount(ids.ravel(), minlength=len(blocks) * levels).reshape(-1, levels) / (block * block)
    ent = -(p * np.log2(p, where=p > 0, out=np.zeros_like(p))).sum(1)
    return float(100 * ent.mean() / np.log2(levels))


def aoi_stats(heat, aois):
    """aois: [{"label": str, "box": [x0, y0, x1, y1]}] with coords in 0-1."""
    p = normalize(heat)
    H, W = p.shape
    out = []
    for a in aois:
        x0, y0, x1, y1 = a["box"]
        r0, c0 = int(y0 * H), int(x0 * W)
        region = p[r0:max(int(y1 * H), r0 + 1), c0:max(int(x1 * W), c0 + 1)]
        area, att = region.size / p.size, region.sum()
        out.append({**a, "attention_pct": round(100 * att, 1), "area_pct": round(100 * area, 1),
                    "lift": round(att / area, 2)})  # >1 = gets more attention than its size
    return out


def impact_score(metrics, ranges=DEFAULT_RANGES):
    """1-10 per metric vs its recommended range, then the plain average (Neurons' published method).

    Inside the range = 10; outside drops linearly from 7 at the edge to 1 at the scale's end.
    Metrics missing from `metrics` are skipped, not penalised.
    """
    parts = {}
    for k, (lo, hi) in ranges.items():
        v = metrics.get(k)
        if v is None:
            continue
        if lo <= v <= hi:
            s = 10.0
        elif v < lo:
            s = 1 + 6 * v / lo
        else:
            s = 1 + 6 * (100 - v) / (100 - hi)
        parts[k] = round(s, 1)
    return (round(sum(parts.values()) / len(parts), 1) if parts else None), parts
