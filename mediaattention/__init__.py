"""MediaAttention: open-source predicted attention and creative scoring."""
import numpy as np

from .metrics import DEFAULT_RANGES, aoi_stats, cognitive_demand, focus, impact_score
from .saliency import predict

__version__ = "0.1.0"


def score(img, aois=None, model="auto", ranges=None):
    """Score a PIL image. Returns (heatmap, report dict)."""
    heat, used = predict(img, model)
    metrics = {"focus": round(focus(heat), 1),
               "cognitive_demand": round(cognitive_demand(np.asarray(img.convert("L"))), 1)}
    stats = aoi_stats(heat, aois or [])
    for a in stats:
        metrics.setdefault(f"{a['label']}_attention", a["attention_pct"])  # first AOI per label
    impact, parts = impact_score(metrics, ranges or DEFAULT_RANGES)
    return heat, {"model": used, "metrics": metrics, "aois": stats, "impact_score": impact, "impact_parts": parts}
