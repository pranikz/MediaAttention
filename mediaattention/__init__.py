"""MediaAttention: open-source predicted attention and creative scoring."""
import numpy as np

from . import memory as _memory
from .metrics import DEFAULT_RANGES, aoi_stats, cognitive_demand, focus, impact_score
from .saliency import predict

__version__ = "0.2.0"


def score(img, aois=None, model="auto", ranges=None):
    """Score a PIL image. Returns (heatmap, report dict)."""
    heat, used = predict(img, model)
    metrics = {"focus": round(focus(heat), 1),
               "cognitive_demand": round(cognitive_demand(np.asarray(img.convert("L"))), 1)}
    if _memory.available():
        metrics["memory"] = _memory.memory(img)
    stats = aoi_stats(heat, aois or [])
    for a in stats:  # a role can have several boxes (e.g. two headline blocks): sum their attention
        k = f"{a['label']}_attention"
        metrics[k] = round(metrics.get(k, 0) + a["attention_pct"], 1)
    impact, parts = impact_score(metrics, ranges or DEFAULT_RANGES)
    return heat, {"model": used, "metrics": metrics, "aois": stats, "impact_score": impact, "impact_parts": parts}


def score_path(path, aois=None, product=None, model="auto", ranges=None, fps=2.0, heat_mp4=None, per_frame=False):
    """Score an image or video file. aois: None, a list, or "auto". Returns (report, img, heat); img/heat are None for video."""
    from pathlib import Path

    from PIL import Image

    from .video import VIDEO_EXT
    if Path(path).suffix.lower() in VIDEO_EXT:
        from .video import score_video
        return score_video(path, aois, product, model, ranges, fps, heat_mp4=heat_mp4, per_frame=per_frame), None, None
    img = Image.open(path)
    if aois == "auto":
        from .detect import detect
        aois = detect(img, product=product)
    heat, report = score(img, aois, model, ranges)
    return report, img, heat
