"""Video scoring: sample frames, smooth attention over time, track cuts and AOIs. Needs the `video` extra."""
from collections import deque
from fractions import Fraction

import numpy as np

from .metrics import DEFAULT_RANGES, aoi_stats, cognitive_demand, focus, impact_score
from .saliency import predict

VIDEO_EXT = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
CUT = 0.4  # histogram change (0-1) between consecutive frames that counts as a hard cut
MIN_SHOT = 0.5  # seconds; ignores flashes and encoder flicker right after a cut


def frames(path, fps=2.0):
    """Yields (t, PIL image, cut) sampled at `fps`. cut = time of a hard cut since the previous sample, else None.

    Cuts are checked on every decoded frame (on tiny grayscale thumbnails), not just the sampled ones.
    """
    import av
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        next_t, prev, cut, last_cut = None, None, None, -MIN_SHOT
        for f in c.decode(s):
            t = float(f.time or 0)
            hist = np.bincount(f.reformat(width=64, height=36, format="gray").to_ndarray().ravel() // 8, minlength=32)
            hist = hist / hist.sum()
            if prev is not None and np.abs(hist - prev).sum() / 2 > CUT and t - last_cut >= MIN_SHOT:
                cut = last_cut = t
            prev = hist
            if next_t is None or t + 1e-6 >= next_t:
                yield t, f.to_image(), cut
                cut, next_t = None, (next_t or t) + 1 / fps


class _Mp4:
    """Streams PIL frames into an mp4. Frames are scaled to <=1280 wide with even sides (yuv420p)."""

    def __init__(self, path, fps):
        import av
        self.av, self.out, self.fps, self.stream = av, av.open(str(path), "w"), Fraction(fps).limit_denominator(), None

    def write(self, img):
        if self.stream is None:
            s = min(1.0, 1280 / img.width)
            self.size = (round(img.width * s) // 2 * 2, round(img.height * s) // 2 * 2)
            self.stream = self.out.add_stream("h264", rate=self.fps)
            self.stream.width, self.stream.height = self.size
            self.stream.pix_fmt = "yuv420p"
        frame = self.av.VideoFrame.from_image(img.convert("RGB").resize(self.size))
        self.out.mux(self.stream.encode(frame))

    def close(self):
        if self.stream is not None:
            self.out.mux(self.stream.encode())
        self.out.close()


def score_video(path, aois=None, product=None, model="auto", ranges=None, fps=2.0, window=1.0, heat_mp4=None):
    """aois: None, a static list for every frame, or "auto" (re-detected at the start and after each cut).

    Heatmaps are averaged over the last `window` seconds, so attention lingers briefly after a cut
    instead of jumping instantly, as eye-tracking shows. Returns a report dict; writes `heat_mp4` if given.
    """
    from .render import aoi_overlay, heat_overlay
    recent, rows, writer, used = deque(), [], _Mp4(heat_mp4, fps) if heat_mp4 else None, model
    shot_aois = None if aois == "auto" else (aois or [])
    try:
        for t, img, cut in frames(path, fps):
            heat, used = predict(img, model)
            recent.append((t, heat))
            while recent[0][0] < t - window + 1e-6 and len(recent) > 1:
                recent.popleft()
            smooth = np.mean([h for _, h in recent], 0)
            if aois == "auto" and (shot_aois is None or cut is not None):
                from .detect import detect
                shot_aois = detect(img, product=product)  # ponytail: per shot, so moving objects drift
            stats = aoi_stats(smooth, shot_aois)
            rows.append({"t": round(t, 2), "cut": None if cut is None else round(cut, 2), "focus": round(focus(smooth), 1),
                         "cognitive_demand": round(cognitive_demand(np.asarray(img.convert("L"))), 1),
                         "aois": {a["label"]: a["attention_pct"] for a in stats}})
            if writer:
                writer.write(aoi_overlay(img, smooth, stats) if stats else heat_overlay(img, smooth))
    finally:
        if writer:
            writer.close()
    if not rows:
        raise ValueError(f"no frames decoded from {path}")
    return _summarise(rows, fps, used, ranges)


def _summarise(rows, fps, used, ranges):
    mean = lambda xs: round(float(np.mean(xs)), 1) if xs else None
    end = rows[-1]["t"]
    metrics = {"focus": mean([r["focus"] for r in rows]),
               "focus_start": mean([r["focus"] for r in rows if r["t"] < 2]),
               "focus_end": mean([r["focus"] for r in rows if r["t"] > end - 2]),
               "cognitive_demand": mean([r["cognitive_demand"] for r in rows])}
    aois = {}
    for label in dict.fromkeys(k for r in rows for k in r["aois"]):
        seen = [r["aois"][label] for r in rows if label in r["aois"]]
        aois[label] = {"total_attention": round(sum(seen) / len(rows), 1),  # share of attention over the whole video
                       "relative_attention": mean(seen),  # share of attention while on screen
                       "availability_s": round(len(seen) / fps, 1)}
        metrics[f"{label}_attention"] = aois[label]["total_attention"]
    impact, parts = impact_score(metrics, ranges or DEFAULT_RANGES)
    return {"model": used, "duration_s": end, "cuts": [r["cut"] for r in rows if r["cut"] is not None], "metrics": metrics,
            "aois": aois, "impact_score": impact, "impact_parts": parts, "timeline": rows}
