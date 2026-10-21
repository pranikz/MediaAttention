"""Video scoring: sample frames, smooth attention over time, track cuts and AOIs. Needs the `video` extra."""
from collections import deque
from fractions import Fraction

import numpy as np

from . import memory as _memory
from .metrics import DEFAULT_RANGES, aoi_stats, cognitive_demand, focus, impact_score
from .saliency import UnisalVideo, predict, resolve

VIDEO_EXT = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
CUT = 0.4  # histogram change (0-1) between consecutive frames that counts as a hard cut
MIN_SHOT = 0.5  # seconds; ignores flashes and encoder flicker right after a cut


TRACK_W = 320  # width of the grayscale frames used for cut detection and AOI tracking


def frames(path, fps=2.0, clip_fps=None):
    """Yields (t, PIL image, cut, grays, clip) sampled at `fps`.

    cut = time of a hard cut since the previous sample, else None. grays = every decoded frame since the
    previous sample (inclusive) as small grayscale arrays, for tracking. Cuts are checked on every frame.
    clip = RGB frames since the previous sample at ~clip_fps, ending with this sample (empty if not asked).
    """
    import av
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        size = (TRACK_W, max(2, round(TRACK_W * s.height / s.width) // 2 * 2))
        next_t, prev, cut, last_cut, grays, clip, next_c = None, None, None, -MIN_SHOT, [], [], 0.0
        for f in c.decode(s):
            t = float(f.time or 0)
            gray = f.reformat(width=size[0], height=size[1], format="gray").to_ndarray()
            hist = np.bincount(gray.ravel() // 8, minlength=32) / gray.size
            if prev is not None and np.abs(hist - prev).sum() / 2 > CUT and t - last_cut >= MIN_SHOT:
                cut = last_cut = t
            prev = hist
            grays.append(gray)
            sample = next_t is None or t + 1e-6 >= next_t
            if clip_fps and (sample or t + 1e-6 >= next_c):
                clip.append(f.to_image())
                next_c = t + 1 / clip_fps
            if sample:
                img = clip[-1] if clip_fps else f.to_image()
                yield t, img, cut, grays, clip
                cut, next_t, grays, clip = None, (next_t or t) + 1 / fps, [gray], []


def track_boxes(aois, grays):
    """Moves AOIs (0-1 boxes) through consecutive grayscale frames with Lucas-Kanade optical flow.

    Each box gets its own shift + uniform scale (camera zooms/pans and objects moving on their own).
    A box with too few trackable corners stays put; boxes that leave the frame are dropped.
    """
    import cv2
    H, W = grays[0].shape
    boxes = [[a["box"][0] * W, a["box"][1] * H, a["box"][2] * W, a["box"][3] * H] for a in aois]
    for prev, cur in zip(grays, grays[1:]):
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            mask = np.zeros_like(prev)
            mask[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)] = 255
            pts = cv2.goodFeaturesToTrack(prev, 40, 0.01, 3, mask=mask)
            if pts is None or len(pts) < 4:
                continue
            nxt, ok, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None)
            ok = ok.ravel() == 1
            m = cv2.estimateAffinePartial2D(pts[ok], nxt[ok])[0] if ok.sum() >= 4 else None
            if m is not None:
                c = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]]) @ m[:, :2].T + m[:, 2]
                boxes[i] = [c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max()]
    out = []
    for a, (x0, y0, x1, y1) in zip(aois, boxes):
        box = [max(0, x0 / W), max(0, y0 / H), min(1, x1 / W), min(1, y1 / H)]
        if box[2] - box[0] > 0.01 and box[3] - box[1] > 0.01:
            out.append({**a, "box": box})
    return out


def _per_label(stats):
    out = {}
    for a in stats:  # several boxes per role are summed
        out[a["label"]] = round(out.get(a["label"], 0) + a["attention_pct"], 1)
    return out


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


def score_video(path, aois=None, product=None, model="auto", ranges=None, fps=2.0, window=1.0, heat_mp4=None,
                per_frame=False):
    """aois: None, a static list for every frame, or "auto" (detected at each shot start, then tracked).

    With UNISAL (the default) attention comes from its recurrent video mode, fed at ~6 fps, so it carries
    over between frames as in real viewing. Image models (or per_frame=True) score each sampled frame
    and average heatmaps over the last `window` seconds instead. Returns a report dict; writes
    `heat_mp4` if given.
    """
    model = resolve(model)
    temporal = model == "unisal" and not per_frame
    from .render import aoi_overlay, heat_overlay
    recent, rows, writer, used = deque(), [], _Mp4(heat_mp4, fps) if heat_mp4 else None, model
    shot_aois = None if aois == "auto" else (aois or [])
    with_memory = _memory.available()
    try:
        video_model = None
        for t, img, cut, grays, clip in frames(path, fps, UnisalVideo.FPS if temporal else None):
            if temporal:
                video_model = video_model or UnisalVideo(img.size)
                heat, used = video_model(clip), "unisal-video"
            else:
                heat, used = predict(img, model)
            recent.append((t, heat))
            while recent[0][0] < t - (0 if temporal else window) + 1e-6 and len(recent) > 1:
                recent.popleft()
            smooth = np.mean([h for _, h in recent], 0)
            if aois == "auto" and (shot_aois is None or cut is not None):
                from .detect import detect
                shot_aois = detect(img, product=product)  # once per shot, then tracked
            elif aois == "auto" and shot_aois:
                shot_aois = track_boxes(shot_aois, grays)
            stats = aoi_stats(smooth, shot_aois)
            rows.append({"t": round(t, 2), "cut": None if cut is None else round(cut, 2), "focus": round(focus(smooth), 1),
                         "cognitive_demand": round(cognitive_demand(np.asarray(img.convert("L"))), 1),
                         "aois": _per_label(stats)})
            if with_memory:  # per frame, like Neurons: no audio or temporal context
                rows[-1]["memory"] = _memory.memory(img)
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
    if "memory" in rows[0]:
        metrics.update(memory=mean([r["memory"] for r in rows]),
                       memory_start=mean([r["memory"] for r in rows if r["t"] < 2]),
                       memory_end=mean([r["memory"] for r in rows if r["t"] > end - 2]))
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
