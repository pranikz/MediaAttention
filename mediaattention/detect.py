"""AOI detection: OCR + layout rules for text roles, OWLv2 for logo and product. Needs the `aoi` extra."""
import re
from functools import cache

import numpy as np

BRAND_PROMPTS = ["a logo", "a brand logo"]
# words too generic to identify a brand when they repeat across an ad
STOP = set("""THE AND FOR WITH YOUR YOU OUR ARE THIS THAT FROM NEW NOW ALL GET MORE FREE BEST ONLY ONE
CAN WILL HAS HAVE NOT BUT ITS WAS WHO OFF SALE SHOP BUY ORDER CALL TODAY HERE EVERY MADE AT
""".split())


@cache
def _owl():
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    name = "google/owlv2-base-patch16-ensemble"
    return Owlv2Processor.from_pretrained(name), Owlv2ForObjectDetection.from_pretrained(name).eval()


@cache
def _ocr():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


def _overlap(a, b):
    """Intersection over the smaller box: ~1 when one box sits inside the other."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / small if small > 0 else 0.0


def _union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _button(rgb, box, bg):
    """If the text sits on a uniform patch unlike the page background, return the patch's box, else None."""
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = map(int, box)
    pad = max(2, (y1 - y0) // 4)  # thin ring: tight buttons leave little margin around the text
    strips = [rgb[max(0, y0 - pad):y0, x0:x1], rgb[y1:min(H, y1 + pad), x0:x1],
              rgb[y0:y1, max(0, x0 - pad):x0], rgb[y0:y1, x1:min(W, x1 + pad)]]
    ring = np.concatenate([s.reshape(-1, 3) for s in strips if s.size]).astype(float)
    colour = np.median(ring, 0)
    if ring.std(0).mean() >= 12 or np.linalg.norm(colour - bg) <= 40:
        return None
    # grow each side while the next row/column (middle half) is still the button colour
    same = lambda px: np.linalg.norm(np.median(px.reshape(-1, 3), 0) - colour) < 30
    mx, my = (x1 - x0) // 4, (y1 - y0) // 4
    while y0 > 0 and same(rgb[y0 - 1, x0 + mx:x1 - mx]): y0 -= 1
    while y1 < H - 1 and same(rgb[y1 + 1, x0 + mx:x1 - mx]): y1 += 1
    while x0 > 0 and same(rgb[y0 + my:y1 - my, x0 - 1]): x0 -= 1
    while x1 < W - 1 and same(rgb[y0 + my:y1 - my, x1 + 1]): x1 += 1
    return [x0, y0, x1, y1]


def blocks(boxes):
    """Groups text-line boxes into layout blocks: lines of similar size, stacked close, sharing a column."""
    out = []
    for b in sorted(boxes, key=lambda b: (b[1], b[0])):
        h = b[3] - b[1]
        for blk in out:
            bottom, left, right = max(x[3] for x in blk), min(x[0] for x in blk), max(x[2] for x in blk)
            last_h = blk[-1][3] - blk[-1][1]
            if -h <= b[1] - bottom < 1.2 * h and min(b[2], right) > max(b[0], left) and 0.5 < h / last_h < 2:
                blk.append(b)
                break
        else:
            out.append([b])
    return [_union(blk) for blk in out]


def text_roles(lines, rgb):
    """lines: [([x0, y0, x1, y1], text)] in pixels -> (cta, headline_blocks, other_lines).

    CTA: a short line, smaller than the headline type, on a button-sized patch (<= 10% of the ad) unlike the page.
    Headline: lines within 75% of the largest text height, grouped into blocks (several boxes if
    the ad has headlines in different places). ponytail: layout heuristics; swap for a layout model
    if ads break them.
    """
    if not lines:
        return [], [], []
    H, W = rgb.shape[:2]
    edge = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]]).astype(float)
    bg = np.median(edge, 0)
    hmax = max(b[3] - b[1] for b, _ in lines)
    cta, rest = [], []
    for b, t in lines:
        patch = not cta and len(t.split()) <= 4 and b[3] - b[1] < 0.75 * hmax and _button(rgb, b, bg)
        if patch and (patch[2] - patch[0]) * (patch[3] - patch[1]) <= 0.1 * W * H:  # a button, not a colour band
            cta.append(patch)
        else:
            rest.append((b, t))
    hmax = max((b[3] - b[1] for b, _ in rest), default=0)
    big = [b for b, _ in rest if b[3] - b[1] >= 0.75 * hmax]
    return cta, blocks(big), [(b, t) for b, t in rest if b not in big]


def _words(text):
    return {w for w in re.findall(r"[A-Z][A-Z'&-]{2,}", text.upper()) if w not in STOP}


def brand_line(lines, height):
    """A wordmark from the text alone: a short line whose words recur elsewhere in the ad (brand names
    repeat in the copy), else a short line in the top or bottom band set near body size or larger. None if
    nothing qualifies: no brand box is better than a wrong one."""
    best = None
    for i, (b, t) in enumerate(lines):
        ws = _words(t)
        if not ws or len(t.split()) > 4:
            continue
        reps = sum(1 for j, (_, t2) in enumerate(lines) if j != i and ws & _words(t2))
        if reps and (best is None or (reps, b[3] - b[1]) > best[0]):
            best = ((reps, b[3] - b[1]), b)
    if best:
        return best[1]
    body_h = np.median([b[3] - b[1] for b, _ in lines]) if lines else 0
    band = [b for b, t in lines if len(t.split()) <= 2 and _words(t) and b[3] - b[1] >= 0.8 * body_h
            and (b[3] < 0.2 * height or b[1] > 0.8 * height)]
    return band[0] if band else None


def _owl_pick(img, roles, avoid, threshold):
    """roles: {label: [prompts]} -> {label: (box, score)}: best box above threshold not overlapping `avoid`."""
    import torch
    proc, model = _owl()
    prompts = [p for ps in roles.values() for p in ps]
    with torch.no_grad():
        out = model(**proc(text=[prompts], images=img, return_tensors="pt"))
    logits, side = out.logits[0].sigmoid(), max(img.size)  # boxes are relative to the padded square
    cx, cy, w, h = (out.pred_boxes[0] * side).unbind(-1)
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1).tolist()
    picked, i = {}, 0
    for label, ps in roles.items():
        scores = logits[:, i:i + len(ps)].max(1).values
        i += len(ps)
        taken = avoid + [b for b, _ in picked.values()]
        for j in scores.argsort(descending=True).tolist():
            if scores[j] < threshold:
                break
            if all(_overlap(boxes[j], a) < 0.5 for a in taken):
                picked[label] = (boxes[j], float(scores[j]))
                break
    return picked


def detect(img, product=None, threshold=0.1):
    """Returns [{"label", "box" (0-1 coords)}] for brand, product, headline, body, cta."""
    img = img.convert("RGB")
    rgb = np.asarray(img)
    res, _ = _ocr()(rgb)
    lines = [([min(p[0] for p in q), min(p[1] for p in q), max(p[0] for p in q), max(p[1] for p in q)], t)
             for q, t, _ in res or []]
    cta, headline, other = text_roles(lines, rgb)
    owl = _owl_pick(img, {"product": [product or "a product"], "brand": BRAND_PROMPTS}, cta + headline, threshold)
    brand, score = owl.get("brand", (None, 0))
    if score < 0.3:  # OWLv2 is unsure: find the wordmark in the text. ponytail: text only, icons are missed
        brand = brand_line(lines, rgb.shape[0])
    body = blocks([b for b, _ in other if not brand or _overlap(b, brand) < 0.5])
    found = [("cta", b) for b in cta[:1]] + [("headline", b) for b in headline] + [("body", b) for b in body]
    found += [(k, b) for k, b in (("product", owl.get("product", (None,))[0]), ("brand", brand)) if b]
    w, h = img.size
    return [{"label": k, "box": [max(0, b[0]) / w, max(0, b[1]) / h, min(w, b[2]) / w, min(h, b[3]) / h]}
            for k, b in found]
