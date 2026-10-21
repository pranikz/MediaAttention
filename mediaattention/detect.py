"""AOI detection: OCR + layout rules for text roles, OWLv2 for logo and product. Needs the `aoi` extra."""
from functools import cache

import numpy as np

BRAND_PROMPTS = ["a logo", "a brand logo"]


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


def text_roles(lines, rgb):
    """lines: [([x0, y0, x1, y1], text)] in pixels -> (cta, headline, other_lines).

    ponytail: layout heuristics (size, button patch, word count); swap for a layout model if ads break them.
    """
    if not lines:
        return [], [], []
    edge = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]]).astype(float)
    bg = np.median(edge, 0)
    cta, rest = [], []
    for b, t in lines:
        patch = not cta and len(t.split()) <= 4 and _button(rgb, b, bg)
        cta.append(patch) if patch else rest.append((b, t))
    hmax = max((b[3] - b[1] for b, _ in rest), default=0)
    headline = [b for b, _ in rest if b[3] - b[1] >= 0.75 * hmax]
    other = [(b, t) for b, t in rest if b not in headline]
    return cta, headline, other


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
    found = {"cta": cta[0] if cta else None, "headline": _union(headline) if headline else None}
    owl = _owl_pick(img, {"product": [product or "a product"], "brand": BRAND_PROMPTS},
                    [b for b in found.values() if b], threshold)
    found["product"] = owl.get("product", (None,))[0]
    brand, score = owl.get("brand", (None, 0))
    if score < 0.3:
        # OWLv2 is unsure: take a short text line in the top/bottom band, where wordmarks usually sit.
        # ponytail: text box only, any icon next to the wordmark is missed.
        H = rgb.shape[0]
        marks = [b for b, t in other if len(t.split()) <= 2 and (b[3] < 0.2 * H or b[1] > 0.8 * H)]
        brand = marks[0] if marks else brand
    found["brand"] = brand
    # body = remaining text not part of the logo. ponytail: one box over all body lines.
    body = [b for b, _ in other if not brand or _overlap(b, brand) < 0.5]
    found["body"] = _union(body) if body else None
    w, h = img.size
    return [{"label": k, "box": [max(0, b[0]) / w, max(0, b[1]) / h, min(w, b[2]) / w, min(h, b[3]) / h]}
            for k, b in found.items() if b]
