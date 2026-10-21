"""Zero-shot AOI detection with OWLv2 (Apache-2.0). Needs the `aoi` extra."""
from functools import cache

PROMPTS = {"brand": "a logo", "product": "a product", "headline": "a headline text", "cta": "a call to action button"}


@cache
def _owl():
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    name = "google/owlv2-base-patch16-ensemble"
    return Owlv2Processor.from_pretrained(name), Owlv2ForObjectDetection.from_pretrained(name).eval()


def detect(img, threshold=0.2):
    """Returns [{"label", "box" (0-1 coords), "score"}]. ponytail: best box per label only."""
    import torch
    proc, model = _owl()
    img = img.convert("RGB")
    with torch.no_grad():
        out = model(**proc(text=[list(PROMPTS.values())], images=img, return_tensors="pt"))
    side = max(img.size)  # OWLv2 pads to a square, so boxes are in padded-square pixels
    res = proc.post_process_grounded_object_detection(out, threshold=threshold, target_sizes=torch.tensor([[side, side]]))[0]
    names, best = list(PROMPTS), {}
    for score, label, box in zip(res["scores"].tolist(), res["labels"].tolist(), res["boxes"].tolist()):
        if score > best.get(names[label], (0,))[0]:
            best[names[label]] = (score, box)
    w, h = img.size
    return [{"label": n, "box": [max(0, b[0]) / w, max(0, b[1]) / h, min(w, b[2]) / w, min(h, b[3]) / h],
             "score": round(s, 3)} for n, (s, b) in best.items()]
