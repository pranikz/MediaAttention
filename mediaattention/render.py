import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _unit(heat):
    m = heat.max()
    return heat / m if m > 0 else heat


def heat_overlay(img, heat):
    """Green -> yellow -> red, transparent where nobody looks."""
    v = _unit(heat)
    rgba = np.dstack([np.clip(2 * v, 0, 1), np.clip(2 - 2 * v, 0, 1), np.zeros_like(v), np.clip(1.5 * v, 0, 0.75)])
    return Image.alpha_composite(img.convert("RGBA"), Image.fromarray((rgba * 255).astype(np.uint8)))


def fog_overlay(img, heat):
    """White fog that clears where attention lands."""
    v = _unit(heat)
    fog = np.dstack([np.ones_like(v)] * 3 + [(1 - np.sqrt(v)) * 0.92])
    return Image.alpha_composite(img.convert("RGBA"), Image.fromarray((fog * 255).astype(np.uint8)))


def aoi_overlay(img, heat, aois):
    """Heatmap with each AOI boxed and labelled with its share of attention."""
    out = heat_overlay(img, heat)
    d, (w, h) = ImageDraw.Draw(out), out.size
    font = ImageFont.load_default(size=max(14, w // 50))
    for a in aois:
        x0, y0, x1, y1 = a["box"][0] * w, a["box"][1] * h, a["box"][2] * w, a["box"][3] * h
        d.rectangle((x0, y0, x1, y1), outline=(0, 120, 255, 255), width=max(2, w // 300))
        d.text((x0 + 4, y0 + 2), f"{a['label']} {a['attention_pct']}%", font=font, fill=(0, 120, 255, 255),
               stroke_width=2, stroke_fill=(255, 255, 255, 255))
    return out
