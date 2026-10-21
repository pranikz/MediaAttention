import numpy as np
from PIL import Image


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
