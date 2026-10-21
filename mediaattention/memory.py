"""Memorability with ResMem (Needell & Bainbridge 2022). Needs the `memory` extra.

ResMem's weights are under a NON-COMMERCIAL license (University of Chicago); check it before commercial use.
"""
from functools import cache

from .saliency import device as _device


def available():
    try:
        import resmem  # noqa: F401
        return True
    except ImportError:
        return False


@cache
def _resmem(device):
    from resmem import ResMem
    return ResMem(pretrained=True).to(device).eval()


def memory(img, device=None):
    """0-100: predicted share of viewers who would recognise the image after a brief exposure (LaMem hit rate)."""
    import torch
    from resmem import transformer
    device = device or _device()
    x = transformer(img.convert("RGB")).view(-1, 3, 227, 227).to(device)
    with torch.no_grad():
        return round(100 * float(_resmem(device)(x)), 1)
