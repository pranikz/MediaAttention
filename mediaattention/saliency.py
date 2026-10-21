"""Attention heatmap prediction. Returns an (H, W) float map at the image's size."""
from functools import cache

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.special import logsumexp


def _resize(a, size):
    return np.asarray(Image.fromarray(a.astype(np.float32)).resize(size, Image.BILINEAR))


def spectral_residual(img):
    """Hou & Zhang 2007. Zero-dependency fallback: fast, crude, no learned weights."""
    w = 64
    small = np.asarray(img.convert("L").resize((w, max(1, round(w * img.height / img.width)))), dtype=np.float64)
    f = np.fft.fft2(small)
    log_amp = np.log(np.abs(f) + 1e-9)
    sal = np.abs(np.fft.ifft2(np.exp(log_amp - uniform_filter(log_amp, 3) + 1j * np.angle(f)))) ** 2
    return _resize(gaussian_filter(sal, 3), img.size)


def center_bias(shape, sigma=0.3):
    """Gaussian log-density prior; people look at the centre first."""
    h, w = shape
    y, x = np.mgrid[0:h, 0:w]
    cb = -(((x - w / 2) / (sigma * w)) ** 2 + ((y - h / 2) / (sigma * h)) ** 2) / 2
    return (cb - logsumexp(cb)).astype(np.float32)


@cache
def _deepgaze(device):
    import deepgaze_pytorch
    return deepgaze_pytorch.DeepGazeIIE(pretrained=True).float().to(device).eval()  # MPS has no float64


def deepgaze(img, device=None):
    """DeepGaze IIE (Linardos et al. 2021), weights fetched by deepgaze_pytorch on first use."""
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    s = min(1.0, 1024 / max(img.size))
    x = np.asarray(img.convert("RGB").resize((round(img.width * s), round(img.height * s))), dtype=np.float32)
    with torch.no_grad():
        logd = _deepgaze(device)(
            torch.tensor(x.transpose(2, 0, 1)[None], device=device),
            torch.tensor(center_bias(x.shape[:2])[None], device=device),
        )
    return _resize(np.exp(logd.cpu().numpy()[0, 0]), img.size)


def predict(img, model="auto"):
    """Returns (heatmap, model_used)."""
    if model == "auto":
        try:
            import deepgaze_pytorch  # noqa: F401
            model = "deepgaze"
        except ImportError:
            model = "spectral"
    return {"deepgaze": deepgaze, "spectral": spectral_residual}[model](img), model
