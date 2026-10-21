"""Attention heatmap prediction. Returns an (H, W) float map at the image's size."""
import contextlib
import sys
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


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


@cache
def _deepgaze(device):
    import deepgaze_pytorch
    with contextlib.redirect_stdout(sys.stderr):  # it prints load messages, keep stdout clean JSON
        return deepgaze_pytorch.DeepGazeIIE(pretrained=True).float().to(device).eval()  # MPS has no float64


def deepgaze(img, dev=None):
    """DeepGaze IIE (Linardos et al. 2021), weights fetched by deepgaze_pytorch on first use."""
    import torch
    dev = dev or device()
    s = min(1.0, 1024 / max(img.size))
    x = np.asarray(img.convert("RGB").resize((round(img.width * s), round(img.height * s))), dtype=np.float32)
    with torch.no_grad():
        logd = _deepgaze(dev)(
            torch.tensor(x.transpose(2, 0, 1)[None], device=dev),
            torch.tensor(center_bias(x.shape[:2])[None], device=dev),
        )
    return _resize(np.exp(logd.cpu().numpy()[0, 0]), img.size)


UNISAL_REPO = "pranikz/MediaAttention"  # UNISAL weights + config, redistributed under Apache-2.0


def _unisal_size(h, w):
    """UNISAL's input size: the 32-px grid (7-13 cells a side, 100-120 cells) closest to the aspect ratio."""
    ar = h / w
    cells = [(a, b) for a in range(7, 14) for b in range(7, 14) if 100 <= a * b <= 120]
    a, b = max(cells, key=lambda c: min(ar, c[0] / c[1]) / max(ar, c[0] / c[1]))
    return a * 32, b * 32


@cache
def _unisal(dev):
    import json

    import torch
    from huggingface_hub import hf_hub_download

    from .unisal.model import UNISAL
    cfg = json.loads(open(hf_hub_download(UNISAL_REPO, "unisal_config.json")).read())
    cfg["cnn_cfg"]["pretrained"] = False  # the full state dict below includes the backbone
    cfg["verbose"] = 0
    m = UNISAL(**cfg)
    m.load_state_dict(torch.load(hf_hub_download(UNISAL_REPO, "unisal_salicon.pth"), map_location="cpu"))
    return m.to(dev).eval()


def _unisal_input(img, size):
    """(3, H, W) tensor: LANCZOS resize to UNISAL's input size, ImageNet normalisation."""
    import torch
    x = np.asarray(img.convert("RGB").resize(size[::-1], Image.LANCZOS), dtype=np.float32) / 255
    x = (x - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
    return torch.tensor(x.transpose(2, 0, 1), dtype=torch.float32)


def unisal(img, dev=None):
    """UNISAL (Droste et al., ECCV 2020), SALICON weights: the default. Apache-2.0, ~4M params."""
    import torch
    dev = dev or device()
    h, w = img.height, img.width
    x = _unisal_input(img, _unisal_size(h, w))[None, None].to(dev)
    with torch.no_grad():
        logd = _unisal(dev)(x, target_size=(h, w), source="SALICON", static=True)
    return np.exp(logd[0, 0, 0].float().cpu().numpy())


class UnisalVideo:
    """UNISAL's video mode (DHF1K domain): a recurrent state carries attention from frame to frame.

    Call with the frames since the previous call, in time order, at about FPS (UNISAL saw DHF1K's
    30 fps video at every 5th frame). Returns the attention map for the last frame.
    """
    FPS = 6.0

    def __init__(self, size, dev=None):
        self.dev, self.size, self.h0 = dev or device(), size, [None]
        self.input_size = _unisal_size(size[1], size[0])

    def __call__(self, frames):
        import torch
        x = torch.stack([_unisal_input(f, self.input_size) for f in frames])[None].to(self.dev)
        with torch.no_grad():
            logd, self.h0 = _unisal(self.dev)(x, target_size=self.size[::-1], h0=self.h0, return_hidden=True,
                                              source="DHF1K", static=False)
        return np.exp(logd[0, -1, 0].float().cpu().numpy())


MODELS = {"unisal": unisal, "deepgaze": deepgaze, "spectral": spectral_residual}


def resolve(model="auto"):
    """auto = UNISAL if torch is installed, else the spectral fallback."""
    if model != "auto":
        return model
    try:
        import huggingface_hub  # noqa: F401
        import torch  # noqa: F401
        return "unisal"
    except ImportError:
        return "spectral"


def predict(img, model="auto"):
    """Returns (heatmap, model_used)."""
    model = resolve(model)
    return MODELS[model](img), model
