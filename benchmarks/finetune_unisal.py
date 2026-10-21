"""Fine-tune UNISAL on design eye tracking (UEyes train split) -> an ad- and UI-tuned attention model.

    python benchmarks/finetune_unisal.py --epochs 8 --out $MA_BENCH/models/unisal_ueyes.pth
    MA_UNISAL_WEIGHTS=$MA_BENCH/models/unisal_ueyes.pth python benchmarks/run.py ueyes_test --models unisal

Starts from UNISAL's SALICON weights. Loss: KL divergence to the fixation density minus correlation (CC),
the usual pair for saliency training. 10% of the training images are held out to pick the best epoch;
the UEyes test split and FiWI are never touched, so they stay fair for evaluation.
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run  # noqa: E402
from mediaattention.saliency import _unisal, _unisal_input, _unisal_size, device  # noqa: E402

OUT_SCALE = 4  # ground truth and predictions at 1/4 of the input resolution: plenty for a blurred density


def target(item, size):
    """Fixation density at `size` (h, w): fixations blurred with the dataset's own sigma, normalised to sum 1."""
    w, h = Image.open(item["path"]).size
    _, sigma = run.per_image_scale(item, {"ppd_screen": 40.0, "screen": (1920, 1200)})
    gt = run.fixation_map((h, w), np.clip(np.asarray(item["xs"], int), 0, w - 1),
                          np.clip(np.asarray(item["ys"], int), 0, h - 1), sigma)
    gt = np.asarray(Image.fromarray(gt.astype(np.float32)).resize(size[::-1], Image.BILINEAR), dtype=np.float32)
    return torch.tensor(gt / max(gt.sum(), 1e-12))


def loss_fn(logp, gt):
    """KL(gt || pred) - CC(pred, gt), per image."""
    kl = (gt * (torch.log(gt + 1e-12) - logp)).sum()
    p = logp.exp()
    a, b = p - p.mean(), gt - gt.mean()
    cc = (a * b).sum() / (a.norm() * b.norm() + 1e-12)
    return kl - cc, kl.item(), cc.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--limit", type=int, help="first N training images (smoke test)")
    ap.add_argument("--out", default=str(run.BENCH / "models/unisal_ueyes.pth"))
    ap.add_argument("--device", help="cpu, mps or cuda (default: best available)")
    ap.add_argument("--ckpt", help="resume checkpoint, saved after every epoch (default: <out>.ckpt)")
    a = ap.parse_args()
    dev = a.device or device()
    items, _ = run.load_ueyes(split="train")
    for it in items:  # keep fixations inside the image
        w, h = Image.open(it["path"]).size
        keep = [(x, y) for x, y in zip(it["xs"], it["ys"]) if 0 <= x < w and 0 <= y < h]
        it["xs"], it["ys"] = [k[0] for k in keep], [k[1] for k in keep]
    items = [it for it in items if it["xs"]]
    random.Random(0).shuffle(items)
    items = items[:a.limit] if a.limit else items
    n_val = max(1, len(items) // 10)
    val, train = items[:n_val], items[n_val:]
    print(f"train {len(train)}, val {len(val)} images, device {dev}", flush=True)

    model = _unisal(dev)

    def train_mode():
        # batch size is 1: keep batch-norm statistics frozen, or one image at a time would corrupt them
        model.train()
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()

    train_mode()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    cache = {}

    def sample(it):
        if it["path"] not in cache:
            img = Image.open(it["path"]).convert("RGB")
            size = _unisal_size(img.height, img.width)
            out = (size[0] // OUT_SCALE, size[1] // OUT_SCALE)
            cache[it["path"]] = (_unisal_input(img, size), target(it, out), out)
        return cache[it["path"]]

    def evaluate():
        model.eval()
        kls, ccs = [], []
        with torch.no_grad():
            for it in val:
                x, gt, out = sample(it)
                logp = model(x[None, None].to(dev), target_size=out, source="SALICON", static=True)[0, 0, 0]
                _, kl, cc = loss_fn(logp, gt.to(dev))
                kls.append(kl)
                ccs.append(cc)
        train_mode()
        return float(np.mean(kls)), float(np.mean(ccs))

    # sessions can be cut off (Colab free tier): everything needed to continue is saved after each epoch
    ckpt = Path(a.ckpt or a.out + ".ckpt")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if ckpt.exists():
        c = torch.load(ckpt, map_location=dev)
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        start, best = c["epoch"] + 1, tuple(c["best"])
        print(f"resuming after epoch {c['epoch']} (best val CC {best[1]:.3f})", flush=True)
    else:
        start, best = 1, evaluate()
        print(f"epoch 0 (SALICON weights): val KLD {best[0]:.3f}  CC {best[1]:.3f}", flush=True)
        torch.save(model.state_dict(), a.out)
    for epoch in range(start, a.epochs + 1):
        t0 = time.time()
        random.shuffle(train)
        for it in train:
            x, gt, out = sample(it)
            logp = model(x[None, None].to(dev), target_size=out, source="SALICON", static=True)[0, 0, 0]
            loss, _, _ = loss_fn(logp, gt.to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        kl, cc = evaluate()
        better = cc > best[1]
        if better:
            best = (kl, cc)
            torch.save(model.state_dict(), a.out)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "best": best}, ckpt)
        print(f"epoch {epoch}: val KLD {kl:.3f}  CC {cc:.3f}  ({time.time() - t0:.0f} s){'  saved' if better else ''}", flush=True)
    print(f"best val CC {best[1]:.3f} -> {a.out}")


if __name__ == "__main__":
    main()
