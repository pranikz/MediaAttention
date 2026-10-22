"""Builds the README images in this folder from docs/img and benchmarks/results.

    python docs/img/readme/make.py
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
IMG = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
import tables  # noqa: E402

INK, RAISED, PAPER, MUTED, LINE, ACCENT = "#08090b", "#101216", "#ececef", "#8b8e98", "#1d1f25", "#5b8cff"
COND, BODY = "/System/Library/Fonts/Avenir Next Condensed.ttc", "/System/Library/Fonts/Avenir Next.ttc"
HEAVY = lambda s: ImageFont.truetype(COND, s, index=8)
MED = lambda s: ImageFont.truetype(BODY, s, index=5)
REG = lambda s: ImageFont.truetype(BODY, s, index=7)
DEMI = lambda s: ImageFont.truetype(BODY, s, index=2)


def fit(im, h):
    im = im.convert("RGB")
    return im.resize((round(im.width * h / im.height), h), Image.LANCZOS)


def hero():
    W, H = 1600, 680
    s = Image.new("RGB", (W, H), INK)
    d = ImageDraw.Draw(s)
    for x in range(0, W, 40):  # faint grid, as on the site
        d.line([(x, 0), (x, H)], fill="#0e1014")
    for y in range(0, H, 40):
        d.line([(0, y), (W, y)], fill="#0e1014")
    a, b = fit(Image.open(IMG / "real/shein-in.jpg"), 540), fit(Image.open(IMG / "real/shein-heat.jpg"), 540)
    x0 = W - 60 - a.width - b.width - 16
    s.paste(a, (x0, 60))
    s.paste(b, (x0 + a.width + 16, 60))
    d.text((x0, 612), "input", fill=MUTED, font=MED(18))
    d.text((x0 + a.width + 16, 612), "MediaAttention-Ads v1 · predicted attention", fill=MUTED, font=MED(18))
    d.text((70, 70), "OPEN SOURCE · APACHE-2.0", fill=MUTED, font=DEMI(18))
    d.text((64, 104), "MEDIA", fill=PAPER, font=HEAVY(112))
    d.text((64, 206), "ATTENTION", fill=PAPER, font=HEAVY(112))
    d.text((70, 370), "See your ad the way people will,", fill=PAPER, font=MED(30))
    d.text((70, 410), "before a single impression is bought.", fill=PAPER, font=MED(30))
    for i, (n, t) in enumerate([("+52%", "vs best open model\non held-out posters"), ("3.7M", "parameters"), ("0.04 s", "per image\non a laptop")]):
        x = 70 + i * 205
        d.text((x, 490), n, fill=ACCENT, font=HEAVY(54))
        d.multiline_text((x, 560), t, fill=MUTED, font=REG(18), spacing=4)
    s.save(HERE / "hero.jpg", quality=88, optimize=True)


def posters():
    rows = sorted((r for r in tables_rows()), reverse=True)
    W, top, rh = 1600, 150, 74
    H = top + rh * len(rows) + 90
    s = Image.new("RGB", (W, H), INK)
    d = ImageDraw.Draw(s)
    d.text((60, 40), "How closely each model matches real eye tracking on posters", fill=PAPER, font=DEMI(34))
    d.text((60, 92), "UEyes test split, 27 posters no model trained on · correlation (CC) · bars show 95% confidence intervals",
           fill=MUTED, font=REG(20))
    L, R = 430, W - 160
    sx = lambda v: L + (R - L) * v / 0.85
    for v in (0.2, 0.4, 0.6, 0.8):
        d.line([(sx(v), top - 10), (sx(v), top + rh * len(rows))], fill=LINE, width=2)
        d.text((sx(v) - 16, top + rh * len(rows) + 14), f"{v:.1f}", fill=MUTED, font=REG(18))
    for i, (cc, key, lo, hi) in enumerate(rows):
        y = top + i * rh
        ours = key == "unisal_ueyes"
        col = ACCENT if ours else "#3a3d46"
        name = "MediaAttention-Ads v1 (ours)" if ours else tables.NAMES[key]
        d.text((60, y + 14), name, fill=PAPER if ours else MUTED, font=(DEMI if ours else MED)(24))
        d.rounded_rectangle([L, y + 12, sx(cc), y + 48], radius=4, fill=col)
        d.line([(sx(lo), y + 30), (sx(hi), y + 30)], fill=PAPER if ours else "#9a9da6", width=3)
        for v in (lo, hi):
            d.line([(sx(v), y + 22), (sx(v), y + 38)], fill=PAPER if ours else "#9a9da6", width=3)
        d.text((sx(hi) + 14, y + 13), f"{cc:.3f}", fill=PAPER if ours else MUTED, font=(DEMI if ours else MED)(24))
    s.save(HERE / "posters.png", optimize=True)


def tables_rows():
    rs = tables.per_image("ueyes_test")
    for k in tables.NAMES:
        v = [float(r["CC"]) for r in rs if r["model"] == k and r["category"] == "poster"]
        if v:
            mu, lo, hi = tables.ci(v)
            yield mu, k, lo, hi


def strip(items, h, out, label_h=56):
    ims = [(t, fit(Image.open(p), h)) for t, p in items]
    gap, pad = 16, 40
    W = sum(i.width for _, i in ims) + gap * (len(ims) - 1) + pad * 2
    s = Image.new("RGB", (W, h + pad * 2 + label_h), INK)
    d = ImageDraw.Draw(s)
    x = pad
    for t, im in ims:
        s.paste(im, (x, pad))
        d.text((x, pad + h + 16), t, fill=MUTED, font=MED(22))
        x += im.width + gap
    s.save(HERE / out, quality=88, optimize=True)


def gallery():
    ads = [("shein", "SHEIN"), ("eagle", "EAGLE"), ("bershka", "Bershka × Myntra"), ("desidresscode", "Desi Dress Code"), ("boat", "boAt Slazer")]
    scores = {"shein": 8.1, "eagle": 7.2, "bershka": 7.7, "desidresscode": 6.4, "boat": 5.5}
    h, gap, pad = 420, 14, 40
    cols = [(fit(Image.open(IMG / f"real/{k}-in.jpg"), h), fit(Image.open(IMG / f"real/{k}-heat.jpg"), h)) for k, _ in ads]
    W = sum(a.width for a, _ in cols) + gap * (len(cols) - 1) + pad * 2
    H = pad + h * 2 + gap + 100
    s = Image.new("RGB", (W, H), INK)
    d = ImageDraw.Draw(s)
    x = pad
    for (a, b), (k, name) in zip(cols, ads):
        s.paste(a, (x, pad))
        s.paste(b, (x, pad + h + gap))
        y = pad + 2 * h + gap + 18
        d.text((x, y), name, fill=PAPER, font=DEMI(22))
        d.text((x, y + 32), f"impact {scores[k]}/10", fill=ACCENT, font=MED(20))
        x += a.width + gap
    s.save(HERE / "real-ads.jpg", quality=86, optimize=True)


if __name__ == "__main__":
    hero()
    posters()
    strip([("1 · your ad", IMG / "ad.jpg"), ("2 · predicted attention", IMG / "ad-heat.jpg"),
           ("3 · fog: what viewers miss", IMG / "ad-fog.jpg"), ("4 · brand, headline, CTA scored", IMG / "ad-aois.jpg")], 380, "pipeline.jpg")
    gallery()
    for f in sorted(HERE.glob("*.[jp][pn]g")):
        print(f.name, Image.open(f).size, f.stat().st_size // 1024, "KB")
