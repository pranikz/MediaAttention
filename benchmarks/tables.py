"""Comparison tables for the README, site and report, built from benchmarks/results/.

    python benchmarks/tables.py          # prints the markdown used in README.md
"""
import csv
import json
import math
import re
import statistics as st
from pathlib import Path

R = Path(__file__).parent / "results"
NAMES = {"unisal_ueyes": "**MediaAttention-Ads v1** (ours)", "deepgaze_msdb": "DeepGaze MSDB", "deepgaze_iie": "DeepGaze IIE",
         "transalnet": "TranSalNet-Dense", "unisal": "UNISAL (stock)", "center": "Centre-bias baseline",
         "spectral": "Spectral residual"}
# exact parameter counts, measured by loading each model
ARCH = {
    "unisal_ueyes": ("UNISAL: MobileNetV2 encoder, Gaussian priors, ConvGRU (video), decoder with skips", 3.7,
                     "SALICON → fine-tuned on UEyes (posters, ads, UIs)", "yes (recurrent)", "Apache-2.0"),
    "unisal": ("same as above", 3.7, "SALICON, DHF1K, Hollywood-2, UCF Sports", "yes (recurrent)", "Apache-2.0"),
    "transalnet": ("DenseNet-161 CNN + transformer encoder + CNN decoder", 76.6, "SALICON", "no", "MIT"),
    "deepgaze_iie": ("ensemble of 4 backbones (DenseNet-201, ResNeXt-50, EfficientNet-B5, ShapeNet-ResNet-50) + readout", 104.0,
                     "SALICON → MIT1003", "no", "no licence file"),
    "deepgaze_msdb": ("CLIP RN50x64 + DINOv2 ViT-B/14 at several scales + readout, dataset-bias parameters", 451.6,
                      "MIT1003, CAT2000, COCO-Freeview, DAEMONS, FIGRIM", "no", "no licence file"),
}


def load(name):
    p = R / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def per_image(prefix):
    rows = []
    for f in R.glob(f"{prefix}_per_image_*.csv"):
        rows += [r for r in csv.DictReader(open(f)) if r.get("image") != "__time__"]
    return rows


def ci(values):
    mu = st.mean(values)
    se = st.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0
    return mu, mu - 1.96 * se, mu + 1.96 * se


def posters_table():
    rows = per_image("ueyes_test")
    out = ["| Model | CC ↑ | 95% CI | NSS ↑ | KLD ↓ |", "|---|---|---|---|---|"]
    stats = []
    for m in NAMES:
        p = [r for r in rows if r["model"] == m and r["category"] == "poster"]
        if p:
            mu, lo, hi = ci([float(r["CC"]) for r in p])
            stats.append((mu, m, lo, hi, st.mean(float(r["NSS"]) for r in p), st.mean(float(r["KLD"]) for r in p), len(p)))
    for mu, m, lo, hi, nss, kld, n in sorted(stats, reverse=True):
        out.append(f"| {NAMES[m]} | {mu:.3f} | [{lo:.3f}, {hi:.3f}] | {nss:.2f} | {kld:.3f} |")
    return "\n".join(out), stats[0][-1] if stats else 0


def dataset_table(name, cats=None):
    r = load(name)
    keys = sorted([k for k in NAMES if k in r], key=lambda k: -r[k]["all"]["CC"][0])
    head = "| Model | CC ↑ | NSS ↑ | sAUC ↑ | SIM ↑ | KLD ↓ |" + "".join(f" {c} CC |" for c in cats or [])
    out = [head, "|---" * (6 + len(cats or [])) + "|"]
    for k in keys:
        a = r[k]["all"]
        out.append(f"| {NAMES[k]} | {a['CC'][0]:.3f} | {a['NSS'][0]:.2f} | {a['sAUC'][0]:.3f} | {a['SIM'][0]:.3f} | {a['KLD'][0]:.3f} |"
                   + "".join(f" {r[k]['by_category'][c]['CC']:.3f} |" for c in cats or []))
    return "\n".join(out)


def arch_table():
    fiwi, test = load("fiwi"), load("ueyes_test")
    speed = {k: fiwi[k]["s_per_image"] for k in fiwi}
    speed["unisal_ueyes"] = speed.get("unisal", 0.04)  # same network as stock UNISAL
    out = ["| Model | Architecture | Params | Trained on | Video | Licence | s / image (M3 Pro) | UEyes test CC | FiWI CC |",
           "|---|---|---|---|---|---|---|---|---|"]
    for k, (arch, params, data, video, lic) in ARCH.items():
        out.append(f"| {NAMES[k]} | {arch} | {params:.1f}M | {data} | {video} | {lic} | {speed.get(k, float('nan')):.2f} | "
                   f"{test[k]['all']['CC'][0]:.3f} | {fiwi[k]['all']['CC'][0]:.3f} |")
    out.append("| Neurons (commercial, for reference) | CNN encoder–decoder (ConvNeXt / ResNet-50 components), per its public tech paper "
               "| not disclosed | proprietary eye tracking (~20k participants) | separate video model | closed | – | not runnable | not runnable |")
    return "\n".join(out)


def to_html(md):
    """A markdown table from this module -> an HTML table; our row is marked class="ours"."""
    lines = [l for l in md.strip().splitlines() if not l.startswith("|---")]
    cells = lambda l: [c.strip() for c in l.strip("|").split("|")]
    head = "".join(f"<th>{c}</th>" for c in cells(lines[0]))
    rows = ""
    for l in lines[1:]:
        cs = cells(l)
        ours = cs[0].startswith("**")
        cs[0] = re.sub(r"\*\*(.+?)\*\*", r"\1", cs[0])
        tds = "".join(f'<td{" class=n" if re.match(r"^[0-9.[–-]", c) else ""}>{c}</td>' for c in cs)
        rows += f'<tr{" class=ours" if ours else ""}>{tds}</tr>'
    return f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"


if __name__ == "__main__":
    t, n = posters_table()
    print(t, f"\n(n = {n} test posters)\n")
    print(dataset_table("ueyes_test", ["poster", "mobile", "desktop", "web"]), "\n")
    print(dataset_table("fiwi", ["pictorial", "mixed", "text"]), "\n")
    print(arch_table())
