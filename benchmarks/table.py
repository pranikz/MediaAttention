"""Markdown tables from benchmarks/results/<dataset>.json, for the README and the model card.

    python benchmarks/table.py fiwi
"""
import json
import sys
from pathlib import Path

NAMES = {"deepgaze_msdb": "DeepGaze MSDB", "deepgaze_iie": "DeepGaze IIE", "unisal": "**UNISAL (default)**",
         "transalnet": "TranSalNet-Dense", "center": "Centre-bias baseline", "spectral": "Spectral residual (fallback)"}
LICENCE = {"deepgaze_msdb": "no licence file", "deepgaze_iie": "no licence file", "unisal": "Apache-2.0",
           "transalnet": "MIT", "center": "–", "spectral": "–"}


def tables(dataset):
    r = json.loads((Path(__file__).parent / "results" / f"{dataset}.json").read_text())
    keys = [k for k in NAMES if k in r]
    main = ["| Model | CC ↑ | NSS ↑ | sAUC ↑ | AUC ↑ | SIM ↑ | KLD ↓ | s / image (M3 Pro) | Licence |",
            "|---|---|---|---|---|---|---|---|---|"]
    for k in keys:
        a = r[k]["all"]
        main.append(f"| {NAMES[k]} | {a['CC'][0]:.3f} | {a['NSS'][0]:.2f} | {a['sAUC'][0]:.3f} | {a['AUC'][0]:.3f} "
                    f"| {a['SIM'][0]:.3f} | {a['KLD'][0]:.3f} | {r[k]['s_per_image']:.2f} | {LICENCE[k]} |")
    cats = sorted(r[keys[0]]["by_category"])
    by_cat = ["| Model | " + " | ".join(c.capitalize() for c in cats) + " |", "|---" * (len(cats) + 1) + "|"]
    for k in keys:
        if k not in ("center", "spectral"):
            by_cat.append(f"| {NAMES[k]} | " + " | ".join(f"{r[k]['by_category'][c]['CC']:.3f}" for c in cats) + " |")
    return "\n".join(main), "\n".join(by_cat)


if __name__ == "__main__":
    for t in tables(sys.argv[1]):
        print(t, end="\n\n")
