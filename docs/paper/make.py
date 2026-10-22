"""Builds the arXiv paper: figures (PDF/JPG) and result tables (LaTeX) from benchmarks/results, then compiles.

    python docs/paper/make.py            # writes docs/paper/{figs,tables}/, main.pdf and arxiv.tar.gz

Needs Google Chrome (SVG -> PDF) and tectonic (brew install tectonic). The .tex stays ASCII so that arXiv's
pdflatex builds it too; the .bbl is shipped in the tarball because arXiv does not run BibTeX.
"""
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "docs/research"), str(ROOT / "benchmarks")]
import build as report  # noqa: E402  (the HTML report's figure code)
import tables  # noqa: E402

FIGS, TABS = HERE / "figs", HERE / "tables"
TEX = {"→": r"$\rightarrow$", "×": r"$\times$", "≈": r"$\approx$", "↑": r"$\uparrow$", "↓": r"$\downarrow$",
       "–": "--", "—": "---", "·": r"$\cdot$", "~": r"$\sim$", "%": r"\%", "&": r"\&", "_": r"\_", "#": r"\#"}


def tex(s):
    s = "".join(TEX.get(c, c) for c in s)
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    assert s.isascii(), s
    return s


def latex_table(md, spec, bold_ours=True):
    lines = [l for l in md.strip().splitlines() if not l.startswith("|---")]
    cells = lambda l: [c.strip() for c in l.strip("|").split("|")]
    out = [r"\begin{tabular}{" + spec + "}", r"\toprule", " & ".join(tex(c) for c in cells(lines[0])) + r" \\", r"\midrule"]
    for l in lines[1:]:
        cs = cells(l)
        ours = cs[0].startswith("**")
        row = [tex(c) for c in cs]
        if ours and bold_ours:
            row = [r"\textbf{" + re.sub(r"\\textbf\{(.+?)\}", r"\1", c) + "}" for c in row]
        out.append(" & ".join(row) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def svg_to_pdf(svg, name, width=760):
    """Chrome prints an HTML page sized to the SVG; the PDF keeps it as vectors."""
    m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
    w, h = float(m.group(1)), float(m.group(2))
    hgt = width * h / w
    html = FIGS / f"{name}.html"
    html.write_text(f"<!doctype html><meta charset=utf-8><style>@page{{size:{width}px {hgt + 2}px;margin:0}}"
                    f"body{{margin:0;font-family:Helvetica,Arial,sans-serif}}svg{{display:block;width:{width}px;height:{hgt}px}}</style>"
                    + svg.replace('font-family="Archivo, sans-serif"', 'font-family="Helvetica, Arial, sans-serif"'))
    subprocess.run([report.CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={FIGS / (name + '.pdf')}", html.as_uri()], check=True, capture_output=True)
    html.unlink()


def main():
    FIGS.mkdir(exist_ok=True)
    TABS.mkdir(exist_ok=True)
    fiwi = report.load("fiwi")
    svg_to_pdf(report.pipeline_svg(), "pipeline")
    svg_to_pdf(report.model_svg(), "model")
    svg_to_pdf(report.curve_svg(), "curve")
    svg_to_pdf(report.bars(fiwi), "fiwi_bars")
    svg_to_pdf(report.scatter(fiwi), "fiwi_speed")
    shutil.copy(ROOT / "docs/research/fig/models_on_ad.jpg", FIGS / "models_on_ad.jpg")

    cats = "l" + "r" * 9
    (TABS / "posters.tex").write_text(latex_table(tables.posters_table()[0], "lrcrr"))
    (TABS / "ueyes_test.tex").write_text(latex_table(tables.BLOCKS["ueyes_test"](), cats))
    (TABS / "ueyes_full.tex").write_text(latex_table(tables.BLOCKS["ueyes_full"](), cats))
    (TABS / "fiwi.tex").write_text(latex_table(tables.BLOCKS["fiwi"](), "l" + "r" * 8))
    rows = [l for l in tables.arch_table().splitlines() if not l.startswith("|---")]
    keep = [0, 2, 4, 5, 6, 7, 8]  # model, params, video, licence, speed, UEyes CC, FiWI CC (architecture is in the text)
    arch = "\n".join("| " + " | ".join(c.strip() for i, c in enumerate(r.strip("|").split("|")) if i in keep) + " |" for r in rows)
    arch = arch.replace("Neurons (commercial, for reference)", "Neurons (commercial)").replace("not runnable", "n/a").replace("s / image (M3 Pro)", "s / image").replace("separate video model", "separate model").replace("not disclosed", "n/d").replace("no licence file", "none")
    (TABS / "arch.tex").write_text(latex_table(arch, "lrllrrr"))

    subprocess.run(["tectonic", "--keep-intermediates", "main.tex"], cwd=HERE, check=True)
    with tarfile.open(HERE / "arxiv.tar.gz", "w:gz") as t:
        for f in ["main.tex", "main.bbl", "refs.bib", *[f"figs/{p.name}" for p in sorted(FIGS.iterdir())],
                  *[f"tables/{p.name}" for p in sorted(TABS.iterdir())]]:
            t.add(HERE / f, arcname=f)
    for f in HERE.glob("main.*"):
        if f.suffix not in (".tex", ".pdf", ".bbl"):
            f.unlink()
    print("ok:", HERE / "main.pdf", HERE / "arxiv.tar.gz")


if __name__ == "__main__":
    main()
