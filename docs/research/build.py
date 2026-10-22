"""Builds the research report (HTML + PDF) from benchmarks/results/*.json.

    python docs/research/build.py        # writes docs/research/report.html and docs/research/MediaAttention-report.pdf
"""
import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import tables  # noqa: E402

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

NAMES = {"deepgaze_msdb": "DeepGaze MSDB", "deepgaze_iie": "DeepGaze IIE", "unisal": "UNISAL",
         "transalnet": "TranSalNet-Dense", "center": "Centre-bias baseline", "spectral": "Spectral residual",
         "unisal_ueyes": "MediaAttention-Ads v1 (ours)"}
LICENCE = {"deepgaze_msdb": "no licence file", "deepgaze_iie": "no licence file", "unisal": "Apache-2.0",
           "transalnet": "MIT", "center": "–", "spectral": "–", "unisal_ueyes": "Apache-2.0"}
BASE = ["deepgaze_msdb", "deepgaze_iie", "unisal", "transalnet", "center", "spectral"]
INK, MUTED, LINE, ACCENT = "#1b1d22", "#6b6e78", "#d8dade", "#2f63e6"


def load(dataset):
    p = ROOT / "benchmarks/results" / f"{dataset}.json"
    return json.loads(p.read_text()) if p.exists() else None


# ------------------------------------------------------------------ figures (static SVG, print)

def bars(r, metric="CC"):
    keys = sorted(BASE, key=lambda k: -r[k]["all"][metric][0])
    W, row, left, right, top = 640, 30, 170, 60, 10
    H = top + row * len(keys) + 30
    x = lambda v: left + v / 0.8 * (W - left - right)
    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Correlation with human attention per model">']
    for t in (0, 0.2, 0.4, 0.6, 0.8):
        s.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top}" y2="{H - 24}" stroke="{LINE}" stroke-width="1"/>'
                 f'<text x="{x(t):.1f}" y="{H - 8}" text-anchor="middle" class="ax">{t:.1f}</text>')
    for i, k in enumerate(keys):
        v, se = r[k]["all"][metric]
        y = top + i * row + 8
        col = ACCENT if k == "unisal" else INK
        s.append(f'<text x="{left - 10}" y="{y + 11}" text-anchor="end" class="lb">{NAMES[k]}{" (default)" if k == "unisal" else ""}</text>'
                 f'<rect x="{left}" y="{y}" width="{x(v) - left:.1f}" height="14" rx="3" fill="{col}"/>'
                 f'<line x1="{x(v - 1.96 * se):.1f}" x2="{x(v + 1.96 * se):.1f}" y1="{y + 7}" y2="{y + 7}" stroke="#fff" stroke-width="1.5"/>'
                 f'<text x="{x(v) + 8:.1f}" y="{y + 11}" class="val">{v:.3f}</text>')
    s.append("</svg>")
    return "".join(s)


def scatter(r):
    W, H, L, B, T, R = 640, 300, 56, 40, 14, 20
    tmin, tmax, cmin, cmax = 0.002, 5.0, 0.1, 0.75
    x = lambda t: L + (math.log10(max(t, tmin)) - math.log10(tmin)) / (math.log10(tmax) - math.log10(tmin)) * (W - L - R)
    y = lambda c: T + (1 - (c - cmin) / (cmax - cmin)) * (H - T - B)
    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Accuracy (CC) against seconds per image, log scale">']
    for c in (0.2, 0.4, 0.6):
        s.append(f'<line x1="{L}" x2="{W - R}" y1="{y(c):.1f}" y2="{y(c):.1f}" stroke="{LINE}"/><text x="{L - 8}" y="{y(c) + 4:.1f}" text-anchor="end" class="ax">{c:.1f}</text>')
    for t, lab in ((0.01, "0.01 s"), (0.1, "0.1 s"), (1, "1 s")):
        s.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{T}" y2="{H - B}" stroke="{LINE}"/><text x="{x(t):.1f}" y="{H - B + 18}" text-anchor="middle" class="ax">{lab}</text>')
    s.append(f'<text x="{(L + W - R) / 2}" y="{H - 4}" text-anchor="middle" class="ax">seconds per image, Apple M3 Pro (log scale)</text>'
             f'<text x="14" y="{(T + H - B) / 2}" text-anchor="middle" class="ax" transform="rotate(-90 14 {(T + H - B) / 2})">CC with human attention</text>')
    offsets = {"deepgaze_msdb": (0, -12, "middle"), "deepgaze_iie": (0, 22, "middle"), "unisal": (0, -12, "middle"),
               "transalnet": (0, 22, "middle"), "center": (10, 4, "start"), "spectral": (10, 4, "start")}
    for k in BASE:
        t, c = r[k]["s_per_image"], r[k]["all"]["CC"][0]
        dx, dy, anchor = offsets[k]
        col = ACCENT if k == "unisal" else INK
        s.append(f'<circle cx="{x(t):.1f}" cy="{y(c):.1f}" r="5.5" fill="{col}" stroke="#fff" stroke-width="2"/>'
                 f'<text x="{x(t) + dx:.1f}" y="{y(c) + dy:.1f}" text-anchor="{anchor}" class="lb">{NAMES[k]}</text>')
    s.append("</svg>")
    return "".join(s)


# ------------------------------------------------------------------ tables

def results_table(r, keys=None, timing=True):
    keys = sorted([k for k in (keys or BASE) if k in r], key=lambda k: -r[k]["all"]["CC"][0])
    rows = []
    for k in keys:
        a = r[k]["all"]
        name = f"<b>{NAMES[k]}</b>" if k in ("unisal", "unisal_ueyes") else NAMES[k]
        rows.append(f"<tr><td>{name}</td>" + "".join(f"<td class=n>{a[m][0]:.3f}</td>" for m in ("CC", "NSS", "sAUC", "AUC", "SIM", "KLD"))
                    + (f"<td class=n>{r[k]['s_per_image']:.2f}</td>" if timing else "") + f"<td>{LICENCE[k]}</td></tr>")
    return ("<table><thead><tr><th>Model</th><th>CC ↑</th><th>NSS ↑</th><th>sAUC ↑</th><th>AUC ↑</th><th>SIM ↑</th><th>KLD ↓</th>"
            + ("<th>s / image</th>" if timing else "") + "<th>Licence</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def category_table(r, keys=None):
    keys = [k for k in (keys or BASE) if k in r]
    cats = sorted(r[keys[0]]["by_category"])
    rows = "".join(f"<tr><td>{NAMES[k]}</td>" + "".join(f"<td class=n>{r[k]['by_category'][c]['CC']:.3f}</td>" for c in cats) + "</tr>"
                   for k in keys)
    return "<table><thead><tr><th>Model</th>" + "".join(f"<th>{c.capitalize()}</th>" for c in cats) + "</tr></thead><tbody>" + rows + "</tbody></table>"


def ueyes_section(ueyes, test, fiwi):
    if not ueyes:
        return "<p>Results on UEyes are added in the next revision of this report.</p>"
    present = [k for k in BASE if k in ueyes]
    best = max(present, key=lambda k: ueyes[k]["all"]["CC"][0])
    html = (f"<h3>5.4 UEyes: posters, ads and user interfaces</h3>"
            f"<p>UEyes [27] holds 1,980 screenshots in four categories (webpages, desktop UIs, mobile UIs and posters, "
            f"about 300 of which are real ads and posters), viewed freely for 7 s by 62 participants on a Gazepoint GP3. "
            f"None of the six models was trained on any of it, so we evaluate on all 1,980 images. Fixations are mapped "
            f"to image pixels exactly as the dataset's own code does; ground truth is blurred with ≈1° of visual angle.</p>"
            + results_table(ueyes, present, timing=False)
            + f"<p class='tcap'><b>Table 6.</b> UEyes, {ueyes[present[0]]['n']} images, models as released. "
              f"Timing is omitted: these runs shared the machine with training (see Table 4 for speed).</p>"
            + category_table(ueyes, [k for k in present if k not in ("center", "spectral")] + [k for k in present if k in ("center", "spectral")])
            + "<p class='tcap'><b>Table 7.</b> CC by UEyes category.</p>")
    if "unisal" in ueyes:
        u = ueyes["unisal"]["all"]["CC"][0]
        html += (f"<p>{NAMES[best]} is the most accurate on UEyes (CC {ueyes[best]['all']['CC'][0]:.3f}); UNISAL reaches {u:.3f}. "
                 f"Interfaces and posters are harder than webpages for every model, which is where design-specific data matters. Model size does not close the gap: DeepGaze MSDB, at 452M parameters, ties DeepGaze IIE and trails 3.7M UNISAL in CC (MSDB was run on a Colab T4 GPU with the same code).</p>")
    if test and "unisal_ueyes" in test and "unisal" in test:
        keys = [k for k in ("unisal_ueyes", "unisal", "deepgaze_iie", "deepgaze_msdb", "transalnet") if k in test]
        gain = test["unisal_ueyes"]["all"]["CC"][0] - test["unisal"]["all"]["CC"][0]
        fiwi_gain = (fiwi["unisal_ueyes"]["all"]["CC"][0] - fiwi["unisal"]["all"]["CC"][0]) if "unisal_ueyes" in fiwi else None
        html += ("<h3>5.5 An ad- and UI-tuned model</h3>"
                 "<p>We fine-tuned UNISAL on the UEyes training split (KL − CC loss, best epoch chosen on a held-out tenth "
                 "of the training images) and evaluated it on the UEyes test split, 108 images it never saw, and on FiWI, "
                 "which is out of domain for both.</p>"
                 + results_table(test, keys, timing=False)
                 + f"<p class='tcap'><b>Table 8.</b> UEyes test split. Fine-tuning changes CC by {gain:+.3f}"
                 + (f"; on FiWI by {fiwi_gain:+.3f}." if fiwi_gain is not None else ".") + "</p>"
                 "<p>We call the fine-tuned network <b>MediaAttention-Ads v1</b>. It keeps UNISAL's architecture (3.7M parameters) and "
                 "was trained for 8 epochs (AdamW, lr 3e-5, batch norm frozen, about 25 minutes on an Apple M3 Pro). Held-out CC rose "
                 "from 0.502 to 0.769, with the best epoch at 5. Table 9 isolates the 27 test posters, the category closest to print "
                 "and display advertising.</p>"
                 + tables.to_html(tables.posters_table()[0])
                 + "<p class='tcap'><b>Table 9.</b> UEyes test posters (n = 27), 95% confidence intervals over images. "
                   "MediaAttention-Ads v1's interval does not overlap any other model's: +0.256 CC (+52%) over the best other open model.</p>")
    return html


def real_ads_section():
    p = ROOT / "benchmarks/results/wikimedia_real_ads.json"
    if not p.exists():
        return ""
    r = json.loads(p.read_text())
    b, a = r["rules"]["before_fix"], r["rules"]["after_fix"]
    row = lambda k, lab: (f"<tr><td>{lab}</td><td class=n>{b['median'].get(k, float('nan')):.1f}</td>"
                          f"<td class=n>{a['median'].get(k, float('nan')):.1f}</td></tr>")
    rng = r["recommended_ranges_after_fix"]
    return ("<h3>5.6 Behaviour on 272 real ads</h3>"
            "<p>To see how the pipeline behaves on real creatives, we scored 272 openly licensed print ads and advertising "
            "posters from Wikimedia Commons (CLIP-filtered from 300 to remove photographs of ads in the wild). There is no "
            "eye tracking here, so this measures behaviour, not accuracy. A visual audit found four layout rules that real "
            "ads broke (coloured bands read as buttons, a top/bottom wordmark fallback that picked fine print, and headline "
            "and body boxes merged across the page). Fixing them changed the medians substantially:</p>"
            "<table><thead><tr><th>Median per ad</th><th>Before fix</th><th>After fix</th></tr></thead><tbody>"
            + row("body_attention", "Body-copy attention (%)") + row("headline_attention", "Headline attention (%)")
            + row("brand_attention", "Brand attention (%)") + row("cta_attention", "CTA attention (%)")
            + row("focus", "Focus") + row("cognitive_demand", "Cognitive demand") + "</tbody></table>"
            f"<p class='tcap'><b>Table 10.</b> 272 Wikimedia ads, stock UNISAL. Targets calibrated on these ads "
            f"(top 20%; middle 50% for complexity): focus {rng['focus'][0]}–{rng['focus'][1]}, cognitive demand "
            f"{rng['cognitive_demand'][0]}–{rng['cognitive_demand'][1]}, brand attention ≥ {rng['brand_attention'][0]}%. "
            "These are vintage print ads, so treat them as a reference, not as targets for modern digital creative.</p>")


def arch_section():
    return ("<h3>5.7 Architectures compared</h3>"
            "<p>Table 11 sets the models side by side, with Neurons' commercial model as described in its public tech paper [30] "
            "for reference. Accuracy columns are from Tables 3 and 8.</p>"
            + tables.to_html(tables.arch_table())
            + "<p class='tcap'><b>Table 11.</b> Architecture, size, training data, video support, licence and speed. "
              "Parameter counts were measured by loading each model.</p>"
              "<div class='key'><p><b>Where MediaAttention-Ads v1 wins.</b> (1) Accuracy on designed content: it leads every "
              "UEyes category by 0.19–0.26 CC and is the best of all models on FiWI's text-heavy pages. (2) Efficiency: it is 20–120× "
              "smaller and 17–70× faster than DeepGaze, so it runs on a laptop CPU. (3) One network for images and video. "
              "(4) Apache-2.0 code and weights.</p>"
              "<p><b>Where it does not.</b> DeepGaze MSDB, at 120× the size, remains ~0.05 CC ahead on FiWI and further ahead on "
              "photo-led pages. The video branch is not ad-tuned. UEyes is one lab panel of 62 participants viewing freely, "
              "unlike a one-second glance in a feed.</p></div>")


# ------------------------------------------------------------------ document

def build():
    fiwi = load("fiwi")
    ueyes = load("ueyes")
    ueyes_test = load("ueyes_test")
    u, iie, msdb, tsn = (fiwi[k]["all"]["CC"][0] for k in ("unisal", "deepgaze_iie", "deepgaze_msdb", "transalnet"))
    speedup = fiwi["deepgaze_iie"]["s_per_image"] / fiwi["unisal"]["s_per_image"]
    text_cc = {k: fiwi[k]["by_category"]["text"]["CC"] for k in BASE}
    pict_cc = {k: fiwi[k]["by_category"]["pictorial"]["CC"] for k in BASE}

    html = TEMPLATE.format(
        css=CSS, bars=bars(fiwi), scatter=scatter(fiwi), results=results_table(fiwi), categories=category_table(fiwi),
        ueyes=ueyes_section(ueyes, ueyes_test, fiwi) + real_ads_section() + arch_section(), u=u, iie=iie, msdb=msdb, tsn=tsn, speedup=speedup,
        gap_lo=iie - u, gap_hi=msdb - u, text_u=text_cc["unisal"], text_msdb=text_cc["deepgaze_msdb"],
        pict_u=pict_cc["unisal"], pict_msdb=pict_cc["deepgaze_msdb"], center=fiwi["center"]["all"]["CC"][0],
        version="v0.2 · MediaAttention-Ads v1")
    out = HERE / "report.html"
    out.write_text(html)
    pdf = HERE / "MediaAttention-report.pdf"
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=10000",
                    f"--print-to-pdf={pdf}", out.resolve().as_uri()], check=True, capture_output=True)
    print(out, pdf, sep="\n")


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..800&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap');
@page { size: A4; margin: 20mm 18mm 20mm 18mm; @bottom-center { content: counter(page); font: 9pt 'Archivo', sans-serif; color: #6b6e78; } }
@page :first { @bottom-center { content: none; } }
:root { --ink: #1b1d22; --muted: #6b6e78; --line: #d8dade; --accent: #2f63e6; }
* { box-sizing: border-box; }
body { margin: 0; color: var(--ink); font: 10.4pt/1.52 'Source Serif 4', Georgia, serif; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
main { max-width: 174mm; margin: 0 auto; }
h1, h2, h3, .label, th, .ax, .lb, .val, figcaption, .meta, .abstract h2 { font-family: 'Archivo', system-ui, sans-serif; }
.cover { padding-top: 6mm; border-bottom: 1.5pt solid var(--ink); margin-bottom: 8mm; padding-bottom: 6mm; }
.label { font-size: 8pt; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); font-variation-settings: 'wdth' 125; }
h1 { font-variation-settings: 'wdth' 62; font-weight: 800; text-transform: uppercase; font-size: 34pt; line-height: .92; letter-spacing: -.01em; margin: 6mm 0 5mm; }
.meta { font-size: 9.5pt; color: var(--muted); }
.meta b { color: var(--ink); font-weight: 600; }
.abstract { background: #f4f5f7; padding: 5mm 6mm; margin: 0 0 8mm; }
.abstract h2 { margin: 0 0 2mm; font-size: 9pt; letter-spacing: .12em; text-transform: uppercase; }
.abstract p { margin: 0; }
h2 { font-variation-settings: 'wdth' 62; font-weight: 800; text-transform: uppercase; font-size: 17pt; line-height: 1; margin: 9mm 0 3mm; break-after: avoid; }
h3 { font-size: 11pt; font-weight: 600; margin: 6mm 0 2mm; break-after: avoid; }
p { margin: 0 0 3mm; text-align: justify; hyphens: auto; }
ul, ol { margin: 0 0 3mm; padding-left: 5mm; }
li { margin-bottom: 1mm; }
table { width: 100%; border-collapse: collapse; margin: 2mm 0 2mm; font-size: 8.4pt; line-height: 1.35; break-inside: auto; }
th { text-align: left; font-size: 7.2pt; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); font-weight: 600; border-bottom: 1pt solid var(--ink); padding: 1.6mm 1.6mm 1.6mm 0; vertical-align: bottom; }
td { border-bottom: .5pt solid var(--line); padding: 1.5mm 1.6mm 1.5mm 0; vertical-align: top; }
tr { break-inside: avoid; }
tr.ours td { font-weight: 600; color: #2f63e6; }
td.n { font-family: 'Archivo', sans-serif; font-variant-numeric: tabular-nums; white-space: nowrap; }
figure { margin: 4mm 0 5mm; break-inside: avoid; }
figure img { width: 100%; }
figcaption, .tcap { font-size: 8.4pt; color: var(--muted); margin-top: 2mm; line-height: 1.4; }
.tcap { margin: 4mm 0 0; font-family: 'Archivo', sans-serif; }
.tcap b, figcaption b { color: var(--ink); }
svg .ax { font-size: 11px; fill: #6b6e78; }
svg .lb { font-size: 12px; fill: #1b1d22; }
svg .val { font-size: 11px; fill: #1b1d22; font-variant-numeric: tabular-nums; }
.key { border-left: 2pt solid var(--accent); padding: 1mm 0 1mm 4mm; margin: 4mm 0; }
.refs { font-size: 8.4pt; line-height: 1.45; padding-left: 7mm; }
.refs li { margin-bottom: 1.4mm; }
a { color: var(--ink); text-decoration-color: var(--line); }
.small { font-size: 8.6pt; color: var(--muted); }
code { font-family: ui-monospace, Menlo, monospace; font-size: 8.4pt; }
pre { font: 8.2pt/1.5 ui-monospace, Menlo, monospace; background: #f4f5f7; padding: 3mm 4mm; white-space: pre-wrap; }
"""

TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Predicting where people look at ads: a review and an open benchmark</title>
<meta name="author" content="Pratyush Mahapatra"><style>{css}</style></head><body><main>

<section class="cover">
  <div class="label">Research report · {version} · October 2026</div>
  <h1>Predicting where people look at ads</h1>
  <p style="font-size:13pt;line-height:1.35;margin:0 0 5mm;text-align:left">A review of commercial and open attention-prediction models, and an open benchmark against human eye tracking</p>
  <div class="meta"><b>Pratyush Mahapatra</b> · <a href="https://pranikz.dev">pranikz.dev</a> · Code, data pointers and results: <a href="https://github.com/pranikz/MediaAttention">github.com/pranikz/MediaAttention</a></div>
</section>

<section class="abstract"><h2>Abstract</h2>
<p>Predicted-attention tools promise eye-tracking insight on any creative in seconds, and are now common in advertising pre-testing. We review 15 commercial predictive tools and five measured-attention vendors from public sources, and find that their headline accuracy claims ("92–95%") are mostly AUC expressed as a share of a human ceiling, a saturated metric on which almost every model scores highly. Only four commercial tools publish third-party-computed scores on the MIT/Tübingen benchmark. We then review the open-source state of the art, including licences, which rule out several leading models for commercial use, and benchmark six openly available models against human eye tracking on 149 webpages (FiWI), none of which they were trained on. DeepGaze MSDB is the most accurate (CC {msdb:.3f}). UNISAL, the strongest model with a permissive licence, reaches CC {u:.3f} at {speedup:.0f}× the speed of DeepGaze IIE. All stock models are weakest on text-heavy layouts and on posters and interfaces (UEyes). Fine-tuning UNISAL on UEyes' training split gives <b>MediaAttention-Ads v1</b>: on held-out posters it reaches CC 0.752 against 0.496 for the best other open model, with non-overlapping 95% confidence intervals, and it also improves on unseen webpages. We release the benchmark, the model and MediaAttention, an open toolkit that uses it, so every number here can be reproduced.</p></section>

<h2>1. Introduction</h2>
<p>Attention is the first gate in advertising. A brand, product or message that is not looked at cannot persuade or be remembered, and eye-tracking studies of print ads show that pictures, text size and brand placement shape where attention goes and how it transfers between elements [17]. Visual salience alone can bias quick consumer choices, even against stated preferences [19]. Measuring attention with eye tracking is accurate but slow and costly, so a market has grown around <i>predicting</i> it: upload a creative, receive a heatmap and a set of scores.</p>
<p>For buyers, the difficulty is that these tools are hard to compare. Vendors quote accuracy figures computed on different metrics, test sets and viewing durations, often on their own data. This report makes four contributions:</p>
<ol>
<li><b>A market review</b> of 15 commercial predictive-attention tools and five measured-attention vendors, decoding what their accuracy claims mean (Section 3).</li>
<li><b>An open-model review</b> covering accuracy, training data (for contamination) and licences (Section 4).</li>
<li><b>An open benchmark</b> of six models against human eye tracking on data none of them was trained on (Section 5).</li>
<li><b>MediaAttention-Ads v1</b>, an attention model fine-tuned on design eye tracking that leads every open model on posters, ads and interfaces (Section 5.5), and <b>MediaAttention</b>, an open-source toolkit that turns its attention maps into ad metrics.</li>
</ol>

<h2>2. Background</h2>
<h3>2.1 Attention-prediction models</h3>
<p>Computational saliency began with biologically inspired feature models that combine intensity, colour and orientation contrast [14]. Since 2015, deep networks trained on large fixation datasets such as SALICON [4] (mouse-contingent viewing on 10,000 MS-COCO images) and MIT1003 [5] (eye tracking, 15 viewers) have dominated. The MIT/Tübingen benchmark [2, 29] evaluates models on held-out data (MIT300, CAT2000) and is the closest thing the field has to a neutral scoreboard.</p>
<h3>2.2 Metrics</h3>
<p>No single metric captures prediction quality [6]. We report six, all computed per image and averaged:</p>
<table><thead><tr><th>Metric</th><th>Compares</th><th>Reading</th></tr></thead><tbody>
<tr><td>AUC</td><td>Prediction at fixated pixels vs. all pixels</td><td>Ranking quality; saturated (centre bias alone ≈ 0.78 on MIT300)</td></tr>
<tr><td>sAUC</td><td>Prediction at fixations vs. other images' fixations</td><td>Cancels centre bias; 0.5 = no better than a centre prior</td></tr>
<tr><td>NSS</td><td>Normalised prediction at fixations</td><td>Higher is better; 0 = chance</td></tr>
<tr><td>CC</td><td>Pearson correlation with the blurred fixation map</td><td>Our headline metric; 1 = identical maps</td></tr>
<tr><td>SIM</td><td>Histogram intersection of the two distributions</td><td>Overlap; 1 = identical</td></tr>
<tr><td>KLD</td><td>KL divergence of prediction from ground truth</td><td>Lower is better; punishes missed fixations</td></tr>
</tbody></table>
<h3>2.3 Datasets and contamination</h3>
<p>Most models are trained on SALICON and MIT1003, natural photographs viewed freely for a few seconds. Advertising differs: it is dense with text, logos and layout. Design-oriented eye-tracking sets are rare. FiWI [26] covers 149 webpages (pictorial, text and mixed), with 11 viewers and 5 s free viewing. UEyes [27] covers 1,980 posters, webpages and app screens, with 62 participants and up to 7 s viewing. Because neither is in the training data of the models we test, both measure generalisation rather than memorisation.</p>

<h2>3. Commercial tools</h2>
<p>We surveyed vendor websites, documentation, white papers and third-party listings in October 2026. Figures below are <b>as stated by each vendor</b> unless marked otherwise. "Not disclosed" means we found no public statement.</p>
<table><thead><tr><th>Tool</th><th>Approach (as disclosed)</th><th>Accuracy claim</th><th>Independent evidence</th><th>Inputs · API · price</th></tr></thead><tbody>
<tr><td>Neurons AI</td><td>CNN encoder–decoder trained on proprietary eye tracking (~20k participants); separate emotion, memory and cognitive-demand models [30]</td><td>"95%+"; tech paper: CC 0.944 on own ad data</td><td>None found; not on MIT300</td><td>Image, video · API · quote</td></tr>
<tr><td>Dragonfly AI</td><td>Patented biologically inspired algorithm (QMUL); no training data disclosed</td><td>"Within 10% of a human sample"; metric not given</td><td>None found</td><td>Image, video, web, shelf · API · quote</td></tr>
<tr><td>Attention Insight</td><td>CNN; 5.5M fixations</td><td>92.5% = MIT300 AUC 0.864 ÷ ceiling 0.934</td><td><b>On MIT300:</b> CC 0.758</td><td>Image, URL, video · API · €29–299/mo</td></tr>
<tr><td>3M VAS</td><td>Feature-based model (edges, intensity, colour, faces)</td><td>"92%"; own study: ~85% of the ROC ceiling</td><td>Own white paper only</td><td>Image, video · no API · ~$550/yr</td></tr>
<tr><td>EyeQuant</td><td>"7 distinct neural models"; 1.6M data points</td><td>"90%+", AUC relative to the human ceiling</td><td>None current</td><td>Image, page · API (enterprise) · from $97/mo</td></tr>
<tr><td>expoze.io (alpha.one)</td><td>Proprietary neural net</td><td>"95%" = 0.87 vs 0.92 (likely AUC)</td><td>One peer-reviewed comparison [31]</td><td>Image, video · API · quote</td></tr>
<tr><td>Brainsight</td><td>Proprietary and MIT/Tübingen data</td><td>"94%"; metric not given</td><td>None found</td><td>Image, video, URL · API · €199–899/mo</td></tr>
<tr><td>Feng-GUI</td><td>Simulates a 5 s, 40-person test</td><td>"92% MIT-validated"</td><td>Not on current MIT300</td><td>Image, web, video · API · quote</td></tr>
<tr><td>Clueify</td><td>Model trained on eye tracking</td><td>"92% consistent"</td><td><b>On MIT300:</b> AUC 0.881, CC 0.575</td><td>Image · quote</td></tr>
<tr><td>Emotiva</td><td>Not disclosed</td><td>Not disclosed</td><td><b>On MIT300:</b> CC 0.743</td><td>Image, video · API · demo</td></tr>
<tr><td>MedullaAI</td><td>Not disclosed</td><td>Not disclosed</td><td><b>On MIT300:</b> CC 0.670</td><td>Not disclosed</td></tr>
<tr><td>EyeSee predictive</td><td>11.5M gaze points</td><td>None</td><td>None</td><td>Packaging, visuals</td></tr>
<tr><td>Kantar LINK AI</td><td>Kantar eye-tracking database</td><td>None</td><td>None</td><td>Video</td></tr>
<tr><td>Amplified AttentionAI</td><td>Attention-decay prediction from 50B+ data points</td><td>None</td><td>None</td><td>Creative · API</td></tr>
<tr><td>VisualEyes / Loceye</td><td>Acquired by Neurons in 2022; discontinued</td><td>84–86% (CC, own data)</td><td>–</td><td>–</td></tr>
</tbody></table>
<p class="tcap"><b>Table 1.</b> Commercial predictive-attention tools (public sources, October 2026). Measured-attention vendors, which use panels with webcam or hardware eye tracking (Realeyes/Adverteyes, Amplified Intelligence, Lumen, Adelaide, Tobii/ATTEX), are a different product: they measure real viewers and model media-level attention, not a gaze map for one creative.</p>

<h3>3.1 What "X% accurate" means</h3>
<p>Where a claim can be decoded, it is usually <b>AUC divided by the inter-observer ceiling</b>. For example, Attention Insight's 92.5% is its MIT300 AUC (0.864) over the gold standard (0.934). AUC is saturated: on MIT300 every listed model scores between 0.84 and 0.89, and a centre-only prior already reaches 0.78. Expressed as a share of the ceiling, small differences look large and weak models look strong. On CC, the same tools separate clearly. Clueify ranks well on AUC (0.881) yet has the lowest CC among them (0.575).</p>
<table><thead><tr><th>Model (MIT300)</th><th>AUC</th><th>sAUC</th><th>NSS</th><th>CC</th><th>KLD</th><th>Type</th></tr></thead><tbody>
<tr><td>DeepGaze MSDB</td><td class=n>0.894</td><td class=n>0.816</td><td class=n>2.74</td><td class=n>0.883</td><td class=n>0.254</td><td>open, research</td></tr>
<tr><td>DeepGaze IIE</td><td class=n>0.883</td><td class=n>0.794</td><td class=n>2.53</td><td class=n>0.824</td><td class=n>0.347</td><td>open, research</td></tr>
<tr><td>TranSalNet-Dense</td><td class=n>0.873</td><td class=n>0.747</td><td class=n>2.41</td><td class=n>0.807</td><td class=n>1.014</td><td>open (MIT)</td></tr>
<tr><td>UNISAL</td><td class=n>0.877</td><td class=n>0.784</td><td class=n>2.37</td><td class=n>0.785</td><td class=n>0.415</td><td>open (Apache-2.0)</td></tr>
<tr><td>Attention Insight</td><td class=n>0.864</td><td class=n>0.745</td><td class=n>2.18</td><td class=n>0.758</td><td class=n>0.887</td><td>commercial</td></tr>
<tr><td>Emotiva</td><td class=n>0.864</td><td class=n>0.770</td><td class=n>2.20</td><td class=n>0.743</td><td class=n>0.496</td><td>commercial</td></tr>
<tr><td>MedullEyes-V1</td><td class=n>0.841</td><td class=n>0.724</td><td class=n>1.89</td><td class=n>0.670</td><td class=n>0.659</td><td>commercial</td></tr>
<tr><td>Clueify</td><td class=n>0.881</td><td class=n>0.765</td><td class=n>1.49</td><td class=n>0.575</td><td class=n>0.889</td><td>commercial</td></tr>
<tr><td>Centre bias</td><td class=n>0.783</td><td class=n>–</td><td class=n>1.10</td><td class=n>0.446</td><td class=n>0.951</td><td>baseline</td></tr>
<tr><td>Gold standard (humans)</td><td class=n>0.934</td><td class=n>0.883</td><td class=n>3.14</td><td class=n>0.983</td><td class=n>0.060</td><td>ceiling</td></tr>
</tbody></table>
<p class="tcap"><b>Table 2.</b> MIT/Tübingen MIT300 leaderboard [29], accessed 21 Oct 2026. These are natural images with free viewing. Four commercial tools have third-party-computed entries, and all of them score below the best open models. The leaderboard computes a metric-specific map from probabilistic models, which favours entries that submit densities (DeepGaze, UNISAL).</p>

<h3>3.2 In-domain claims are not benchmark results</h3>
<p>The most detailed public technical document we found, Neurons' tech paper, reports CC 0.944 against held-out data from its own ad corpus and compares this with the best research model's MIT300 CC (0.883) [30]. The two numbers come from different test sets, ceilings and viewing durations (5 s vs ~3 s), so they cannot be compared directly. A fair comparison needs every model on the same held-out data, which is what Section 5 does for the open models. Secondary metrics (clarity, cognitive load, memory, emotion) are defined differently by each vendor, and we found no external validation of any of them.</p>

<h2>4. Open models</h2>
<table><thead><tr><th>Model</th><th>Year</th><th>Trained on</th><th>MIT300 CC</th><th>Code / weights licence</th><th>Laptop-friendly</th></tr></thead><tbody>
<tr><td>DeepGaze MSDB [28]</td><td>2025</td><td>MIT1003, CAT2000, COCO-Freeview, DAEMONS, FIGRIM</td><td class=n>0.883</td><td>No licence file</td><td>Yes (slow)</td></tr>
<tr><td>DeepGaze IIE [1]</td><td>2021</td><td>SALICON → MIT1003</td><td class=n>0.824</td><td>No licence file</td><td>Yes</td></tr>
<tr><td>TranSalNet [32]</td><td>2022</td><td>SALICON</td><td class=n>0.807</td><td>MIT</td><td>Yes</td></tr>
<tr><td>UNISAL [25]</td><td>2020</td><td>SALICON, DHF1K, Hollywood-2, UCF-Sports</td><td class=n>0.785</td><td>Apache-2.0</td><td>Yes (4M params)</td></tr>
<tr><td>MSI-Net</td><td>2020</td><td>SALICON</td><td class=n>0.779</td><td>MIT</td><td>TensorFlow</td></tr>
<tr><td>SUM [33]</td><td>2024</td><td>SALICON, MIT1003, CAT2000, OSIE, SalECI, UEyes</td><td class=n>–</td><td>MIT</td><td>CUDA only</td></tr>
<tr><td>UMSI</td><td>2020</td><td>Imp1k importance (clicks, not gaze)</td><td class=n>–</td><td>Non-commercial</td><td>Keras</td></tr>
</tbody></table>
<p class="tcap"><b>Table 3.</b> Open image models. Licences matter: the most accurate family (DeepGaze) ships without a licence file, which by default means all rights reserved. Video models (ViNet, TMFI-Net, SalFoM) score well on DHF1K. On real TV ads, a recent study found per-frame DeepGaze IIE reached CC 0.231 against 0.560 for the video model ViNet-S [34], so image models applied frame by frame understate attention dynamics in video.</p>

<h2>5. Benchmark</h2>
<h3>5.1 Setup</h3>
<ul>
<li><b>Data.</b> FiWI [26]: 149 webpages at 1360×768, 11 viewers, 5 s free viewing, EyeLink 1000. Ground truth is all viewers' fixations, blurred with the dataset's own Gaussian (σ = 32 px) for CC, SIM and KLD. None of the six models was trained on FiWI.</li>
<li><b>Models.</b> We ran each model as released, without fine-tuning:
<ul>
<li>DeepGaze IIE and DeepGaze MSDB, with a Gaussian centre-bias prior; MSDB uses averaged dataset parameters and the dataset's 50 px per degree.</li>
<li>TranSalNet-Dense, SALICON weights, with its own resize and padding.</li>
<li>UNISAL, SALICON weights, as packaged in MediaAttention (identical to the original repository's output, CC &gt; 0.9999).</li>
<li>Two baselines: a Gaussian centre prior and spectral residual [3].</li>
</ul></li>
<li><b>Metrics.</b> pysaliency [2], with AUC computed over all pixels, sAUC over other images' fixations, and MIT/Tübingen KLD regularisation. All metrics are computed per image, then averaged.</li>
<li><b>Hardware.</b> Apple M3 Pro, PyTorch MPS. Timing is per-image inference after the model has loaded.</li>
</ul>

<h3>5.2 Results</h3>
{results}
<p class="tcap"><b>Table 4.</b> FiWI, 149 images. Mean per-image scores; ↑ higher is better, ↓ lower is better.</p>
<figure>{bars}<figcaption><b>Figure 1.</b> Correlation (CC) between each model's prediction and human fixations on FiWI. White ticks show 95% confidence intervals (±1.96 SE over images). The highlighted model is stock UNISAL, the network MediaAttention-Ads v1 is built on (Section 5.5).</figcaption></figure>
<figure>{scatter}<figcaption><b>Figure 2.</b> Accuracy against speed. DeepGaze MSDB is the most accurate. UNISAL is the fastest learned model and the most accurate one with a permissive licence.</figcaption></figure>

<h3>5.3 Findings</h3>
<div class="key"><p><b>Ranking.</b> DeepGaze MSDB leads on every metric (CC {msdb:.3f}), followed by DeepGaze IIE ({iie:.3f}), UNISAL ({u:.3f}) and TranSalNet ({tsn:.3f}). Every learned model beats the centre prior (CC {center:.3f}) by a wide margin. Spectral residual, a classic low-level model, falls below the centre prior.</p></div>
<p><b>The licence–accuracy trade-off is small.</b> UNISAL trails DeepGaze IIE by {gap_lo:.3f} CC and MSDB by {gap_hi:.3f}. It runs at 0.04 s per image, about {speedup:.0f}× faster than IIE, and is the only model in the top three that can be redistributed and used commercially without asking. That made it the base for MediaAttention's own model (Section 5.5); DeepGaze stays available as an option.</p>
<p><b>Text is the hard case.</b> Every model scores lowest on text-heavy pages: UNISAL scores {text_u:.3f} on text pages against {pict_u:.3f} on pictorial ones, and MSDB {text_msdb:.3f} against {pict_msdb:.3f}. Natural-image training data teaches models to look at faces and objects, not at the reading patterns that dominate text layouts. Advertising copy is exactly this kind of content, which makes ad-specific training data the most valuable next step.</p>
{categories}
<p class="tcap"><b>Table 5.</b> CC by FiWI page type.</p>
<figure><img src="fig/models_on_ad.jpg" alt="Six models' predicted attention on the same synthetic ad"><figcaption><b>Figure 3.</b> Each model's prediction on the same synthetic ad (MediaAttention's example). The learned models agree on the headline and call to action, and differ in how much attention they give the logo. Spectral residual fixates on high-contrast shapes.</figcaption></figure>
{ueyes}

<h2>6. Discussion</h2>
<h3>6.1 Questions to ask a vendor</h3>
<ol>
<li>Which metric? Ask for CC and NSS, not only AUC.</li>
<li>Which test set, was it held out, and was it scored by a third party?</li>
<li>Which viewing duration and which human ceiling?</li>
<li>How were secondary scores (clarity, memory, emotion) validated against real outcomes?</li>
</ol>
<p>A tool that publishes MIT/Tübingen entries, or runs on an open benchmark such as this one, can be compared. A bare percentage cannot.</p>
<h3>6.2 What predicted attention can and cannot tell you</h3>
<p>Models trained on free viewing predict bottom-up, first-seconds attention: what pops out. They do not model goal-directed search (a shopper looking for a price), audio in video, or the placement context of an ad. Small independent comparisons of commercial tools with eye tracking point the same way: predicted maps track salience but miss goal-driven attention. Predicted attention is a fast first filter for comparing creative variants, not a replacement for testing with people when stakes are high.</p>

<h3>6.3 An open panel</h3>
<p>Commercial tools rest on private panels, and their secondary metrics (engagement, trust, memory) cannot be checked from outside. MediaAttention ships the panel instead: a static browser study with no server and no uploads.</p>
<ul>
<li><b>Gaze:</b> estimated from MediaPipe face landmarks with a per-participant ridge calibration, a click-to-fixate 9-point grid followed by a 5-point accuracy check.</li>
<li><b>Implicit responses:</b> timed Yes/No word tests, with answers weighted by each person's relative speed.</li>
<li><b>Memory:</b> a recognition test against unseen foil ads.</li>
<li><b>Data handling:</b> participants download their own results file and choose whether to send it back. <code>mediaattention collect</code> converts the files into an eye-tracking dataset that any model can be scored on, plus per-ad implicit scores that plug into KPI calibration.</li>
</ul>
<p>Webcam gaze is coarse (2–4° versus about 0.5° for lab trackers), but aggregated over participants it is enough to validate heatmaps on one's own creatives. It is also a route to openly licensed training data for ad-specific attention and emotion models.</p>

<h2>7. Limitations and roadmap</h2>
<ul>
<li><b>Data.</b> FiWI is small (149 webpages, 11 viewers). UEyes is larger but is one lab panel (62 participants, free viewing, 7 s). Neither measures a one-second glance in a scrolling feed.</li>
<li><b>Scope.</b> We evaluated only openly available models; commercial tools could not be run on the same data.</li>
<li><b>Video and outcomes.</b> Video attention is not yet ad-tuned or validated on ad eye tracking, and secondary metrics have not been validated against campaign outcomes.</li>
</ul>
<p><b>Roadmap.</b> Next: tune the recurrent video branch on ad video eye tracking; score creatives in feed context with 1–2 s exposures; retrain the memory model on openly licensed data; and train a v2 that distils DeepGaze MSDB on photographs while keeping the design data, to close the remaining gap on photo-led creatives. Later: emotion, engagement and trust models trained on panel data from the open study kit; audio for video ads; and an open, growing ad eye-tracking dataset built from contributed studies.</p>

<h2>8. Conclusion</h2>
<p>Commercial attention-prediction claims are rarely comparable, and where third-party scores exist, the best open models match or beat them. A small, permissively licensed network fine-tuned on design eye tracking, MediaAttention-Ads v1, is the most accurate open model on posters, ads and interfaces by a wide margin, and stays within a few hundredths of CC of the state of the art on unseen webpages while running in tens of milliseconds. The benchmark, the toolkit and this report are open, so these claims can be checked and extended.</p>

<h2>Appendix A · Reproducibility</h2>
<pre>git clone https://github.com/pranikz/MediaAttention && cd MediaAttention
pip install -e ".[unisal]" "git+https://github.com/matthias-k/pysaliency.git@dev" pandas scikit-image
pip install torch torchvision einops clip-anytorch "deepgaze_pytorch @ git+https://github.com/matthias-k/DeepGaze" fire
python benchmarks/finetune_unisal.py --epochs 8   # reproduces MediaAttention-Ads v1
python benchmarks/run.py fiwi --models center,spectral,unisal,transalnet,deepgaze_iie,deepgaze_msdb,unisal_ueyes
python benchmarks/run.py ueyes_test --models center,spectral,unisal,transalnet,deepgaze_iie,deepgaze_msdb,unisal_ueyes
python docs/research/build.py</pre>
<p class="small">Dataset and third-party model locations are documented in <code>benchmarks/run.py</code>. Per-image scores are in <code>benchmarks/results/</code>.</p>

<h2>References</h2>
<ol class="refs">
<li>Linardos, A., Kümmerer, M., Press, O. &amp; Bethge, M. (2021). DeepGaze IIE: Calibrated prediction in and out-of-domain for state-of-the-art saliency modeling. ICCV. arXiv:2105.12441</li>
<li>Kümmerer, M., Wallis, T. S. A. &amp; Bethge, M. (2018). Saliency benchmarking made easy: Separating models, maps and metrics. ECCV. arXiv:1704.08615</li>
<li>Hou, X. &amp; Zhang, L. (2007). Saliency detection: A spectral residual approach. CVPR. doi:10.1109/CVPR.2007.383267</li>
<li>Jiang, M., Huang, S., Duan, J. &amp; Zhao, Q. (2015). SALICON: Saliency in context. CVPR. doi:10.1109/CVPR.2015.7298710</li>
<li>Judd, T., Ehinger, K., Durand, F. &amp; Torralba, A. (2009). Learning to predict where humans look. ICCV. doi:10.1109/ICCV.2009.5459462</li>
<li>Bylinskii, Z., Judd, T., Oliva, A., Torralba, A. &amp; Durand, F. (2019). What do different evaluation metrics tell us about saliency models? IEEE TPAMI 41(3). arXiv:1604.03605</li>
<li>Tatler, B. W. (2007). The central fixation bias in scene viewing. Journal of Vision 7(14). doi:10.1167/7.14.4</li>
<li>Minderer, M., Gritsenko, A. &amp; Houlsby, N. (2023). Scaling open-vocabulary object detection. NeurIPS. arXiv:2306.09683</li>
<li>Du, Y. et al. (2020). PP-OCR: A practical ultra lightweight OCR system. arXiv:2009.09941</li>
<li>Needell, C. D. &amp; Bainbridge, W. A. (2022). Embracing new techniques in deep learning for estimating image memorability. Computational Brain &amp; Behavior 5. doi:10.1007/s42113-022-00126-5</li>
<li>Khosla, A., Raju, A. S., Torralba, A. &amp; Oliva, A. (2015). Understanding and predicting image memorability at a large scale. ICCV. doi:10.1109/ICCV.2015.275</li>
<li>Isola, P., Xiao, J., Torralba, A. &amp; Oliva, A. (2011). What makes an image memorable? CVPR. doi:10.1109/CVPR.2011.5995721</li>
<li>Goetschalckx, L. &amp; Wagemans, J. (2019). MemCat: a new category-based image set quantified on memorability. PeerJ 7. doi:10.7717/peerj.8169</li>
<li>Itti, L., Koch, C. &amp; Niebur, E. (1998). A model of saliency-based visual attention for rapid scene analysis. IEEE TPAMI 20(11). doi:10.1109/34.730558</li>
<li>Shannon, C. E. (1948). A mathematical theory of communication. Bell System Technical Journal 27. doi:10.1002/j.1538-7305.1948.tb01338.x</li>
<li>Rosenholtz, R., Li, Y. &amp; Nakano, L. (2007). Measuring visual clutter. Journal of Vision 7(2). doi:10.1167/7.2.17</li>
<li>Pieters, R. &amp; Wedel, M. (2004). Attention capture and transfer in advertising: Brand, pictorial, and text-size effects. Journal of Marketing 68(2). doi:10.1509/jmkg.68.2.36.27794</li>
<li>Pieters, R., Wedel, M. &amp; Batra, R. (2010). The stopping power of advertising: Measures and effects of visual complexity. Journal of Marketing 74(5). doi:10.1509/jmkg.74.5.048</li>
<li>Milosavljevic, M., Navalpakkam, V., Koch, C. &amp; Rangel, A. (2012). Relative visual saliency differences induce sizable bias in consumer choice. Journal of Consumer Psychology 22(1). doi:10.1016/j.jcps.2011.10.002</li>
<li>Teixeira, T., Wedel, M. &amp; Pieters, R. (2010). Moment-to-moment optimal branding in TV commercials: Preventing avoidance by pulsing. Marketing Science 29(5). doi:10.1287/mksc.1100.0567</li>
<li>Lucas, B. D. &amp; Kanade, T. (1981). An iterative image registration technique with an application to stereo vision. IJCAI.</li>
<li>Shi, J. &amp; Tomasi, C. (1994). Good features to track. CVPR. doi:10.1109/CVPR.1994.323794</li>
<li>Radvansky, G. A., Krawietz, S. A. &amp; Tamplin, A. K. (2011). Walking through doorways causes forgetting: Further explorations. QJEP 64(8). doi:10.1080/17470218.2011.571267</li>
<li>Abid, A. et al. (2019). Gradio: Hassle-free sharing and testing of ML models in the wild. arXiv:1906.02569</li>
<li>Droste, R., Jiao, J. &amp; Noble, J. A. (2020). Unified image and video saliency modeling. ECCV. arXiv:2003.05477</li>
<li>Shen, C. &amp; Zhao, Q. (2014). Webpage saliency. ECCV.</li>
<li>Jiang, Y., Leiva, L. A., Rezazadegan Tavakoli, H., Houssel, P. R. B., Kylmälä, J. &amp; Oulasvirta, A. (2023). UEyes: Understanding visual saliency across user interface types. CHI. doi:10.1145/3544548.3581096</li>
<li>Kümmerer, M., Khanuja, H. S. &amp; Bethge, M. (2025). Modeling saliency dataset bias (DeepGaze MSDB). arXiv:2505.10169</li>
<li>MIT/Tübingen Saliency Benchmark, MIT300 results. saliency.tuebingen.ai/results.html, accessed 21 October 2026.</li>
<li>Neurons Inc. (2025). Neurons AI tech paper. Public document linked from neuronsinc.com/neurons-ai, accessed October 2026.</li>
<li>Ahtik, J. (2023). Using artificial intelligence for predictive eye-tracking analysis to evaluate photographs. Journal of Graphic Engineering and Design 14(1).</li>
<li>Lou, J., Lin, H., Marshall, D., Saupe, D. &amp; Liu, H. (2022). TranSalNet: Towards perceptually relevant visual saliency prediction. Neurocomputing 494. arXiv:2110.03593</li>
<li>Hosseini, A., Kazerouni, A., Akhavan, S., Brudno, M. &amp; Taati, B. (2024). SUM: Saliency unification through Mamba for visual attention modeling. arXiv:2406.17815</li>
<li>Ye, J. &amp; Wedel, M. (2026). ViASNet: A video ad saliency network for predicting dynamic saliency and viewer engagement. arXiv:2605.29302</li>
</ol>
<p class="small">Vendor information in Section 3 is from public websites, documentation and listings and may change. Vendor accuracy figures are their claims, not our measurements.</p>
</main></body></html>
"""

if __name__ == "__main__":
    sys.exit(build())
