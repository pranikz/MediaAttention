# MediaAttention

[![PyPI](https://img.shields.io/pypi/v/mediaattention)](https://pypi.org/project/mediaattention/) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23540409.svg)](https://doi.org/10.5281/zenodo.23540409)

**Site:** https://pranikz.github.io/MediaAttention/ · **Model on Hugging Face:** https://huggingface.co/pranikz/MediaAttention · **Research report (PDF):** [docs/research/MediaAttention-report.pdf](docs/research/MediaAttention-report.pdf)

Open-source predicted attention for ads and media. Give it an image or video creative and it predicts where people will look, then scores it:

- **Attention heatmap and fog map**: where the eye lands in the first seconds
- **Focus** (0–100): attention on a few elements (high) vs. scattered everywhere (low)
- **Cognitive Demand** (0–100): visual complexity, measured as local Shannon entropy
- **Memory** (0–100, optional): predicted share of viewers who would recognise the creative after a brief exposure
- **AOI attention**: % of attention on brand, product, headline and CTA, plus *lift* (attention share ÷ area share)
- **Impact Score** (1–10): each metric is scored against a recommended range, then averaged

It's an open alternative to proprietary "neuro-prediction" tools. The methods are documented and the numbers are reproducible.

## Install

```bash
pip install "mediaattention[all]"   # UNISAL model, AOI detection, video, web UI (recommended)
pip install mediaattention          # core only: spectral-residual fallback, no ML dependencies
pip install "mediaattention[memory]"   # + memory score (ResMem). NON-COMMERCIAL license, see Licenses
```

Or from source: `git clone https://github.com/pranikz/MediaAttention && pip install -e ".[all]"`. The extras can also be installed one at a time: `unisal`, `aoi`, `video`, `ui`, `memory`.

DeepGaze IIE (optional, `--model deepgaze`) isn't on PyPI. Install it from source:

```bash
pip install torch torchvision einops clip-anytorch "deepgaze_pytorch @ git+https://github.com/matthias-k/DeepGaze"
```

When ResMem is installed, `memory` is added to every report automatically. Videos also get `memory_start` / `memory_end`.

## Use

### Web UI

```bash
mediaattention ui        # opens http://localhost:7860
```

Drop in an image or video (or click one of the built-in examples), optionally name the product, and hit **Analyse**. You get:

- the Impact Score with a verdict
- every metric with its 1–10 score and target range
- the AOI table
- heatmap, fog map and AOI overlay for images; heatmap video and a Focus/Cognitive Demand/Memory timeline for videos
- `scores.json` to download

The **Calibrate** tab does `bench` in the browser: upload a batch of ads from one category (select many files) and it builds the target ranges, shows them, and loads them into the Analyse tab. You can also download that `benchmarks.json`, or upload one under *Options* in the Analyse tab. Everything runs on your machine. `--share` creates a temporary public link if you want to show someone.

### Command line

```bash
mediaattention ad.png --aoi auto --product "a running shoe" --out report/
```

This writes `report/heat.png`, `report/fog.png`, `report/aois.png` (heatmap with each AOI boxed and its attention %) and `report/scores.json`:

```json
{
  "model": "unisal",
  "metrics": {"focus": 64.2, "cognitive_demand": 38.5, "brand_attention": 12.1},
  "aois": [{"label": "brand", "box": [0.05, 0.04, 0.3, 0.14], "attention_pct": 12.1, "area_pct": 3.5, "lift": 3.46}],
  "impact_score": 10.0,
  "impact_parts": {"focus": 10.0, "cognitive_demand": 10.0, "brand_attention": 10.0}
}
```

The example values are illustrative.

### Calibrate the Impact Score on your own ads

The default recommended ranges are placeholders. Build real ones from a folder of creatives, ideally one folder per industry or use case with 30+ assets each:

```bash
mediaattention bench ads/fmcg-social/ --aoi auto --out bench/
mediaattention new_ad.png --aoi auto --benchmarks bench/benchmarks.json
```

`bench` scores every image and video in the folder and writes `assets.jsonl` (one line of metrics per asset) and `benchmarks.json`:

- **Higher-is-better metrics** (Focus, attention on brand, product, headline and CTA): the recommended range is the top 20%.
- **Cognitive Demand:** the middle 50%, because moderate complexity works best.
- **Body-text attention:** excluded, because it doesn't predict results.

**Have campaign results?** Pass them in and the targets come from your best-*performing* ads instead of your best-*scoring* ones:

```bash
mediaattention bench ads/fmcg-social/ --aoi auto --kpi results.csv --kpi-column ctr --out bench/
```

- **Format:** `results.csv` has a `file` column (the ad's filename, or its path inside the folder) plus KPI columns such as `ctr`, `recall` or `cpc`. Add `--kpi-lower-better` for costs like CPC or CPA.
- **Targets:** each metric's target becomes the middle 50% of what your top 20% of ads by that KPI scored.
- **`kpi_report.json`:** shows which metrics actually track the KPI for your ads (Spearman ρ and p-value).
- **Web UI:** the Calibrate tab takes the same CSV.
- **Size:** you need at least 25 ads with a KPI, so there are 5 winners.

### Video

```bash
mediaattention ad.mp4 --aoi auto --product "a running shoe" --fps 2 --out report/
```

This writes `report/heat.mp4` (the heatmap and AOIs on every scored frame) and `report/scores.json`. The JSON has a summary plus a per-frame `timeline`:

- Attention comes from UNISAL's **video mode**: a recurrent model fed ~6 frames per second, so attention carries over from frame to frame as in real viewing. `--per-frame` scores frames independently instead (any `--model`), averaged over the last second.
- `cuts`: hard scene cuts in seconds. Key messages placed right after a cut are easily forgotten.
- `focus_start` / `focus_end`: Focus over the first and last 2 seconds.
- Per AOI: `total_attention` (share over the whole video), `relative_attention` (share while on screen) and `availability_s` (seconds on screen).

You can also pass your own AOIs (`--aoi aois.json`, as `[{"label": "brand", "box": [x0, y0, x1, y1]}]` in 0–1 coords) and your own recommended ranges (`--benchmarks ranges.json`, as `{"focus": [60, 100]}`).

From Python:

```python
from PIL import Image
from mediaattention import score
heat, report = score(Image.open("ad.png"))  # model="unisal" by default; "deepgaze" or "spectral" also work
```

## How each number is computed

| Metric | Method |
|---|---|
| Heatmap | UNISAL (Droste et al., ECCV 2020) with its SALICON weights, downloaded on first use from [pranikz/MediaAttention](https://huggingface.co/pranikz/MediaAttention). Optional: DeepGaze IIE with a Gaussian centre-bias prior. Fallback: spectral residual (Hou & Zhang 2007). |
| Focus | Share of the image needed to hold 80% of predicted attention, mapped to 0–100 (an attention-dispersion measure). |
| Cognitive Demand | Mean Shannon entropy of 16×16 blocks of the grayscale image (32 levels), normalized to 0–100. |
| Memory | ResMem (Needell & Bainbridge 2022), trained on LaMem/MemCat recognition data. It predicts the share of people who recognise an image when it repeats in a rapid stream, the same game Neurons' memory model is built on. Scores cluster high (a plain grey image gets ~81, ads ~90–98), so compare within a `bench` corpus rather than reading absolute values. Videos are scored per frame, without audio. |
| AOI detection | OCR (RapidOCR) reads every text line. A short line on a solid colour patch is the CTA, and the box grows out to the button's edges. The largest text is the headline, and the rest is body. Logo and product come from OWLv2 zero-shot detection (`--product` says what to look for). If OWLv2 is unsure about the logo, a short text line in the top or bottom band is used instead. |
| AOI attention % | Sum of heatmap mass inside the box. |
| Video | Frames are sampled at `--fps` for scoring. With UNISAL, attention comes from its recurrent video mode (DHF1K weights), fed at ~6 fps with its state carried across the whole video. With `--per-frame` or an image model, each sampled frame is scored on its own and heatmaps are averaged over the last second, so attention lingers briefly after a cut. A cut is a jump of more than 40% in the grayscale histogram between consecutive frames, with a 0.5 s minimum shot length. Auto AOIs are detected at the start of each shot, then tracked through every frame with Lucas–Kanade optical flow. Each box gets its own shift and zoom, so camera moves and moving objects are both followed. AOIs from a JSON file stay fixed for the whole video. |
| Impact Score | 10 inside the recommended range; outside, it falls linearly from 7 at the edge to 1 at the end of the scale. The final score is the mean over available metrics, and missing AOIs are skipped. |

## Benchmark

How close are the predictions to where real people look? We scored six open attention models against human eye tracking on **FiWI** (Shen & Zhao, ECCV 2014): 149 webpages, 11 viewers, 5 s free viewing. None of the models was trained on it. All metrics were computed with pysaliency on an Apple M3 Pro. The method and the script are in [`benchmarks/`](benchmarks/). Rebuild the tables with `python benchmarks/table.py fiwi`.

| Model | CC ↑ | NSS ↑ | sAUC ↑ | AUC ↑ | SIM ↑ | KLD ↓ | s / image (M3 Pro) | Licence |
|---|---|---|---|---|---|---|---|---|
| DeepGaze MSDB | 0.676 | 1.54 | 0.748 | 0.840 | 0.642 | 0.462 | 2.90 | no licence file |
| DeepGaze IIE | 0.649 | 1.50 | 0.729 | 0.824 | 0.629 | 0.504 | 0.67 | no licence file |
| **UNISAL (default)** | 0.616 | 1.31 | 0.661 | 0.796 | 0.613 | 0.575 | 0.04 | Apache-2.0 |
| TranSalNet-Dense | 0.608 | 1.25 | 0.646 | 0.788 | 0.605 | 0.603 | 0.10 | MIT |
| Centre-bias baseline | 0.345 | 0.62 | 0.500 | 0.676 | 0.476 | 0.899 | 0.01 | – |
| Spectral residual (fallback) | 0.209 | 0.40 | 0.562 | 0.633 | 0.449 | 1.052 | 0.00 | – |

Correlation (CC) with human attention, by page type:

| Model | Mixed | Pictorial | Text |
|---|---|---|---|
| DeepGaze MSDB | 0.686 | 0.751 | 0.592 |
| DeepGaze IIE | 0.680 | 0.726 | 0.541 |
| **UNISAL (default)** | 0.641 | 0.676 | 0.531 |
| TranSalNet-Dense | 0.642 | 0.650 | 0.532 |

What this means:

- **UNISAL is the default.** It is the most accurate model with a permissive licence, and by far the fastest: 0.04 s per image, ~17× faster than DeepGaze IIE.
- **DeepGaze MSDB is the most accurate** on every metric, ~0.06 CC ahead, but 70× slower and with no licence file. DeepGaze IIE stays available as an option (`--model deepgaze`).
- **Text-heavy layouts are the hardest case for every model.** This is where ad-specific training data would help most.

The full write-up, including a review of 15 commercial tools and what their accuracy claims actually measure, is in the [research report (PDF)](docs/research/MediaAttention-report.pdf). An ad and poster benchmark on UEyes (CHI 2023) is in progress.

## Honest limitations

- The attention model is trained on public natural-image eye-tracking data (SALICON; DHF1K for video), not ads. Expect it to be weaker on text-heavy creatives than an ad-specific model.
- **The default recommended ranges are placeholders.** Run `bench` on your own ads to calibrate them. Without `--kpi`, `bench` ranks assets by the metric itself, not by real results. Pass campaign KPIs (CTR, recall) when you have them.
- AOI detection relies on layout rules. Ads that break them (a headline smaller than the body text, an outlined ghost button, a logo in the middle of the ad) need `--aoi aois.json`.
- **Video mode is not yet validated on ad video.** Published results on TV ads favour video models over frame-by-frame prediction (CC 0.56 vs 0.23, Ye & Wedel 2026), but UNISAL's video mode hasn't been benchmarked on ad eye tracking here. Use `--per-frame` to compare. Video has no audio model.
- No emotion, engagement or trust scores yet. These need human response data, which nobody has released openly so far (see the roadmap).

## Roadmap

- [x] Video: per-frame scoring, first/last-2s windows, scene-cut detection
- [x] Video: track AOIs within a shot
- [x] `mediaattention bench <folder>`: build recommended ranges from your own corpus
- [x] `bench --kpi`: targets from your best-performing ads, plus which metrics track the KPI
- [x] Memory score (ResMem)
- [ ] Commercial-friendly memory model (retrain on openly licensed data)
- [x] Eye-tracking benchmark of the open models (FiWI webpages; see Benchmark)
- [ ] Ad and poster benchmark on UEyes (in progress)
- [x] Permissive default model: UNISAL (Apache-2.0), chosen by the benchmark
- [ ] Open data-collection kit: webcam eye-tracking plus timed Yes/No response tests, so the community can build ad-specific and emotion models

## Licenses

The code is Apache-2.0. Model weights are downloaded at runtime and keep their own terms:

- **UNISAL** (default) is Apache-2.0. Its model code is vendored in `mediaattention/unisal/` with its licence, and its weights are redistributed unchanged on Hugging Face with a NOTICE.
- **DeepGaze IIE has no license file** in its repository, which by default means all rights reserved. It's an optional extra and is downloaded from the authors at runtime, never redistributed here. Ask the authors before commercial use. Its weights are trained on SALICON/MIT1003.
- OWLv2, RapidOCR and OpenCV are Apache-2.0. PyAV is BSD-3-Clause.
- **ResMem is non-commercial only** (University of Chicago license). Research use, including inside a company, is fine. Using it in a commercial product needs permission from the Brain Bridge Lab. It's an optional extra, so nothing else depends on it.

Check the dataset terms before any commercial use.
