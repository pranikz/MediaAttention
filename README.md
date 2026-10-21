# MediaAttention

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
pip install -e .                 # core: spectral-residual fallback, no ML deps
pip install -e ".[deepgaze]"     # DeepGaze IIE attention model (recommended)
pip install -e ".[aoi]"          # automatic AOI detection (OCR + OWLv2)
pip install -e ".[video]"        # video support (PyAV)
pip install -e ".[memory]"       # memory score (ResMem). NON-COMMERCIAL license, see Licenses
```

When ResMem is installed, `memory` is added to every report automatically. Videos also get `memory_start` / `memory_end`.

## Use

```bash
mediaattention ad.png --aoi auto --product "a running shoe" --out report/
```

This writes `report/heat.png`, `report/fog.png`, `report/aois.png` (heatmap with each AOI boxed and its attention %) and `report/scores.json`:

```json
{
  "model": "deepgaze",
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

### Video

```bash
mediaattention ad.mp4 --aoi auto --product "a running shoe" --fps 2 --out report/
```

This writes `report/heat.mp4` (the heatmap and AOIs on every scored frame) and `report/scores.json`. The JSON has a summary plus a per-frame `timeline`:

- `cuts`: hard scene cuts in seconds. Key messages placed right after a cut are easily forgotten.
- `focus_start` / `focus_end`: Focus over the first and last 2 seconds.
- Per AOI: `total_attention` (share over the whole video), `relative_attention` (share while on screen) and `availability_s` (seconds on screen).

You can also pass your own AOIs (`--aoi aois.json`, as `[{"label": "brand", "box": [x0, y0, x1, y1]}]` in 0–1 coords) and your own recommended ranges (`--benchmarks ranges.json`, as `{"focus": [60, 100]}`).

From Python:

```python
from PIL import Image
from mediaattention import score
heat, report = score(Image.open("ad.png"), model="deepgaze")
```

## How each number is computed

| Metric | Method |
|---|---|
| Heatmap | DeepGaze IIE with a Gaussian center-bias prior. Fallback: spectral residual (Hou & Zhang 2007). |
| Focus | Share of the image needed to hold 80% of predicted attention, mapped to 0–100 (an attention-dispersion measure). |
| Cognitive Demand | Mean Shannon entropy of 16×16 blocks of the grayscale image (32 levels), normalized to 0–100. |
| Memory | ResMem (Needell & Bainbridge 2022), trained on LaMem/MemCat recognition data. It predicts the share of people who recognise an image when it repeats in a rapid stream, the same game Neurons' memory model is built on. Scores cluster high (a plain grey image gets ~81, ads ~90–98), so compare within a `bench` corpus rather than reading absolute values. Videos are scored per frame, without audio. |
| AOI detection | OCR (RapidOCR) reads every text line. A short line on a solid colour patch is the CTA, and the box grows out to the button's edges. The largest text is the headline, and the rest is body. Logo and product come from OWLv2 zero-shot detection (`--product` says what to look for). If OWLv2 is unsure about the logo, a short text line in the top or bottom band is used instead. |
| AOI attention % | Sum of heatmap mass inside the box. |
| Video | Frames are sampled at `--fps`. Each heatmap is averaged over the last second, so attention lingers briefly after a cut, as eye-tracking shows. A cut is a jump of more than 40% in the grayscale histogram between consecutive frames, with a 0.5 s minimum shot length. Auto AOIs are detected at the start of each shot, then tracked through every frame with Lucas–Kanade optical flow. Each box gets its own shift and zoom, so camera moves and moving objects are both followed. AOIs from a JSON file stay fixed for the whole video. |
| Impact Score | 10 inside the recommended range; outside, it falls linearly from 7 at the edge to 1 at the end of the scale. The final score is the mean over available metrics, and missing AOIs are skipped. |

## Honest limitations

- The attention model is trained on public natural-image eye-tracking data (SALICON, MIT1003), not ads. Expect it to be weaker on text-heavy creatives than an ad-specific model.
- **The default recommended ranges are placeholders.** Run `bench` on your own ads to calibrate them. `bench` ranks assets by the metric itself, not by real results; once you have campaign KPIs (CTR, recall), rank by those instead.
- AOI detection relies on layout rules. Ads that break them (a headline smaller than the body text, an outlined ghost button, a logo in the middle of the ad) need `--aoi aois.json`.
- No emotion, engagement or trust scores yet. These need human response data, which nobody has released openly so far (see the roadmap).

## Roadmap

- [x] Video: per-frame scoring, first/last-2s windows, scene-cut detection
- [x] Video: track AOIs within a shot
- [x] `mediaattention bench <folder>`: build recommended ranges from your own corpus
- [ ] `bench` ranked by real KPIs (CTR, recall) when you have them
- [x] Memory score (ResMem)
- [ ] Commercial-friendly memory model (retrain on openly licensed data)
- [ ] Open ad eye-tracking benchmark (CC / SIM / KLD) to publish real accuracy numbers
- [ ] Open data-collection kit: webcam eye-tracking plus timed Yes/No response tests, so the community can build ad-specific and emotion models

## Licenses

The code is Apache-2.0. Model weights are downloaded at runtime and keep their own terms:

- DeepGaze IIE code is MIT; its weights are trained on SALICON/MIT1003.
- OWLv2, RapidOCR and OpenCV are Apache-2.0. PyAV is BSD-3-Clause.
- **ResMem is non-commercial only** (University of Chicago license). Research use, including inside a company, is fine. Using it in a commercial product needs permission from the Brain Bridge Lab. It's an optional extra, so nothing else depends on it.

Check the dataset terms before any commercial use.
