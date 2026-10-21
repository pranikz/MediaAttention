# MediaAttention

Open-source predicted attention for ads and media. Give it a creative and it predicts where people will look, then scores it:

- **Attention heatmap and fog map**: where the eye lands in the first seconds
- **Focus** (0–100): attention on a few elements (high) vs. scattered everywhere (low)
- **Cognitive Demand** (0–100): visual complexity, measured as local Shannon entropy
- **AOI attention**: % of attention on brand, product, headline and CTA, plus *lift* (attention share ÷ area share)
- **Impact Score** (1–10): each metric is scored against a recommended range, then averaged

It's an open alternative to proprietary "neuro-prediction" tools. The methods are documented and the numbers are reproducible.

## Install

```bash
pip install -e .                 # core: spectral-residual fallback, no ML deps
pip install -e ".[deepgaze]"     # DeepGaze IIE attention model (recommended)
pip install -e ".[aoi]"          # automatic AOI detection (OWLv2)
```

## Use

```bash
mediaattention ad.png --aoi auto --out report/
```

This writes `report/heat.png`, `report/fog.png` and `report/scores.json`:

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
| AOI attention % | Sum of heatmap mass inside the box. |
| Impact Score | 10 inside the recommended range; outside, it falls linearly from 7 at the edge to 1 at the end of the scale. The final score is the mean over available metrics, and missing AOIs are skipped. |

## Honest limitations

- The attention model is trained on public natural-image eye-tracking data (SALICON, MIT1003), not ads. Expect it to be weaker on text-heavy creatives than an ad-specific model.
- **The default recommended ranges are placeholders.** The Impact Score only means something once you calibrate the ranges on a corpus. The intended method is the top 20% of scores per industry/use case.
- No emotion, engagement or trust scores yet. These need human response data, which nobody has released openly so far (see the roadmap).

## Roadmap

- [ ] Video: per-frame scoring, first/last-2s windows, scene-cut detection
- [ ] `mediaattention bench <folder>`: build recommended ranges from your own corpus
- [ ] Memory score (ResMem)
- [ ] Open ad eye-tracking benchmark (CC / SIM / KLD) to publish real accuracy numbers
- [ ] Open data-collection kit: webcam eye-tracking plus timed Yes/No response tests, so the community can build ad-specific and emotion models

## Licenses

The code is Apache-2.0. Model weights are downloaded at runtime and keep their own terms:

- DeepGaze IIE code is MIT; its weights are trained on SALICON/MIT1003.
- OWLv2 is Apache-2.0.

Check the dataset terms before any commercial use.
