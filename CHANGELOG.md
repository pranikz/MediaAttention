# Changelog

## 0.2.1 (2026-10-22)

**MediaAttention-Ads v1: our own attention model**
- The new default image model (`--model ads`): UNISAL fine-tuned on UEyes eye tracking of posters, ads and interfaces. Weights: [huggingface.co/pranikz/MediaAttention-Ads](https://huggingface.co/pranikz/MediaAttention-Ads).
- On held-out UEyes posters it reaches CC 0.752 vs 0.496 for the best other open model, with non-overlapping 95% CIs. It leads every UEyes category, and also improves on unseen webpages (FiWI 0.631 vs 0.616).
- Video keeps stock UNISAL's video weights; `--model unisal` selects the stock image model.

**Benchmarks**
- UEyes (CHI 2023) loaders and results for the full set and the test split. Seven models are covered, with per-image scores.
- `benchmarks/finetune_unisal.py` reproduces the model, and `benchmarks/tables.py` generates every comparison table.
- Architecture comparison and roadmap in the README, site and research report (v0.2).

## 0.2.0 (2026-10-21)

**Default model**
- UNISAL (Apache-2.0) is now the default attention model, chosen by the eye-tracking benchmark: the most accurate openly licensed model, and ~17× faster than DeepGaze IIE. Weights are downloaded from [huggingface.co/pranikz/MediaAttention](https://huggingface.co/pranikz/MediaAttention).
- DeepGaze IIE is still supported (`--model deepgaze`), installed from source because it isn't on PyPI.

**Video**
- Video uses UNISAL's recurrent video mode by default; `--per-frame` keeps frame-by-frame scoring.
- AOIs are tracked through each shot with optical flow.
- Hard cuts are detected.

**Calibration (`bench`)**
- `bench` builds target ranges from a folder of your own ads.
- `--kpi` builds them from your best-performing ads instead, and reports which metrics track the KPI.

**Scores and UI**
- New memory score (ResMem, optional, non-commercial licence).
- Web UI (`mediaattention ui`) with Analyse and Calibrate tabs.

**Research**
- Eye-tracking benchmark of six open models (`benchmarks/`).
- Research report: `docs/research/MediaAttention-report.pdf`.

## 0.1.0 (2026-10-21)

- Attention heatmap and fog map: DeepGaze IIE, with a spectral-residual fallback.
- Scores: Focus, Cognitive Demand, AOI attention and lift, Impact Score.
- Automatic AOI detection: OCR plus OWLv2.
- Video scoring.
- CLI.
