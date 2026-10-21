---
title: MediaAttention
emoji: 👁️
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Predict where people look in an ad, then score it
---

# MediaAttention

Predicts where viewers look in an image or video ad and scores it for focus, cognitive demand and attention on the brand, product, headline and call to action, plus a 1–10 impact score. Use the **Analyse** tab for one creative, and **Calibrate** to build target ranges from a batch of your own ads.

Runs on free CPU hardware, so expect ~20–40 s per image and longer for video. For speed, run it locally on a GPU or Apple Silicon:

```bash
pip install "mediaattention[deepgaze,aoi,video,ui] @ git+https://github.com/pranikz/MediaAttention"
mediaattention ui
```

- Docs and references: https://pranikz.github.io/MediaAttention/
- Source: https://github.com/pranikz/MediaAttention

Built by [Pratyush Mahapatra](https://pranikz.dev). The memory score (ResMem, non-commercial licence) is not enabled in this demo.
