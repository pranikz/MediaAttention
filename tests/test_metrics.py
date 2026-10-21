import numpy as np
from PIL import Image

from mediaattention import score
from mediaattention.metrics import aoi_stats, cognitive_demand, focus, impact_score


def test_focus():
    assert focus(np.ones((50, 50))) < 1
    point = np.zeros((50, 50))
    point[25, 25] = 1
    assert focus(point) > 99


def test_cognitive_demand():
    assert cognitive_demand(np.full((64, 64), 128)) == 0
    assert cognitive_demand(np.random.default_rng(0).integers(0, 256, (64, 64))) > 90


def test_aoi_stats():
    heat = np.zeros((10, 10))
    heat[:, :5] = 1
    [left] = aoi_stats(heat, [{"label": "brand", "box": [0, 0, 0.5, 1]}])
    assert left["attention_pct"] == 100 and left["area_pct"] == 50 and left["lift"] == 2


def test_impact_score():
    total, parts = impact_score({"focus": 70, "cognitive_demand": 80}, {"focus": [60, 100], "cognitive_demand": [25, 55], "brand_attention": [10, 100]})
    assert parts["focus"] == 10 and 1 <= parts["cognitive_demand"] < 7
    assert "brand_attention" not in parts and total == round((10 + parts["cognitive_demand"]) / 2, 1)


def test_end_to_end_spectral():
    # a dark square on a plain background should draw the attention
    a = np.full((200, 300, 3), 240, np.uint8)
    a[80:120, 200:240] = 20
    heat, report = score(Image.fromarray(a), [{"label": "brand", "box": [0.6, 0.35, 0.85, 0.65]}], model="spectral")
    assert heat.shape == (200, 300)
    assert report["aois"][0]["lift"] > 2
    assert report["impact_score"] is not None
