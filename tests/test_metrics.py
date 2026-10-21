import numpy as np
import pytest
from PIL import Image

from mediaattention import score
from mediaattention.detect import text_roles
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


def test_text_roles():
    # page with a big headline, a body line, a wordmark and a red button behind "SHOP NOW"
    rgb = np.full((400, 400, 3), 240, np.uint8)
    rgb[300:360, 40:240] = (200, 40, 40)
    lines = [([40, 60, 360, 120], "Run further"), ([40, 150, 360, 170], "Recycled foam and all-day comfort"),
             ([60, 315, 220, 345], "SHOP NOW"), ([300, 10, 360, 30], "NOVA")]
    cta, headline, other = text_roles(lines, rgb)
    assert cta == [[40, 300, 239, 359]]  # grown from the text out to the button edges
    assert headline == [[40, 60, 360, 120]]
    assert [t for _, t in other] == ["Recycled foam and all-day comfort", "NOVA"]


def test_video_cuts_and_aoi_metrics(tmp_path):
    av = pytest.importorskip("av")
    from mediaattention.video import score_video
    # 3 s at 10 fps: 1.5 s of a dark square on the left, then a hard cut to a light square on the right
    path = tmp_path / "cut.mp4"
    with av.open(str(path), "w") as out:
        s = out.add_stream("mpeg4", rate=10)
        s.width, s.height, s.pix_fmt = 320, 180, "yuv420p"
        for i in range(30):
            a = np.full((180, 320, 3), 235, np.uint8) if i < 15 else np.full((180, 320, 3), 25, np.uint8)
            if i < 15:
                a[60:120, 40:100] = 20
            else:
                a[60:120, 220:280] = 230
            out.mux(s.encode(av.VideoFrame.from_ndarray(a, format="rgb24")))
        out.mux(s.encode())
    report = score_video(path, [{"label": "left", "box": [0, 0, 0.5, 1]}], model="spectral", fps=2, window=0.5)
    assert report["cuts"] == [1.5]
    assert len(report["timeline"]) == 6
    left = [r["aois"]["left"] for r in report["timeline"]]
    assert min(left[:3]) > 60 and max(left[4:]) < 40  # attention follows the square across the cut
    assert report["aois"]["left"]["availability_s"] == 3.0


def test_bench_ranges():
    from mediaattention.bench import ranges_from
    rows = [{"focus": v, "cognitive_demand": v, "body_attention": v} for v in range(1, 101)] + [{"rare": 50}]
    r = ranges_from(rows)
    assert r["focus"] == [80.2, 100]  # top 20%
    assert r["cognitive_demand"] == [25.8, 75.2]  # middle 50%: moderate is best
    assert "rare" not in r  # too few assets to trust
    assert "body_attention" not in r


def test_memory_score():
    pytest.importorskip("resmem")
    from mediaattention.memory import memory
    m = memory(Image.fromarray(np.random.default_rng(0).integers(0, 256, (240, 320, 3), dtype=np.uint8)))
    assert 0 <= m <= 100


def test_track_boxes_follows_pan_and_zoom():
    pytest.importorskip("cv2")
    from mediaattention.video import track_boxes
    from PIL import Image as I
    patch = np.random.default_rng(1).integers(0, 256, (40, 40), dtype=np.uint8)
    base = np.full((180, 320), 128, np.uint8)
    base[70:110, 60:100] = patch
    # pan: the textured patch moves 4 px right per frame for 10 frames
    pan = [np.roll(base, 4 * i, axis=1) for i in range(11)]
    [a] = track_boxes([{"label": "x", "box": [60 / 320, 70 / 180, 100 / 320, 110 / 180]}], pan)
    assert abs(a["box"][0] * 320 - 100) < 3 and abs(a["box"][2] * 320 - 140) < 3
    # zoom: the frame scales up 2% per frame around its centre for 10 frames (~1.22x)
    zoom = []
    for i in range(11):
        z = 1.02 ** i
        w, h = 320 / z, 180 / z
        zoom.append(np.asarray(I.fromarray(base).crop(((320 - w) / 2, (180 - h) / 2, (320 + w) / 2, (180 + h) / 2)).resize((320, 180))))
    [a] = track_boxes([{"label": "x", "box": [60 / 320, 70 / 180, 100 / 320, 110 / 180]}], zoom)
    width = (a["box"][2] - a["box"][0]) * 320
    assert abs(width - 40 * 1.02 ** 10) < 4


def test_ui_analyse(tmp_path):
    pytest.importorskip("gradio")
    from mediaattention.ui import analyse, build
    build()  # the layout constructs
    a = np.full((200, 300, 3), 240, np.uint8)
    a[80:120, 200:240] = 20
    Image.fromarray(a).save(tmp_path / "ad.png")
    out = analyse(str(tmp_path / "ad.png"), False, "", "spectral", 2, None)
    assert out[0].startswith("## Impact score") and len(out) == 9
    assert "Focus" in [row[0] for row in out[1].value["data"]]
