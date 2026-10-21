import json
from pathlib import Path

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


def test_ui_calibrate(tmp_path):
    pytest.importorskip("gradio")
    from mediaattention.ui import calibrate
    rng = np.random.default_rng(0)
    files = []
    for i in range(6):
        a = np.full((120, 160, 3), 230, np.uint8)
        y, x = rng.integers(0, 80), rng.integers(0, 120)
        a[y:y + 30, x:x + 30] = 20
        Image.fromarray(a).save(tmp_path / f"ad{i}.png")
        files.append(str(tmp_path / f"ad{i}.png"))
    (tmp_path / "notes.txt").write_text("ignored")
    md, table, download, bench_json = calibrate(files + [str(tmp_path / "notes.txt")], False, "", "spectral", 2, 5)
    assert md.startswith("### 6 assets scored")
    assert "Focus" in [row[0] for row in table.value["data"]]
    assert "focus" in json.loads(Path(bench_json).read_text())


def test_unisal_input_size():
    from mediaattention.saliency import _unisal_size
    assert _unisal_size(768, 1360) == (256, 416)  # 8x13 cells, UNISAL's closest grid to 16:9
    assert _unisal_size(1000, 1000) == (320, 320)


def test_video_clip_frames_end_with_the_sample(tmp_path):
    av = pytest.importorskip("av")
    from mediaattention.video import frames
    path = tmp_path / "grey.mp4"
    with av.open(str(path), "w") as out:
        s = out.add_stream("mpeg4", rate=30)
        s.width, s.height, s.pix_fmt = 64, 64, "yuv420p"
        for i in range(60):  # 2 s
            out.mux(s.encode(av.VideoFrame.from_ndarray(np.full((64, 64, 3), 4 * i, np.uint8), format="rgb24")))
        out.mux(s.encode())
    got = list(frames(path, fps=2, clip_fps=6))
    assert len(got) == 4  # samples at 0, 0.5, 1.0, 1.5 s
    for t, img, cut, grays, clip in got[1:]:
        assert 2 <= len(clip) <= 4 and clip[-1] is img  # ~6 fps between samples, ending on the sample


def test_bench_ranges_from_kpi():
    from mediaattention.bench import ranges_from_kpi
    rng = np.random.default_rng(0)
    by_file, kpis = {}, {}
    for i in range(50):  # focus drives CTR; cognitive demand is noise
        focus = rng.uniform(40, 95)
        by_file[f"ad{i}.png"] = {"focus": focus, "cognitive_demand": rng.uniform(10, 60)}
        kpis[f"ad{i}.png"] = focus / 100 + rng.normal(0, 0.05)
    ranges, report = ranges_from_kpi(by_file, kpis)
    assert report["winners"] == 10 and ranges["focus"][0] > 70  # winners are the high-focus ads
    corr = report["correlation_with_kpi"]
    assert corr["focus"]["spearman_rho"] > 0.7 and abs(corr["cognitive_demand"]["spearman_rho"]) < 0.4
    _, low = ranges_from_kpi(by_file, {f: -v for f, v in kpis.items()}, lower_better=True)
    assert low["correlation_with_kpi"]["focus"]["spearman_rho"] > 0.7  # e.g. CPC: lower is better


def test_collect_fixations_and_implicit_scores(tmp_path):
    from mediaattention.collect import fixations, implicit_scores, import_results
    # 300 ms dwelling at (0.2, 0.3), a saccade, then 400 ms at (0.7, 0.6); 30 Hz samples
    gaze = [(t, 0.2, 0.3) for t in range(0, 300, 33)] + [(330, 0.45, 0.45)] + [(t, 0.7, 0.6) for t in range(363, 763, 33)]
    fx = fixations(gaze)
    assert len(fx) == 2 and abs(fx[0][0] - 0.2) < 0.01 and abs(fx[1][1] - 0.6) < 0.01
    # fast yeses on ad A, slow noes on ad B
    ans = {"p1": [("A", "engagement", True, 450)] * 4 + [("B", "engagement", False, 1400)] * 4}
    s = implicit_scores(ans)
    assert s["A"]["engagement"] > 50 > s["B"]["engagement"]
    result = {"format": "mediaattention-study/1", "participant": "p1", "sim": True, "calibration": None,
              "trials": [{"ad": "A", "src": "img/a.png", "gaze": gaze, "answers": [{"scale": "trust", "yes": True, "rt": 500}] * 3}],
              "recognition": [{"ad": "A", "old": True, "said_seen": True}, {"ad": "X", "old": False, "said_seen": False}]}
    (tmp_path / "r.json").write_text(json.dumps(result))
    rows, quality = import_results([tmp_path / "r.json"], tmp_path / "out")
    assert rows[0]["file"] == "a.png" and rows[0]["memory"] == 100.0 and quality[0]["fixations"] == 2
    assert (tmp_path / "out/fixations.csv").read_text().count("\n") == 3
