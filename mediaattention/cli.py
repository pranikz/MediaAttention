import argparse
import json
import sys
from pathlib import Path

from . import score_path


def bench_main(argv):
    ap = argparse.ArgumentParser(prog="mediaattention bench",
                                 description="Score a folder of creatives and derive recommended ranges (benchmarks.json).")
    ap.add_argument("folder", help="one folder per category (industry / use case)")
    ap.add_argument("--aoi", choices=["auto"], help="detect AOIs so AOI metrics get ranges too")
    ap.add_argument("--product", help="what the product is, for --aoi auto")
    ap.add_argument("--model", choices=["auto", "unisal", "deepgaze", "spectral"], default="auto")
    ap.add_argument("--fps", type=float, default=2.0, help="video: frames scored per second")
    ap.add_argument("--min-n", type=int, default=5, help="skip metrics seen in fewer assets than this")
    ap.add_argument("--out", default="mediaattention-bench")
    a = ap.parse_args(argv)
    from .bench import bench
    ranges, n, skipped = bench(a.folder, a.out, aois=a.aoi, product=a.product, model=a.model, fps=a.fps, min_n=a.min_n)
    print(json.dumps({"assets": n, "skipped": skipped, "ranges": ranges}, indent=2))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["bench"]:
        return bench_main(argv[1:])
    if argv[:1] == ["ui"]:
        from .ui import main as ui_main
        return ui_main(argv[1:])
    ap = argparse.ArgumentParser(prog="mediaattention", description="Predict where people look in an image or video and score it.")
    ap.add_argument("media", help="image or video file")
    ap.add_argument("--aoi", help="AOI JSON file ([{label, box:[x0,y0,x1,y1]}] in 0-1 coords), or 'auto'")
    ap.add_argument("--product", help="what the product is, for --aoi auto (e.g. 'a running shoe')")
    ap.add_argument("--model", choices=["auto", "unisal", "deepgaze", "spectral"], default="auto")
    ap.add_argument("--benchmarks", help='JSON {"metric": [lo, hi]} recommended ranges')
    ap.add_argument("--fps", type=float, default=2.0, help="video: frames scored per second")
    ap.add_argument("--out", default="mediaattention-out")
    a = ap.parse_args(argv)

    aois = json.loads(Path(a.aoi).read_text()) if a.aoi and a.aoi != "auto" else a.aoi
    ranges = json.loads(Path(a.benchmarks).read_text()) if a.benchmarks else None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    report, img, heat = score_path(a.media, aois, a.product, a.model, ranges, a.fps, heat_mp4=out / "heat.mp4")
    if img is None:  # video
        (out / "scores.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k != "timeline"}, indent=2))
        return

    from .render import aoi_overlay, fog_overlay, heat_overlay
    heat_overlay(img, heat).save(out / "heat.png")
    fog_overlay(img, heat).save(out / "fog.png")
    if report["aois"]:
        aoi_overlay(img, heat, report["aois"]).save(out / "aois.png")
    (out / "scores.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
