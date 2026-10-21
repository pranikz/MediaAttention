import argparse
import json
from pathlib import Path

from PIL import Image

from . import score


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mediaattention", description="Predict where people look in an image or video and score it.")
    ap.add_argument("media", help="image or video file")
    ap.add_argument("--aoi", help="AOI JSON file ([{label, box:[x0,y0,x1,y1]}] in 0-1 coords), or 'auto'")
    ap.add_argument("--product", help="what the product is, for --aoi auto (e.g. 'a running shoe')")
    ap.add_argument("--model", choices=["auto", "deepgaze", "spectral"], default="auto")
    ap.add_argument("--benchmarks", help='JSON {"metric": [lo, hi]} recommended ranges')
    ap.add_argument("--fps", type=float, default=2.0, help="video: frames scored per second")
    ap.add_argument("--out", default="mediaattention-out")
    a = ap.parse_args(argv)

    aois = json.loads(Path(a.aoi).read_text()) if a.aoi and a.aoi != "auto" else a.aoi
    ranges = json.loads(Path(a.benchmarks).read_text()) if a.benchmarks else None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from .video import VIDEO_EXT
    if Path(a.media).suffix.lower() in VIDEO_EXT:
        from .video import score_video
        report = score_video(a.media, aois, a.product, a.model, ranges, a.fps, heat_mp4=out / "heat.mp4")
        (out / "scores.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k != "timeline"}, indent=2))
        return

    img = Image.open(a.media)
    if aois == "auto":
        from .detect import detect
        aois = detect(img, product=a.product)
    heat, report = score(img, aois, a.model, ranges)

    from .render import aoi_overlay, fog_overlay, heat_overlay
    heat_overlay(img, heat).save(out / "heat.png")
    fog_overlay(img, heat).save(out / "fog.png")
    if report["aois"]:
        aoi_overlay(img, heat, report["aois"]).save(out / "aois.png")
    (out / "scores.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
