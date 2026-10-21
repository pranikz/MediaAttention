"""Local web UI (Gradio). Needs the `ui` extra: pip install -e ".[ui]"; run: mediaattention ui"""
import json
import tempfile
from pathlib import Path

from . import score_path
from .metrics import DEFAULT_RANGES

EXAMPLES = Path(__file__).parent / "examples"

BANDS = [(7.0, "Ready to launch"), (3.5, "Optimise before launch"), (0, "Needs significant work")]  # Neurons' NIS bands


def _verdict(impact, cuts=None):
    if impact is None:
        return "## Impact score: n/a\nNo metric had a recommended range."
    label = next(text for floor, text in BANDS if impact >= floor)
    md = f"## Impact score: {impact} / 10\n**{label}**"
    if cuts:
        md += f"\n\nHard cuts at {', '.join(f'{c}s' for c in cuts)}. Avoid putting the brand or key message right after one."
    return md


def _label(key):
    """focus_start -> Focus start, cta_attention -> CTA attention."""
    s = key.replace("_", " ").replace("cta", "CTA")
    return s[0].upper() + s[1:]


def _metrics_table(report, ranges):
    import pandas as pd
    rows = [{"Metric": _label(k), "Value": v, "Score": report["impact_parts"].get(k, "–"),
             "Target": "–" if k not in ranges else f"{ranges[k][0]}–{ranges[k][1]}"} for k, v in report["metrics"].items()]
    return pd.DataFrame(rows)


def _aoi_table(report, video):
    import pandas as pd
    if video:
        return pd.DataFrame([{"AOI": _label(k), "Attention %": v["total_attention"], "While visible %": v["relative_attention"],
                              "On screen (s)": v["availability_s"]} for k, v in report["aois"].items()])
    return pd.DataFrame([{"AOI": _label(a["label"]), "Attention %": a["attention_pct"], "Area %": a["area_pct"], "Lift": a["lift"]}
                         for a in report["aois"]])


def _timeline(report):
    import pandas as pd
    keys = [k for k in ("focus", "cognitive_demand", "memory") if k in report["timeline"][0]]
    return pd.DataFrame([{"seconds": r["t"], "metric": _label(k), "score": r[k]} for r in report["timeline"] for k in keys])


def analyse(path, auto_aoi, product, model, fps, bench_file):
    """Gradio handler: returns values for every output component, hiding the image or video ones."""
    import gradio as gr
    from .render import aoi_overlay, fog_overlay, heat_overlay
    if not path:
        raise gr.Error("Upload an image or a video first.")
    ranges = json.loads(Path(bench_file).read_text()) if bench_file else DEFAULT_RANGES
    out = Path(tempfile.mkdtemp(prefix="mediaattention-"))
    report, img, heat = score_path(path, "auto" if auto_aoi else None, product or None, model, ranges, fps,
                                   heat_mp4=out / "heat.mp4")
    (out / "scores.json").write_text(json.dumps(report, indent=2))
    video = img is None
    hide = lambda c: c(visible=False)
    return (
        _verdict(report["impact_score"], report.get("cuts")),
        gr.Dataframe(_metrics_table(report, ranges), visible=True),
        gr.Dataframe(_aoi_table(report, video), visible=bool(report["aois"])),
        hide(gr.Image) if video else gr.Image(heat_overlay(img, heat), visible=True),
        hide(gr.Image) if video else gr.Image(fog_overlay(img, heat), visible=True),
        hide(gr.Image) if video or not report["aois"] else gr.Image(aoi_overlay(img, heat, report["aois"]), visible=True),
        gr.Video(str(out / "heat.mp4"), visible=True) if video else hide(gr.Video),
        gr.LinePlot(_timeline(report), visible=True) if video else hide(gr.LinePlot),
        gr.File(str(out / "scores.json"), visible=True),
    )


def calibrate(files, auto_aoi, product, model, fps, min_n, progress=None):
    """Gradio handler for the Calibrate tab: bench the uploaded files, hand benchmarks.json to the Analyse tab."""
    import gradio as gr
    import pandas as pd
    from .bench import TWO_SIDED, bench_files
    if not files:
        raise gr.Error("Upload some ads first: select several files, or a whole folder.")
    out = Path(tempfile.mkdtemp(prefix="mediaattention-bench-"))
    step = (lambda i, n, p: progress((i - 1) / n, desc=f"Scoring {Path(p).name} ({i}/{n})")) if progress else None
    ranges, n, skipped = bench_files(sorted(files), out, "auto" if auto_aoi else None, product or None, model, fps,
                                     int(min_n), on_file=step)
    if not n:
        raise gr.Error("None of the files could be scored. Are they images or videos?")
    md = f"### {n} assets scored"
    if skipped:
        md += f"\nSkipped (unreadable): {', '.join(skipped)}"
    if n < 30:
        md += f"\n\n**Only {n} assets.** Ranges get stable around 30+ per category."
    md += "\n\nThese targets are now loaded in the **Analyse** tab."
    table = pd.DataFrame([{"Metric": _label(k), "Target": f"{lo}–{hi}", "Rule": "middle 50%" if k in TWO_SIDED else "top 20%"}
                          for k, (lo, hi) in ranges.items()])
    bench_json = str(out / "benchmarks.json")
    return md, gr.Dataframe(table, visible=True), gr.File(bench_json, visible=True), bench_json


def build():
    import gradio as gr
    with gr.Blocks(title="MediaAttention") as demo:
        gr.Markdown("# MediaAttention\nPredict where people look in an ad, then score it. Runs locally; nothing is uploaded anywhere.")
        with gr.Tab("Analyse"), gr.Row():
            with gr.Column(scale=1):
                media = gr.File(label="Image or video", file_types=["image", "video"], type="filepath")
                auto_aoi = gr.Checkbox(value=True, label="Detect AOIs (brand, product, headline, body, CTA)")
                product = gr.Textbox(label="Product", placeholder="e.g. a running shoe (helps product detection)")
                with gr.Accordion("Options", open=False):
                    model = gr.Dropdown(["auto", "deepgaze", "spectral"], value="auto", label="Attention model")
                    fps = gr.Slider(0.5, 10, value=2, step=0.5, label="Video: frames scored per second")
                    bench_file = gr.File(label="benchmarks.json from `mediaattention bench` (optional)",
                                         file_types=[".json"], type="filepath")
                run = gr.Button("Analyse", variant="primary")
                gr.Examples([[str(EXAMPLES / "ad.png")], [str(EXAMPLES / "ad.mp4")]], [media], label="Try an example")
            with gr.Column(scale=2):
                verdict = gr.Markdown()
                metrics = gr.Dataframe(label="Scores", interactive=False, visible=False)
                aoi_table = gr.Dataframe(label="Areas of interest", interactive=False, visible=False)
                with gr.Row():
                    heat = gr.Image(label="Attention heatmap", visible=False)
                    fog = gr.Image(label="Fog map", visible=False)
                aois = gr.Image(label="AOIs", visible=False)
                heat_video = gr.Video(label="Attention over time", visible=False)
                timeline = gr.LinePlot(x="seconds", y="score", color="metric", label="Timeline", visible=False)
                download = gr.File(label="scores.json", visible=False)
        run.click(analyse, [media, auto_aoi, product, model, fps, bench_file],
                  [verdict, metrics, aoi_table, heat, fog, aois, heat_video, timeline, download], api_name="analyse")

        with gr.Tab("Calibrate"), gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("Upload a batch of ads from one category (industry or use case). Their scores set the "
                            "target ranges the Impact Score is measured against.")
                corpus = gr.File(label="Ads (select many files)", file_count="multiple", type="filepath",
                                 file_types=["image", "video"])
                c_aoi = gr.Checkbox(value=True, label="Detect AOIs (gives brand, product and CTA targets too)")
                c_product = gr.Textbox(label="Product", placeholder="e.g. a running shoe")
                with gr.Accordion("Options", open=False):
                    c_model = gr.Dropdown(["auto", "deepgaze", "spectral"], value="auto", label="Attention model")
                    c_fps = gr.Slider(0.5, 10, value=2, step=0.5, label="Video: frames scored per second")
                    c_min_n = gr.Number(value=5, precision=0, label="Skip metrics seen in fewer assets than")
                c_run = gr.Button("Build targets", variant="primary")
            with gr.Column(scale=2):
                c_summary = gr.Markdown()
                c_table = gr.Dataframe(label="Target ranges", interactive=False, visible=False)
                c_download = gr.File(label="benchmarks.json", visible=False)
        c_run.click(lambda *a, progress=gr.Progress(): calibrate(*a, progress=progress),
                    [corpus, c_aoi, c_product, c_model, c_fps, c_min_n],
                    [c_summary, c_table, c_download, bench_file], api_name="calibrate")
    return demo


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="mediaattention ui", description="Run the MediaAttention web UI locally.")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="create a temporary public gradio.live link")
    a = ap.parse_args(argv)
    build().queue(default_concurrency_limit=1).launch(server_port=a.port, share=a.share)  # one job at a time: one GPU
