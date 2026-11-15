"""Assemble report.md, class_table.csv and the demo frames into one HTML page.

Images are embedded as data URIs so the page is self-contained. Output:
<two_stage_dir>/two_stage_demo.html
"""

import base64
import html
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from spyfish.ml.classifier import artifacts  # noqa: E402
from spyfish.ml.classifier.crops import OUT_DIR  # noqa: E402

CSS = """
<title>Two-Stage Fish ID</title>
<style>
/* layout: one reading column for the method and tables, full-bleed frames below */
:root{--bg:#f6f7f4;--fg:#1d2b2a;--muted:#5d6f6d;--line:#d6ddd9;--accent:#0e7490;--ok:#1f9d55;--bad:#c53030;--panel:#ffffff;
 --display:"Fraunces",Georgia,serif;--body:"Source Sans 3","Helvetica Neue",Arial,sans-serif;--mono:"JetBrains Mono",Menlo,monospace}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0f1716;--fg:#e4ebe9;--muted:#9fb0ad;--line:#2a3836;--accent:#5fc3dc;--ok:#4ade80;--bad:#f87171;--panel:#162120;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0f1716;--fg:#e4ebe9;--muted:#9fb0ad;--line:#2a3836;--accent:#5fc3dc;--ok:#4ade80;--bad:#f87171;--panel:#162120;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font-family:var(--body);font-size:17px;line-height:1.55;padding-block:32px;padding-inline:clamp(16px,4vw,48px)}
main{max-width:1200px;margin:0 auto;display:grid;gap:40px}
h1{font-family:var(--display);font-weight:600;font-size:clamp(30px,4vw,44px);line-height:1.1;margin:0;text-wrap:balance}
h2{font-family:var(--display);font-weight:500;font-size:26px;margin:0 0 12px;text-wrap:balance}
h3{font-size:18px;margin:24px 0 8px}
.lede{color:var(--muted);max-width:68ch;margin:8px 0 0}
.kpis{display:flex;flex-wrap:wrap;gap:14px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 16px;min-width:150px}
.kpi b{display:block;font-size:28px;font-variant-numeric:tabular-nums;font-family:var(--display);font-weight:600}
.kpi span{color:var(--muted);font-size:14px;text-transform:uppercase;letter-spacing:.04em}
.tbl{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:15px;font-variant-numeric:tabular-nums}
th,td{padding:7px 12px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:600;font-size:13px;text-transform:uppercase;letter-spacing:.04em}
tr:last-child td{border-bottom:none}
.frames{display:grid;gap:28px}
figure{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden}
figure img{display:block;width:100%;max-width:100%}
figcaption{padding:10px 14px;font-size:15px;color:var(--muted)}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:15px;color:var(--muted)}
.sw{display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:-2px;margin-right:6px}
code{font-family:var(--mono);font-size:.9em}
.note{border-left:3px solid var(--accent);padding:8px 14px;color:var(--muted);max-width:72ch}
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Source+Sans+3:wght@400;600&display=swap">
"""


def md_table_to_html(block: str) -> str:
    lines = [ln for ln in block.strip().splitlines() if ln.strip()]
    head = [c.strip() for c in lines[0].strip("|").split("|")]
    body = [[c.strip() for c in ln.strip("|").split("|")] for ln in lines[2:]]
    out = (
        ["<div class='tbl'><table><thead><tr>"]
        + [f"<th>{html.escape(h)}</th>" for h in head]
        + ["</tr></thead><tbody>"]
    )
    for row in body:
        out.append(
            "<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>"
        )
    out.append("</tbody></table></div>")
    return "".join(out)


def render_report(md: str) -> str:
    parts = []
    for chunk in re.split(r"\n(?=## )", md):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.startswith("# "):
            continue  # page has its own h1
        m = re.match(r"## ([^\n]*)\n(.*)", chunk, re.S)
        title, rest = m.group(1), m.group(2)
        table = re.search(r"(\|.*\n)+", rest)
        text = rest[: table.start()].strip() if table else rest.strip()
        parts.append(f"<section><h2>{html.escape(title)}</h2>")
        if text:
            parts.append(f"<p class='lede'>{html.escape(text)}</p>")
        if table:
            parts.append(md_table_to_html(table.group(0)))
        parts.append("</section>")
    return "\n".join(parts)


def extra_reports() -> str:
    """Fine-tune and YOLO+prior reports, when their scripts have run."""
    parts = []
    for rel in (
        "finetune/finetune_report.md",
        "yolo_prior_report.md",
        "negatives/negatives_report.md",
    ):
        path = OUT_DIR / rel
        if not path.exists():
            continue
        text = path.read_text()
        title = re.match(r"# (.*)", text).group(1)
        intro = re.search(r"^# .*\n\n(.*?)\n\n", text, re.S)
        parts.append(f"<section><h2>{html.escape(title)}</h2>")
        if intro and not intro.group(1).startswith("##"):
            parts.append(f"<p class='lede'>{html.escape(intro.group(1))}</p>")
        parts.append(
            render_report(text).replace("<h2>", "<h3>").replace("</h2>", "</h3>")
        )
        if rel.startswith("negatives"):
            parts.append(
                "<h3>Two of the sixteen review sheets</h3><p class='lede'>Cluster 3 is mostly water, sediment, rope and the bait bag. "
                "Cluster 2 is mostly real fish the expert did not box. The review is one decision per cluster, not per box.</p>"
            )
            for sheet in ("cluster_03.jpg", "cluster_02.jpg"):
                sp = OUT_DIR / "negatives" / "sheets" / sheet
                if sp.exists():
                    b64 = base64.b64encode(sp.read_bytes()).decode()
                    parts.append(
                        f"<figure><img src='data:image/jpeg;base64,{b64}' alt='{sheet}'><figcaption>{sheet}</figcaption></figure>"
                    )
        parts.append("</section>")
    return "\n".join(parts)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="one HTML page from a two-stage report"
    )
    parser.add_argument(
        "--report",
        default="report.md",
        help="markdown under OUT_DIR (UoA: uoa_report.md)",
    )
    parser.add_argument("--out", default="two_stage_demo.html")
    parser.add_argument("--title", default="Two-stage fish identification, first look")
    args = parser.parse_args()
    md = (OUT_DIR / args.report).read_text()
    m = re.search(r"^# .*\n\n(.*?)\n\n", md, re.S)
    intro = m.group(1) if m else ""
    # The demo frames and their KPI strip exist for the DOC run only; a report
    # without them (UoA) renders as tables alone.
    demo_csv = OUT_DIR / artifacts.DEMO_DIR / artifacts.DEMO_BOXES
    kpis, figs = [], []
    if demo_csv.exists():
        demo = pd.read_csv(demo_csv)
        scored = demo.dropna(subset=["expert"])
        kpis = [
            (len(scored), "detected boxes with an expert label"),
            (f"{scored['yolo_ok'].astype(float).mean():.0%}", "current YOLO right"),
            (f"{scored['two_stage_ok'].astype(float).mean():.0%}", "two-stage right"),
            (demo["filename"].nunique(), "demo frames"),
        ]
        for img in sorted((OUT_DIR / artifacts.DEMO_DIR).glob("*.jpg")):
            b64 = base64.b64encode(img.read_bytes()).decode()
            rows = demo[demo["filename"] == img.stem + ".jpg"]
            cap = f"{rows['drop'].iloc[0]} · {len(rows)} detections · YOLO right on {int(rows['yolo_ok'].eq(True).sum())}, two-stage right on {int(rows['two_stage_ok'].eq(True).sum())} of {rows['expert'].notna().sum()} expert-matched boxes"
            figs.append(
                f"<figure><img src='data:image/jpeg;base64,{b64}' alt='{html.escape(img.stem)}'><figcaption>{html.escape(cap)}</figcaption></figure>"
            )
    kpi_section = (
        "<section class='kpis'>"
        + "".join(f"<div class='kpi'><b>{v}</b><span>{k}</span></div>" for v, k in kpis)
        + "</section>"
        if kpis
        else ""
    )
    frames_section = (
        "<section><h2>What each frame shows</h2>"
        "<p class='legend'><span><i class='sw' style='background:#28c85a'></i>two-stage agrees with the expert</span>"
        "<span><i class='sw' style='background:#e63c3c'></i>two-stage disagrees</span>"
        "<span><i class='sw' style='background:#787878'></i>no expert box there</span>"
        "<span><i class='sw' style='background:#fff;border:1px solid #888'></i>expert box and label</span></p>"
        "<p class='note'>The species stage for each frame was trained on every other drop, so it has never seen that "
        "deployment's background or fish. The detector is the current YOLO model; only its boxes are used, its labels "
        "are shown for comparison.</p>"
        f"<div class='frames'>{''.join(figs)}</div></section>"
        if figs
        else ""
    )
    extra = extra_reports() if args.report == "report.md" else ""
    page = f"""{CSS}
<main>
<header>
<h1>{html.escape(args.title)}</h1>
<p class="lede">{html.escape(intro)}</p>
</header>
{kpi_section}
{frames_section}
{render_report(md)}
{extra}
</main>
"""
    out = OUT_DIR / args.out
    out.write_text(page)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
