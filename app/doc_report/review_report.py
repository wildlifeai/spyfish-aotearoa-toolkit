"""The Survey review as one HTML file. No Streamlit in here.

The same survey the page shows (`survey_review.py`), written as a single file
with every image inside it, to send to someone without the dashboard. It takes
what the page has already worked out (the deployment table, MaxN per deployment
from its best source, the exported frames) and only writes markup, so the page
and the file cannot disagree about a number.

The species figure is inline SVG rather than a chart library: the file stays
self-contained and prints to PDF as vectors.
"""

import base64
import html
import math
from collections import Counter
from typing import Callable

import pandas as pd

from spyfish.config.wrapper import config

from . import review_data
from .review_data import Frame

IMAGE_WIDTH = 1200  # px; three across on a wide screen at 2x
IMAGE_QUALITY = 80
FEW = 5  # deployments in a group below which its means are flagged
NOT_IN_DB = "not in pipeline DB"

_CSS = """
:root { --bg:#f6f7f5; --fg:#1d2420; --muted:#5d665f; --line:#d6dbd4; --chip:#e9ece7; --accent:#1d6d8f; --warn:#b4541c; --ok:#2e7d4f;
  --g-protected:#0b3d6b; --g-unprotected:#c98a2b; --g-other:#6b8f71; --g-not:#9aa39c; }
@media (prefers-color-scheme: dark) { :root { --bg:#171a18; --fg:#e4e8e3; --muted:#9ca69e; --line:#2d332f; --chip:#222724; --accent:#6cb6d6; --warn:#e8945c; --ok:#6fc08f;
  --g-protected:#6ea8de; --g-unprotected:#e3a64b; --g-other:#8fbf96; --g-not:#6d756f; color-scheme: dark } }
body { background:var(--bg); color:var(--fg); font-family:system-ui,sans-serif; margin:0; padding:1.5rem 16px; }
main { max-width:1400px; margin:0 auto; }
h1 { font-size:1.5rem; margin:0 0 .25rem; font-family:ui-monospace,Menlo,monospace; }
h2 { font-size:1.1rem; margin:2.5rem 0 .75rem; border-bottom:1px solid var(--line); padding-bottom:.3rem; }
h3 { font-size:1rem; margin:2rem 0 .4rem; font-family:ui-monospace,Menlo,monospace; font-weight:600; }
h3.plain { font-family:inherit; }
.draft { font-size:1.1rem; font-weight:800; letter-spacing:.3em; color:var(--warn); margin:0 0 .8rem; }
p.lead { color:var(--muted); max-width:70ch; margin:0 0 1rem; line-height:1.5; }
p.small { color:var(--muted); font-size:.8rem; line-height:1.45; max-width:90ch; margin:0 0 .8rem; }
p.maxn { margin:0 0 .8rem; font-size:.9rem; } p.maxn b { font-variant-numeric:tabular-nums; }
.none { font-size:.85rem; color:var(--warn); margin:0 0 1rem; }
.muted { color:var(--muted); font-size:.8em; } .warn { color:var(--warn); }
.src { font-weight:600; color:var(--accent); text-transform:uppercase; font-size:.75rem; letter-spacing:.04em; }
dl.stats { display:grid; grid-template-columns:max-content 1fr; gap:.3rem 1.2rem; margin:0; font-size:.95rem; }
dl.stats dt { color:var(--muted); } dl.stats dd { margin:0; font-variant-numeric:tabular-nums; }
.grp { display:inline-block; width:.7em; height:.7em; border-radius:2px; margin-right:.4em; vertical-align:-.05em; }
.g-protected { background:var(--g-protected); } .g-unprotected { background:var(--g-unprotected); }
.g-other { background:var(--g-other); } .g-not { background:var(--g-not); }
.tbl { overflow-x:auto; } table { border-collapse:collapse; font-size:.85rem; font-variant-numeric:tabular-nums; }
th, td { padding:.3em .7em; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }
th { color:var(--muted); font-weight:600; } td.n { text-align:right; }
td.drop { font-family:ui-monospace,Menlo,monospace; font-size:.8rem; }
td.tick { color:var(--ok); font-weight:700; text-align:center; } td.tick.muted { color:var(--muted); font-weight:400; }
.fig { overflow-x:auto; margin:0 0 1rem; } svg.sp { display:block; max-width:100%; height:auto; }
svg.sp .grid { stroke:var(--line); } svg.sp .band { fill:var(--chip); opacity:.6; }
svg.sp .tick { font-size:10px; fill:var(--muted); text-anchor:middle; dominant-baseline:middle; }
svg.sp text.sp { font-size:12px; fill:var(--fg); text-anchor:end; dominant-baseline:middle; } svg.sp text.ind { font-weight:700; }
svg.sp g { background:none; }
svg.sp .g-protected { --c:var(--g-protected); } svg.sp .g-unprotected { --c:var(--g-unprotected); }
svg.sp .g-other { --c:var(--g-other); } svg.sp .g-not { --c:var(--g-not); }
svg.sp .iqr { fill:var(--c); fill-opacity:.35; stroke:var(--c); stroke-width:1.2; }
svg.sp .med { stroke:var(--c); stroke-width:2.5; } svg.sp .whisk { stroke:var(--c); stroke-width:1.2; } svg.sp .dot { fill:var(--c); }
.chips { display:flex; flex-wrap:wrap; gap:.35rem; margin:0 0 .6rem; }
.chip { font-size:.75rem; padding:.15em .55em; border-radius:3px; background:var(--chip); color:var(--muted); letter-spacing:.02em; }
.chip.ok { color:var(--ok); } .chip.warn { color:var(--warn); } .chip.src { color:var(--accent); }
figure { margin:0; display:grid; gap:.35rem; min-width:0; }
img { width:100%; height:auto; display:block; box-sizing:border-box; border:1px solid var(--line); }
figcaption { font-size:.85rem; line-height:1.35; height:1.35em; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; }
.slaterow { margin:0 0 .8rem; } .slaterow img { width:auto; max-width:min(100%, 520px); max-height:300px; object-fit:contain; }
.strip { display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:.9rem; align-items:start; margin:0 0 1.2rem; }
@media (max-width:760px) { .strip { grid-template-columns:1fr; } }
.strip img { max-height:300px; object-fit:contain; }
img.agree-match { outline:2px solid #3caa3c; } img.agree-off { outline:2px solid #e0aa00; } img.agree-none { outline:2px solid #e62828; }
.c-match { color:#3caa3c; font-weight:600; } .c-off { color:#c99400; font-weight:600; } .c-none { color:#e62828; font-weight:600; }
.caveat { border-left:3px solid var(--warn); padding:.4rem .9rem; margin:0 0 1rem; max-width:80ch; font-size:.9rem; line-height:1.5; }
.caveat ul { margin:.4rem 0 0; padding-left:1.1rem; } .caveat li { margin:.2rem 0; }
"""

_SLATE_CELL = {
    "confirmed": "<td class='tick'>✓</td>",
    "not confirmed": "<td class='tick muted' title='not confirmed by eye'>✓</td>",
    "no board shown": "<td class='muted'>no board</td>",
    "missing": "<td class='warn'>missing</td>",
}

esc = html.escape


def _swatch(group: str) -> str:
    """The group's colour square: `Not recorded` -> class `g-not`."""
    return f"<span class='grp g-{group.split()[0].lower()}'></span>"


def _counts(values: pd.Series) -> str:
    """`12 ok, 3 no video`: the commonest first."""
    return ", ".join(f"{n} {k}" for k, n in values.value_counts().items()) or "none"


def _seen(values: pd.Series) -> str:
    seen = int((values > 0).sum())
    return f"{seen}/{len(values)} <span class='muted'>{seen / len(values):.0%}</span>"


# ── summary and deployments table ────────────────────────────────────────────


def _summary(table: pd.DataFrame, source_labels: dict, group_order: list) -> str:
    in_db = table[table["Status"] != NOT_IN_DB]
    groups = in_db["Protection"].value_counts()
    rows = [
        ("Deployments in survey", str(len(in_db))),
        ("Status", _counts(in_db["Status"])),
        ("Video", _counts(in_db["video_presence"].fillna("unknown"))),
        (
            "Protection",
            ", ".join(f"{groups[g]} {g}" for g in group_order if g in groups) or "none",
        ),
        (
            "Slates found",
            f"{in_db['Slate'].isin(['confirmed', 'not confirmed']).sum()} of {len(in_db)}",
        ),
        (
            "Annotated",
            ", ".join(
                f"{n} {name}"
                for s, name in source_labels.items()
                if (n := in_db["sources"].map(lambda who: s in who).sum())
            )
            or "none",
        ),
    ]
    return (
        "<h2>Summary</h2><dl class='stats'>"
        + "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in rows)
        + "</dl>"
    )


def _status_table(table: pd.DataFrame) -> str:
    """One row per deployment: is it analysed, and if not, why; slate; who annotated it."""
    rows = []
    for d, r in table.iterrows():
        if r["Status"] == "ok":
            status = "<td class='tick'>✓</td>"
        else:
            why = f": {esc(r['Why'])}" if r["Why"] else ""
            status = f"<td class='warn'>{esc(r['Status'])}{why}</td>"
        who = esc(r["Annotated by"]) or "<span class='warn'>none</span>"
        rows.append(
            f"<tr><td class='drop'><a href='#{esc(d)}'>{esc(r['Deployment'])}</a></td>"
            f"<td>{_swatch(r['Protection'])}{esc(r['Protection'])}</td>"
            f"{_SLATE_CELL[r['Slate']]}<td>{who}</td>{status}"
            f"<td class='muted'>{esc(r['Notes'])}</td></tr>"
        )
    return (
        "<h2>Deployments</h2><div class='tbl'><table><tr><th>Deployment</th>"
        "<th>Protection</th><th>Slate</th><th>Annotated by</th><th>Status</th>"
        "<th>Notes</th></tr>" + "".join(rows) + "</table></div>"
    )


# ── inside vs outside ────────────────────────────────────────────────────────


def nice_max(v: float) -> int:
    """Smallest of 1, 2, 5, 10, 20, 50, ... at or above ``v``."""
    step = 1
    while True:
        for m in (1, 2, 5):
            if m * step >= v:
                return m * step
        step *= 10


def box_stats(values: pd.Series) -> dict:
    """Tukey box: quartiles, whiskers to the furthest value within 1.5 IQR, the rest outliers."""
    v = values.astype(float)
    q1, med, q3 = (float(v.quantile(q)) for q in (0.25, 0.5, 0.75))
    reach = 1.5 * (q3 - q1)
    inside = v[(v >= q1 - reach) & (v <= q3 + reach)]
    return {
        "q1": q1,
        "med": med,
        "q3": q3,
        "lo": float(inside.min()),
        "hi": float(inside.max()),
        # (value, deployments at that value)
        "out": sorted(Counter(v[(v < q1 - reach) | (v > q3 + reach)]).items()),
    }


def _species_figure(
    per_drop: pd.DataFrame,
    order: list[str],
    drops_in: dict[str, list[str]],
    bold: set[str],
    label: Callable[[str], str],
) -> str:
    """Horizontal boxplots of MaxN per deployment, 0s included like the means.

    One lane per protection group, every species on one shared log(MaxN + 1)
    axis. Shared so rows compare; log + 1 because it keeps the 0s (plain log
    cannot) while one school of 39 no longer squeezes every 1, 2 and 3 of the
    survey against 0. The boxes are fitted on that scale too, so an outlier is
    one far out in ratio, not in raw count.
    """
    groups = list(drops_in)
    lane, gap, head = 14, 12, 26
    label_w, x0, x_w = 170, 180, 640
    w = x0 + x_w + 20
    block = lane * len(groups) + gap
    h = head + block * len(order)
    xmax = nice_max(max(float(per_drop[order].max().max()), 1.0))
    ticks = [t for t in review_data.LOG_TICKS if t <= xmax]
    if xmax not in ticks:
        ticks.append(xmax)

    def x(t: float) -> float:  # t is already log(MaxN + 1)
        return x0 + x_w * t / math.log1p(xmax)

    out = [
        f"<svg class='sp' viewBox='0 0 {w} {h}' width='{w}' role='img' "
        "aria-label='MaxN per deployment, per species and protection group'>"
    ]
    for t in ticks:
        xt = x(math.log1p(t))
        out.append(
            f"<line class='grid' x1='{xt:.1f}' x2='{xt:.1f}' y1='{head - 6}' "
            f"y2='{h - gap / 2}'/><text class='tick' x='{xt:.1f}' y='{head - 14}'>{t}</text>"
        )
    for i, sp in enumerate(order):
        y0 = head + block * i
        if i % 2:
            out.append(
                f"<rect class='band' x='0' y='{y0 - gap / 2}' width='{w}' height='{block}'/>"
            )
        out.append(
            f"<text class='sp{' ind' if sp in bold else ''}' x='{label_w}' "
            f"y='{y0 + lane * len(groups) / 2:.1f}'>{esc(label(sp))}</text>"
        )
        for j, g in enumerate(groups):
            vals = per_drop.loc[drops_in[g], sp]
            cy = y0 + lane * j + lane / 2
            st = box_stats(vals.astype(float).map(math.log1p))
            # raw counts for the reader
            q1, med, q3 = (vals.quantile(q) for q in (0.25, 0.5, 0.75))
            tip = (
                f"{g}: seen on {int((vals > 0).sum())} of {len(vals)}; median {med:g}, "
                f"middle half {q1:g} to {q3:g}, max {vals.max():g}, mean {vals.mean():.2f}"
            )
            out.append(
                f"<g class='g-{g.split()[0].lower()}'><title>{esc(tip)}</title>"
                f"<line class='whisk' x1='{x(st['lo']):.1f}' x2='{x(st['hi']):.1f}' "
                f"y1='{cy:.1f}' y2='{cy:.1f}'/>"
            )
            for end in (st["lo"], st["hi"]):
                out.append(
                    f"<line class='whisk' x1='{x(end):.1f}' x2='{x(end):.1f}' "
                    f"y1='{cy - 3:.1f}' y2='{cy + 3:.1f}'/>"
                )
            out.append(
                f"<rect class='iqr' x='{x(st['q1']):.1f}' y='{cy - 5:.1f}' "
                f"width='{x(st['q3']) - x(st['q1']):.1f}' height='10'/>"
                f"<line class='med' x1='{x(st['med']):.1f}' x2='{x(st['med']):.1f}' "
                f"y1='{cy - 5:.1f}' y2='{cy + 5:.1f}'/>"
            )
            # one dot per deployment: equal values fan out across the lane
            for v, k in st["out"]:
                for n in range(k):
                    dy = 0 if k == 1 else -4.5 + 9 * n / (k - 1)
                    out.append(
                        f"<circle class='dot' cx='{x(v):.1f}' cy='{cy + dy:.1f}' r='2'/>"
                    )
            out.append("</g>")
    out.append("</svg>")
    return "".join(out)


def _inside_outside(
    table: pd.DataFrame,
    chosen: pd.Series,
    per_drop: pd.DataFrame,
    label: Callable[[str], str],
    source_labels: dict,
    group_order: list,
    wanted: list[str],
) -> str:
    """Indicator species per protection group, then every species: table and boxplot."""
    if chosen.empty:
        return (
            "<h2>Inside vs outside</h2><p class='none'>No annotated deployments "
            "to compare yet.</p>"
        )
    group = table.loc[chosen.index, "Protection"]
    drops_in = {
        g: list(group.index[group == g]) for g in group_order if (group == g).any()
    }
    used = ", ".join(
        f"{n} {source_labels[s]}" for s, n in chosen.value_counts().items()
    )
    parts = [
        "<h2>Inside vs outside</h2><p class='small'>Protected = "
        + esc(" / ".join(config.protected_statuses))
        + "; Unprotected = "
        + esc(" / ".join(config.unprotected_statuses))
        + "; Other = partial regimes (Type II MPA, mātaitai, taiāpure). Each "
        "deployment uses its best source: BIIGLE expert, legacy expert, citizen "
        f"science, then ML ({esc(used)}). Means include the zeros: a deployment "
        "where the species was not seen counts as MaxN 0.</p>"
    ]

    # The fished species a reserve exists to protect; summing every species
    # together answers no question.
    indicators = [sp for sp in wanted if sp in per_drop.columns]
    eligible = table[table["eligible"]]
    parts.append(
        "<div class='tbl'><table><tr><th>Group</th><th>Site statuses</th>"
        "<th class='n'>Annotated</th>"
        + "".join(
            f"<th class='n'>{esc(label(sp))} MaxN</th><th class='n'>seen</th>"
            for sp in indicators
        )
        + "<th class='n'>Species per deployment</th></tr>"
    )
    for g, drops in drops_in.items():
        sub = per_drop.loc[drops]
        in_group = eligible[eligible["Protection"] == g]
        statuses = ", ".join(
            f"{esc(str(k))} ({n})"
            for k, n in in_group["protection_status"]
            .fillna("no site record")
            .value_counts()
            .items()
        )
        few = " <span class='warn'>(few)</span>" if len(sub) < FEW else ""
        cells = "".join(
            f"<td class='n'><b>{sub[sp].mean():.2f}</b></td><td class='n'>{_seen(sub[sp])}</td>"
            for sp in indicators
        )
        parts.append(
            f"<tr><td>{_swatch(g)}{g}</td><td>{statuses}</td>"
            f"<td class='n'>{len(sub)} of {len(in_group)}{few}</td>{cells}"
            f"<td class='n'>{(sub > 0).sum(axis=1).mean():.1f}</td></tr>"
        )
    parts.append("</table></div>")
    unseen = [sp for sp in wanted if sp not in per_drop.columns]
    if unseen:
        parts.append(
            "<p class='lead'>Not seen on any annotated deployment of this survey: "
            + ", ".join(esc(label(sp)) for sp in unseen)
            + ".</p>"
        )

    seen_any = (per_drop > 0).sum().sort_values(ascending=False)
    order = indicators + [sp for sp in seen_any.index if sp not in indicators]
    if not order:  # every annotated deployment was reviewed with nothing seen
        return "\n".join(parts)
    parts.append(
        "<h3 class='plain'>Species</h3><p class='small'>Mean MaxN includes the 0s.</p>"
        "<div class='tbl'><table><tr><th>Species</th>"
        + "".join(f"<th>{g}: seen</th><th class='n'>mean MaxN</th>" for g in drops_in)
        + "</tr>"
    )
    for sp in order:
        cells = "".join(
            f"<td class='n'>{_seen(per_drop.loc[drops, sp])}</td>"
            f"<td class='n'>{per_drop.loc[drops, sp].mean():.2f}</td>"
            for drops in drops_in.values()
        )
        name = esc(label(sp))
        parts.append(
            f"<tr><td>{f'<b>{name}</b>' if sp in indicators else name}</td>{cells}</tr>"
        )
    legend = " ".join(f"{_swatch(g)}{g}" for g in drops_in)
    parts.append(
        "</table></div><p class='small'>MaxN per deployment, 0s included (as in "
        "the means). Each species has one lane per group, top to bottom "
        f"{legend}. The box spans the middle half of the deployments, the line "
        "across it is the median, the whiskers reach the furthest deployment "
        "within 1.5 box-lengths, and each dot is one deployment beyond them. One "
        "axis for every species, on a log(MaxN + 1) scale: 0 stays 0, and 1 to 3 "
        "get as much room as 10 to 50. Where most deployments did not see a "
        "species, the box sits at 0 and only the dots show. Hover a lane for its "
        "numbers.</p><div class='fig'>"
        + _species_figure(per_drop, order, drops_in, set(indicators), label)
        + "</div>"
    )
    return "\n".join(parts)


# ── deployments ──────────────────────────────────────────────────────────────


def _figure(frame: Frame, caption: str) -> str:
    """Caption above the image; the agreement colour is an outline around it."""
    image = review_data.jpeg_bytes(frame.path, IMAGE_WIDTH, IMAGE_QUALITY)
    agree = f" class='agree-{frame.agreement}'" if frame.agreement else ""
    return (
        "<figure>"
        + (f"<figcaption>{esc(caption)}</figcaption>" if caption else "")
        + f"<img{agree} loading='lazy' alt='{esc(caption or 'Slate')}' "
        f"src='data:image/jpeg;base64,{base64.b64encode(image).decode()}'></figure>"
    )


def _chips(row: pd.Series, source_labels: dict) -> str:
    """The deployment's facts as a row of chips."""

    def chip(text, kind: str = "") -> str:
        return f"<span class='chip {kind}'>{text}</span>"

    if row["Status"] == NOT_IN_DB:
        chips = [chip(NOT_IN_DB, "warn")]
    else:
        chips = [chip(esc(row["Status"]), "ok" if row["Status"] == "ok" else "warn")]
        if row["Why"]:
            chips.append(chip(esc(row["Why"]), "warn"))
        status = row["protection_status"]
        chips.append(
            chip(
                _swatch(row["Protection"])
                + esc(str(status) if pd.notna(status) else "no site record")
            )
        )
        if pd.notna(row["Site"]):
            chips.append(chip(esc(str(row["Site"]))))
        if pd.notna(row["depth"]) and row["depth"]:
            chips.append(chip(f"{float(row['depth']):g} m"))
    chips += [
        chip(esc(name), "src")
        for s, name in source_labels.items()
        if s in row["sources"]
    ]
    return "<div class='chips'>" + "".join(chips) + "</div>"


def _about(told: set) -> str:
    """How to read the page: once, at the end."""
    items = [
        f"<b>{name}</b> {text}"
        for s, (name, text) in review_data.SOURCE_NOTES.items()
        if s in told
    ]
    if review_data.BORDER in told:
        items.append(
            "<b>Frame colour</b> is the model's count of that species on the "
            "frame against the recorded MaxN. <span class='c-none'>Red</span>: "
            "it sees none of them. <span class='c-off'>Yellow</span>: some, or "
            "more. <span class='c-match'>Green</span>: it agrees."
        )
    if review_data.RANDOM in told:
        items.append(review_data.RANDOM_NOTE)
    items.append(
        "<b>Slate</b> column: ✓ checked by eye; grey ✓ picked by the detector, "
        "not yet checked."
    )
    return (
        f"<h2>About this review</h2><div class='caveat'><p>{review_data.ABOUT_INTRO}"
        "</p><ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul></div>"
    )


def build_html(
    survey_id: str,
    table: pd.DataFrame,
    chosen: pd.Series,
    per_drop: pd.DataFrame,
    frames: dict[str, list[Frame]],
    sources: list[str],
    *,
    label: Callable[[str], str],
    source_labels: dict[str, str],
    group_order: list[str],
    indicators: list[str],
    no_board: set[str],
    slates: bool = False,
) -> str:
    """One survey as a self-contained HTML page.

    ``table`` is the page's deployment table (indexed by DropID), ``chosen``
    and ``per_drop`` come from `review_data.best_source_maxn`, ``frames`` from
    `review_data.collect_frames`. ``sources`` are the ones whose frames show,
    best first; ``label`` turns a scientific name into the one a reader sees;
    ``indicators`` are current scientific names.

    ``slates`` puts each deployment's slate picture in the file. Off by
    default: the board has the site's GPS position written on it, and this
    file is made to be sent on. The Slate column of the table (found or not)
    is in the file either way.
    """
    sid = esc(survey_id)
    parts = [
        "<!doctype html><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>{sid} review</title><style>{_CSS}</style><main>",
        f"<h1>{sid}</h1><div class='draft'>DRAFT</div>",
        _summary(table, source_labels, group_order),
        _inside_outside(
            table, chosen, per_drop, label, source_labels, group_order, indicators
        ),
        _status_table(table),
    ]
    told: set = set()
    for d, row in table.iterrows():
        slate, picks = review_data.pick_frames(frames.get(d, []), sources)
        told |= review_data.caveats(picks)
        parts.append(f"<h3 id='{esc(d)}'>{esc(d)}</h3>{_chips(row, source_labels)}")
        for note in review_data.skipped_notes(frames.get(d, []), sources, label):
            parts.append(f"<p class='none'>⚠ {esc(note)}</p>")
        if row["bad"]:
            parts.append(
                "<p class='none'>Bad deployment: not counted in the comparison.</p>"
            )
        if d in chosen.index:
            counts = per_drop.loc[d]
            counts = counts[counts > 0].sort_values(ascending=False)
            parts.append(
                f"<p class='maxn'><span class='src'>{esc(source_labels[chosen[d]])}"
                "</span> MaxN: "
                + (
                    ", ".join(
                        f"{esc(label(sp))} <b>{int(n)}</b>" for sp, n in counts.items()
                    )
                    or "nothing seen"
                )
                + "</p>"
            )
        # The slate on its own line, no caption (the table says whether it is
        # confirmed); the MaxN frames on the next, three across.
        if slates and slate:
            parts.append(f"<div class='slaterow'>{_figure(slate[0], '')}</div>")
        elif slates and d in no_board:
            parts.append(
                "<p class='none'>No slate: the video never shows a board "
                "(checked by eye).</p>"
            )
        elif slates:
            parts.append("<p class='none'>No slate found.</p>")
        if picks:
            parts.append(
                "<div class='strip'>"
                + "".join(_figure(f, f.caption(label)) for f in picks)
                + "</div>"
            )
    parts += [_about(told), "</main>"]
    return "\n".join(parts)
