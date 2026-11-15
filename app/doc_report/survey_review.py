"""Survey review view for the DOC reporting page.

One survey at a time, for checking slates and MaxN frames, read live from the
deployment folders: `{drop}_slate.jpg` and `review/` (filled by
`scripts/wip/export_maxn_frames.py`). The Export section writes the same survey
as one HTML file (`review_report.py`).

That file embeds every image of the survey. Here images are the expensive
part, so they are kept small and few:

* deployments are paged, and only the page on screen has its images rendered
  (an expander would not do: collapsed content is still sent);
* each image is a cached thumbnail, a tenth of the exported file's bytes;
* "Full size" is per deployment and a fragment, so asking for one deployment's
  originals reruns that deployment only.

The sidebar filters (season, MPA, region, protection) decide which surveys are
offered. Inside the chosen survey every deployment shows, whatever the filters:
the inside vs outside comparison needs both sides.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from ecology_data import (
    OTHER_PROTECTION,
    PROTECTED,
    SOURCE_PRIORITY,
    UNPROTECTED,
    UNRECORDED_PROTECTION,
    protection_group,
    source_key,
)
from theme import NEUTRAL, REVIEW_AGREEMENT_COLORS
from utils import CACHE_TTL_SECONDS

from spyfish.config.base import INGEST_ISSUE, IngestStatus
from spyfish.config.species import species_registry
from spyfish.config.wrapper import config

from . import data as report_data
from . import review_data, review_report
from .charting import group_colors, style
from .error_review import load_error_data
from .layout import chips, extra_filters, section
from .review_data import Frame

PAGE_SIZE = 8  # deployments per page: about 30 images
THUMB_WIDTH = 640  # px; a third of the page at 2x
THUMB_QUALITY = 72
FULL_QUALITY = 90
THUMB_BORDER = 6  # px of agreement colour around a thumbnail
FULL_BORDER = 14
PER_ROW = 3

SOURCES = sorted(SOURCE_PRIORITY, key=SOURCE_PRIORITY.get)  # best first
SOURCE_LABELS = {
    "expert": "BIIGLE expert",
    "expert_legacy": "Legacy expert",
    "expert_training": "Expert training frames",
    "citsci": "Citizen science",
    "ml": "ML",
}
GROUP_ORDER = [PROTECTED, UNPROTECTED, OTHER_PROTECTION, UNRECORDED_PROTECTION]
AGREEMENT_NOTES = {
    "match": ":green[model agrees]",
    "off": ":orange[model count differs]",
    "none": ":red[model sees none]",
}
SLATE_TEXT = {
    "confirmed": "✓",
    "not confirmed": "✓ not confirmed",
    "no board shown": "no board",
    "missing": "missing",
}


# ── cached loads ─────────────────────────────────────────────────────────────


def _canonical(name: str) -> str:
    """Current scientific name; an unknown name comes back as written."""
    return species_registry().resolve_name(name)[0]


def _label(scientific: str) -> str:
    return species_registry().label_for(scientific)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _frames() -> dict[str, list[Frame]]:
    return review_data.collect_frames(config.deployment_data_dir, _canonical)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _annotations() -> pd.DataFrame:
    """Annotations with the source key and current species names.

    Rows inserted before names were canonicalised still say "Pagrus auratus":
    read as written, snapper would be two species here.
    """
    ann = report_data.load_annotations()[
        [
            "drop_id",
            "survey_id",
            "scientific_name",
            "max_interval",
            "annotated_by",
            "external_id",
        ]
    ].copy()
    ann["source"] = source_key(ann)
    names = ann["scientific_name"].dropna().unique()
    ann["scientific_name"] = ann["scientific_name"].map(
        {n: _canonical(n) for n in names}
    )
    return ann


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _index() -> pd.DataFrame:
    """`drop_id`, `survey_id` for every deployment of a survey with something exported."""
    exported = {d: s for d in _frames() if (s := review_data.survey_of(d))}
    dep = report_data.load_deployments()
    dep = dep.loc[
        dep["survey_id"].isin(set(exported.values())), ["drop_id", "survey_id"]
    ]
    unknown = [d for d in exported if d not in set(dep["drop_id"])]
    return pd.concat(
        [
            dep,
            pd.DataFrame(
                {"drop_id": unknown, "survey_id": [exported[d] for d in unknown]}
            ),
        ],
        ignore_index=True,
    )


@st.cache_data(max_entries=3000, show_spinner=False)
def _thumbnail(path: str, mtime: float, border: str | None) -> bytes:
    """A frame at thumbnail size. `mtime` is in the key so a re-export shows."""
    return review_data.jpeg_bytes(
        Path(path), THUMB_WIDTH, THUMB_QUALITY, border, THUMB_BORDER
    )


# ── one survey ───────────────────────────────────────────────────────────────


def _flag(value) -> bool:
    """A 0/1 DB column as a bool; NULL is False."""
    return bool(value) and not pd.isna(value)


def _short(drop_id: str) -> str:
    """`SLI_065_01` of `SLI_20220228_BUV_SLI_065_01`: the survey is already chosen."""
    return drop_id.split("_BUV_", 1)[-1]


def _deployment_table(
    drops: list[str], dep: pd.DataFrame, ann: pd.DataFrame, frames: dict
) -> pd.DataFrame:
    """One row per deployment: is it analysed, and if not, why; slate; who annotated it."""
    errors = load_error_data()
    reasons: dict[str, list[str]] = {}  # block the deployment
    noted: dict[str, list[str]] = {}  # recorded at ingest, still analysed
    if not errors.empty:
        for e in errors[errors["DropID"].isin(drops)].to_dict("records"):
            d = e["DropID"]
            msg = str(e.get("ErrorMessage") or e.get("ErrorType")).replace(f"{d}: ", "")
            bucket = noted if e.get("ErrorType") == INGEST_ISSUE else reasons
            bucket.setdefault(d, []).append(msg)

    verified = review_data.read_drop_list(config.slates_verified_path)
    no_board = review_data.read_drop_list(config.slates_none_path)
    by_source = ann.groupby("drop_id")["source"].agg(set)
    rows = []
    for d in drops:
        row = dep.loc[d].to_dict() if d in dep.index else {}
        if row:
            status, why = review_data.deployment_status(row, reasons.get(d, []))
        else:
            status, why = "not in pipeline DB", ""
        has_slate = any(f.kind == "slate" for f in frames.get(d, [])) or (
            bool(row) and config.get_slate_frame_path(d).exists()
        )
        who = by_source.get(d, set())
        rows.append(
            {
                "drop_id": d,
                "Deployment": _short(d),
                "Site": row.get("site_name"),
                "Protection": row.get("group", UNRECORDED_PROTECTION),
                "protection_status": row.get("protection_status"),
                "Slate": review_data.slate_state(
                    d, {d} if has_slate else set(), verified, no_board
                ),
                "Annotated by": ", ".join(
                    SOURCE_LABELS[s] for s in SOURCES if s in who
                ),
                "sources": who,
                "Status": status,
                "Why": why,
                "Notes": "; ".join(
                    review_data.short_reason(m) for m in noted.get(d, [])
                ),
                "video_presence": row.get("video_presence"),
                "depth": row.get("depth"),
                "bad": _flag(row.get("is_bad_deployment")),
                "eligible": row.get(IngestStatus.COLUMN) == IngestStatus.OK
                and not _flag(row.get("is_bad_deployment")),
            }
        )
    return pd.DataFrame(rows).set_index("drop_id")


def _counts(values: pd.Series) -> str:
    """`12 ok, 3 no video`: the commonest first."""
    return ", ".join(f"{n} {k}" for k, n in values.value_counts().items()) or "none"


def _summary(table: pd.DataFrame) -> None:
    in_db = table[table["Status"] != "not in pipeline DB"]
    kpis = st.columns(4)
    kpis[0].metric("Deployments in survey", f"{len(in_db):,}")
    kpis[1].metric(
        "Ready to analyse",
        f"{(in_db['Status'] == 'ok').sum():,}",
        help="Ingested ok with a video in S3. The Deployments table says why "
        "each of the others is not.",
    )
    kpis[2].metric(
        "Slates found",
        f"{(in_db['Slate'].isin(['confirmed', 'not confirmed'])).sum():,} of {len(in_db):,}",
    )
    kpis[3].metric(
        "Annotated",
        f"{(in_db['sources'].map(len) > 0).sum():,}",
        help="Deployments with rows from any source.",
    )
    by_source = ", ".join(
        f"{n} {SOURCE_LABELS[s]}"
        for s in SOURCES
        if (n := in_db["sources"].map(lambda who: s in who).sum())
    )
    st.markdown(
        f"**Status:** {_counts(in_db['Status'])}  \n"
        f"**Video:** {_counts(in_db['video_presence'].fillna('unknown'))}  \n"
        f"**Protection:** {_counts(in_db['Protection'])}  \n"
        f"**Annotated by:** {by_source or 'none'}"
    )


def _seen(values: pd.Series) -> str:
    seen = int((values > 0).sum())
    return f"{seen}/{len(values)} ({seen / len(values):.0%})"


def _indicators() -> list[str]:
    """The fished species a reserve exists to protect, as current names."""
    return list(dict.fromkeys(_canonical(sp) for sp in config.indicator_species))


def _inside_outside(
    table: pd.DataFrame, chosen: pd.Series, per_drop: pd.DataFrame
) -> None:
    """Indicator species per protection group, then every species: table and boxplot."""
    if chosen.empty:
        st.info("No annotated deployments to compare yet.")
        return
    group = table.loc[chosen.index, "Protection"]
    groups = [g for g in GROUP_ORDER if (group == g).any()]
    used = ", ".join(
        f"{n} {SOURCE_LABELS[s]}" for s, n in chosen.value_counts().items()
    )
    st.caption(
        f"Protected = {' / '.join(config.protected_statuses)}. Unprotected = "
        f"{' / '.join(config.unprotected_statuses)}. Other = partial regimes "
        "(Type II MPA, mātaitai, taiāpure). Each deployment uses its best "
        f"source: BIIGLE expert, legacy expert, citizen science, then ML ({used}). "
        "Means include the zeros: a deployment where the species was not seen "
        "counts as MaxN 0."
    )

    # Summing every species together answers no question.
    wanted = _indicators()
    indicators = [sp for sp in wanted if sp in per_drop.columns]
    eligible = table[table["eligible"]]
    rows = []
    for g in groups:
        sub = per_drop.loc[group.index[group == g]]
        row = {
            "Group": g,
            "Site statuses": ", ".join(
                f"{k} ({n})"
                for k, n in eligible.loc[
                    eligible["Protection"] == g, "protection_status"
                ]
                .fillna("no site record")
                .value_counts()
                .items()
            ),
            "Annotated": f"{len(sub)} of {(eligible['Protection'] == g).sum()}"
            + (" (few)" if len(sub) < 5 else ""),
        }
        for sp in indicators:
            row[f"{_label(sp)} mean MaxN"] = round(sub[sp].mean(), 2)
            row[f"{_label(sp)} seen"] = _seen(sub[sp])
        row["Species per deployment"] = round((sub > 0).sum(axis=1).mean(), 1)
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    unseen = [sp for sp in wanted if sp not in per_drop.columns]
    if unseen:
        st.caption(
            "Not seen on any annotated deployment of this survey: "
            + ", ".join(_label(sp) for sp in unseen)
            + "."
        )

    seen_any = (per_drop > 0).sum().sort_values(ascending=False)
    order = indicators + [sp for sp in seen_any.index if sp not in indicators]
    rows = []
    for sp in order:
        row = {"Species": _label(sp)}
        for g in groups:
            values = per_drop.loc[group.index[group == g], sp]
            row[f"{g}: seen"] = _seen(values)
            row[f"{g}: mean MaxN"] = round(values.mean(), 2)
        rows.append(row)
    st.markdown("**Species**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=320)

    long = per_drop[order].rename(columns=_label).stack().rename("MaxN").reset_index()
    long.columns = ["drop_id", "Species", "MaxN"]
    long["Group"] = long["drop_id"].map(group)
    # log(MaxN + 1), like the HTML report: it keeps the 0s (plain log cannot)
    # and one school of 39 no longer squeezes every 1, 2 and 3 against 0. The
    # boxes are fitted on that scale too, so an outlier is far out in ratio.
    long["scaled"] = np.log1p(long["MaxN"].astype(float))
    top = max(float(long["MaxN"].max()), 1.0)
    ticks = [t for t in review_data.LOG_TICKS if t <= top]
    fig = px.box(
        long,
        x="scaled",
        y="Species",
        color="Group",
        orientation="h",
        points="outliers",
        category_orders={"Species": [_label(sp) for sp in order], "Group": groups},
        color_discrete_map={**group_colors(), UNRECORDED_PROTECTION: NEUTRAL},
        custom_data=["MaxN", "drop_id"],
    )
    style(fig, height=110 + len(order) * (12 * len(groups) + 12), legend="top")
    fig.update_yaxes(title_text="")
    fig.update_xaxes(
        title_text="MaxN per deployment (log scale)",
        tickvals=[float(np.log1p(t)) for t in ticks],
        ticktext=[str(t) for t in ticks],
        range=[-0.05, float(np.log1p(top)) * 1.03],
    )
    # the box's own hover would read out log values; the dots say the count
    fig.update_traces(
        hoveron="points",
        hovertemplate="%{customdata[1]}<br>MaxN %{customdata[0]}<extra></extra>",
    )
    st.plotly_chart(fig, key="review_species_box")
    st.caption(
        "MaxN per deployment, zeros included. The box spans the middle half of "
        "the deployments, the line across it is the median, and each dot is one "
        "deployment beyond the whiskers. One axis for every species, on a "
        "log(MaxN + 1) scale: 0 stays 0, and 1 to 3 get as much room as 10 to "
        "50. Where most deployments did not see a species the box sits at 0 "
        "and only the dots show. Indicator species are listed first."
    )


# ── frames ───────────────────────────────────────────────────────────────────


def _badges(row: pd.Series) -> str:
    """The deployment's facts as a row of badges."""
    if row["Status"] == "not in pipeline DB":
        return ":orange-badge[not in pipeline DB]"

    def badge(text, colour: str = "gray") -> str:
        return f":{colour}-badge[{str(text).replace('[', '(').replace(']', ')')}]"

    out = [badge(row["Status"], "green" if row["Status"] == "ok" else "orange")]
    if row["Why"]:
        out.append(badge(row["Why"], "orange"))
    if pd.notna(row["protection_status"]):
        out.append(badge(row["protection_status"]))
    if pd.notna(row["Site"]):
        out.append(badge(row["Site"]))
    if _flag(row["depth"]):
        out.append(badge(f"{float(row['depth']):g} m"))
    out += [badge(SOURCE_LABELS[s], "blue") for s in SOURCES if s in row["sources"]]
    return " ".join(out)


def _show(frame: Frame, full: bool) -> None:
    """One image. The agreement colour goes around the picture, never over it."""
    border = REVIEW_AGREEMENT_COLORS.get(frame.agreement)
    if not full:
        image = _thumbnail(str(frame.path), frame.mtime, border)
    elif border:
        image = review_data.jpeg_bytes(
            frame.path, None, FULL_QUALITY, border, FULL_BORDER
        )
    else:
        image = str(frame.path)  # the exported file as it is
    st.image(image, width="stretch")


@st.fragment
def _card(
    drop_id: str, badges: str, maxn: str, slate_note: str, frames: list[Frame]
) -> None:
    """One deployment: its facts, its slate, its MaxN frames.

    A fragment, so the "Full size" switch reruns this deployment alone and the
    rest of the page keeps its thumbnails.
    """
    title, switch = st.columns([5, 1], vertical_alignment="bottom")
    title.markdown(f"#### {drop_id}")
    full = switch.toggle(
        "Full size",
        key=f"review_full_{drop_id}",
        help="Load this deployment's images as exported. Off: thumbnails.",
    )
    st.markdown(badges)
    if maxn:
        st.markdown(maxn)
    if slate_note:
        st.caption(slate_note)
    label = _label
    cells = st.columns(1 if full else PER_ROW)
    for i, frame in enumerate(frames):
        with cells[i % len(cells)]:
            note = AGREEMENT_NOTES.get(frame.agreement)
            st.caption(frame.caption(label) + (f" · {note}" if note else ""))
            _show(frame, full)


def _step(key: str, by: int) -> None:
    st.session_state[key] = st.session_state.get(key, 0) + by


def _pages(shown: list[str], key: str) -> tuple[list[list[str]], int]:
    """The deployments cut into pages, and the page on screen."""
    pages = [shown[i : i + PAGE_SIZE] for i in range(0, len(shown), PAGE_SIZE)]
    page = min(st.session_state.get(key, 0), len(pages) - 1)
    st.session_state[key] = page
    return pages, page


def _pager(pages: list[list[str]], page: int, key: str, bottom: bool = False) -> None:
    """Previous / which page / Next. Drawn above the cards and again below them.

    The page picker is the top one only: a widget key can be drawn once, and
    under the last card the next page is the one thing a reader wants.
    """
    if len(pages) == 1:
        return
    total = sum(len(p) for p in pages)

    def describe(i: int) -> str:
        return (
            f"{i * PAGE_SIZE + 1} to {i * PAGE_SIZE + len(pages[i])} of {total}"
            f" · {_short(pages[i][0])} to {_short(pages[i][-1])}"
        )

    where = "bottom" if bottom else "top"
    prev, pick, nxt = st.columns([1, 4, 1], vertical_alignment="bottom")
    prev.button(
        "← Previous",
        on_click=_step,
        args=(key, -1),
        disabled=page == 0,
        width="stretch",
        key=f"{key}_prev_{where}",
    )
    if bottom:
        pick.caption(f"Deployments {describe(page)}")
    else:
        pick.selectbox("Deployments", range(len(pages)), format_func=describe, key=key)
    nxt.button(
        "Next →",
        on_click=_step,
        args=(key, 1),
        disabled=page == len(pages) - 1,
        width="stretch",
        key=f"{key}_next_{where}",
    )


def _frames_section(
    shown: list[str],
    table: pd.DataFrame,
    frames: dict,
    sources: list[str],
    chosen: pd.Series,
    per_drop: pd.DataFrame,
    page_key: str,
) -> set:
    """The paged deployment cards. Returns which caveats the page needs explaining."""
    no_board = review_data.read_drop_list(config.slates_none_path)
    told: set = set()
    pages, page = _pages(shown, page_key)
    _pager(pages, page, page_key)
    for d in pages[page]:
        mine = frames.get(d, [])
        slate, picks = review_data.pick_frames(mine, sources)
        told |= review_data.caveats(picks)

        maxn = ""
        if d in chosen.index:
            counts = per_drop.loc[d]
            counts = counts[counts > 0].sort_values(ascending=False)
            maxn = f"**{SOURCE_LABELS[chosen[d]]}** MaxN: " + (
                ", ".join(f"{_label(sp)} **{int(n)}**" for sp, n in counts.items())
                or "nothing seen"
            )
        notes = []
        if table.loc[d, "bad"]:
            notes.append("Bad deployment: not counted in the comparison.")
        if not slate:
            notes.append(
                "No slate: the video never shows a board (checked by eye)."
                if d in no_board
                else "No slate found."
            )
        notes += [f"⚠ {n}" for n in review_data.skipped_notes(mine, sources, _label)]
        with st.container(border=True):
            _card(d, _badges(table.loc[d]), maxn, " ".join(notes), slate + picks)
    _pager(pages, page, page_key, bottom=True)
    return told


def _about(told: set) -> None:
    lines = [
        f"**{name}** {text}"
        for key, (name, text) in review_data.SOURCE_NOTES.items()
        if key in told
    ]
    if review_data.BORDER in told:
        lines.append(
            "**Frame colour** is the model's count of that species on the "
            "frame against the recorded MaxN. :red[Red]: it sees none of them. "
            ":orange[Yellow]: some, or more. :green[Green]: it agrees."
        )
    if review_data.RANDOM in told:
        lines.append(review_data.RANDOM_NOTE)
    lines += [
        "**Slate** column: ✓ checked by eye; “✓ not confirmed” picked by "
        "the detector, not yet checked.",
        "The sidebar filters (season, MPA, region, protection) decide which "
        "surveys are offered. The survey itself is always shown whole.",
    ]
    st.markdown(
        review_data.ABOUT_INTRO + "\n\n" + "\n".join(f"* {line}" for line in lines)
    )


def _survey_order(survey_id: str) -> tuple:
    """Reserve A to Z, and within a reserve the latest survey first."""
    reserve, _, date = survey_id.partition("_")
    return reserve, -int(date.split("_")[0])


def _export(
    survey: str,
    table: pd.DataFrame,
    chosen: pd.Series,
    per_drop: pd.DataFrame,
    frames: dict,
    sources: list[str],
) -> None:
    """Build the survey's one-file HTML report and offer it for download."""
    st.caption(
        "The whole survey as one HTML file with its MaxN frames inside it, to "
        "send to someone without the dashboard. It follows Review sources. "
        "Building takes a few seconds; the file is several MB."
    )
    slates = st.checkbox(
        "Include slates :red[(slates might show GPS coordinates, so take "
        "care who you share the file with)]",
        key="review_export_slates",
        help="Off: the file has the MaxN frames only. The Slate column of the "
        "table (found or not) is in the file either way.",
    )
    # Its own file name, so the copy with coordinates is never the one sent
    # by mistake.
    name = f"{survey}_review_with_slates.html" if slates else f"{survey}_review.html"
    out = config.review_reports_dir / name
    built = (survey, tuple(sources), slates)
    if st.button("Build HTML report", key=f"review_build:{survey}"):
        with st.spinner(f"Building {out.name}"):
            page = review_report.build_html(
                survey,
                table,
                chosen,
                per_drop,
                frames,
                sources,
                label=_label,
                source_labels=SOURCE_LABELS,
                group_order=GROUP_ORDER,
                indicators=_indicators(),
                no_board=review_data.read_drop_list(config.slates_none_path),
                slates=slates,
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(page, encoding="utf-8")
        st.session_state["review_built"] = built
    # Offered only after a build in this session, and only while the sources
    # and the slates choice are the ones it was built with: the button holds the whole file, so it
    # should not be read on every rerun of the page.
    if st.session_state.get("review_built") == built and out.exists():
        st.download_button(
            f"Download {out.name} ({out.stat().st_size / 1e6:.1f} MB)",
            out.read_bytes(),
            file_name=out.name,
            mime="text/html",
            key=f"review_download:{survey}",
        )
        st.caption(f"Also saved at `{out}`.")


def render(ctx: dict) -> None:
    """Render the Survey review view.

    The filtered context picks which surveys are offered; the survey itself is
    read whole.
    """
    index = _index()
    if index.empty:
        st.info(
            "Nothing to review yet. This view reads each deployment's slate "
            "and the frames `scripts/wip/export_maxn_frames.py` writes to its "
            f"`{config.review_dir_name}/` folder, under "
            f"`{config.deployment_data_dir}`."
        )
        return
    filtered = ctx.get("deployments")
    if filtered is not None:
        index = index[index["survey_id"].isin(filtered["survey_id"])]
        if index.empty:
            st.info(
                "None of the surveys with exported review frames is inside the "
                "sidebar filters. Widen the season or the MPA to see them."
            )
            return

    chips(
        [
            "Summary",
            "Inside vs outside",
            "Deployments",
            "Frames",
            "Export",
            "About this review",
        ]
    )

    totals = index.groupby("survey_id").size()
    surveys = sorted(totals.index, key=_survey_order)
    if st.session_state.get("review_survey") not in surveys:
        st.session_state.pop("review_survey", None)
    survey = st.selectbox(
        "Survey",
        surveys,
        key="review_survey",
        format_func=lambda s: f"{s} · {totals[s]} deployments",
        width=420,
    )

    with extra_filters(1) as slots, slots[0]:
        picked = st.multiselect(
            "Review sources",
            [s for s in SOURCES if s != "expert_training"],
            format_func=SOURCE_LABELS.get,
            key="review_sources",
            placeholder="All sources",
            help="Survey review only: which annotation sources supply the "
            "MaxN counts and frames. Empty means all.",
        )
    # Best first whatever order they were ticked in: the order is the ranking.
    sources = [s for s in SOURCES if s in picked] or SOURCES

    st.badge("Draft", color="orange")

    drops = sorted(index.loc[index["survey_id"] == survey, "drop_id"])
    frames = _frames()
    dep = report_data.load_deployments()
    dep = dep[dep["survey_id"] == survey].set_index("drop_id")
    dep["group"] = protection_group(dep["protection_status"]).fillna(
        UNRECORDED_PROTECTION
    )
    ann = _annotations()
    ann = ann[ann["drop_id"].isin(drops)]
    table = _deployment_table(drops, dep, ann, frames)
    chosen, per_drop = review_data.best_source_maxn(
        ann, list(table.index[table["eligible"]]), sources
    )

    section("Summary")
    _summary(table)

    section("Inside vs outside")
    _inside_outside(table, chosen, per_drop)

    section("Deployments")
    listing = table.copy()
    listing["Slate"] = listing["Slate"].map(SLATE_TEXT)
    listing["Status"] = listing["Status"].where(
        listing["Why"] == "", listing["Status"] + ": " + listing["Why"]
    )
    st.dataframe(
        listing[
            [
                "Deployment",
                "Site",
                "Protection",
                "Slate",
                "Annotated by",
                "Status",
                "Notes",
            ]
        ],
        hide_index=True,
        width="stretch",
        height=min(38 * (len(listing) + 1) + 2, 420),
    )

    section("Frames")
    told = _frames_section(
        drops, table, frames, sources, chosen, per_drop, f"review_page:{survey}"
    )

    section("Export")
    _export(survey, table, chosen, per_drop, frames, sources)

    section("About this review")
    _about(told)
