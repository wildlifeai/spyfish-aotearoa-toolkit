"""UoA ML vs expert-annotation MaxN comparison, the render body.

Compares YOLO ML output (per GoPro chapter file) against the UoA paper's expert
annotations (per deployment, sliced into 6-min intervals) for the
Underwood & Jeffs BUV dataset.

The two time grids are reconciled using the mapping already baked into
``Annotations_CLEANED_v2.csv``: every interval row carries ``video_symlink``
(→ deployment + chapter number) and ``start_sec_in_chapter``. So for each expert
interval we know the exact chapter file + second-offset it begins at, take a
6-min window from there (spilling into the next chapter when it overruns the
~707-s chapter length), and compute ML MaxN inside that same window.

Lives here (importable name) rather than in the emoji-named page file so two
callers can share one implementation:

* ``pages/_advanced/🆚_UoA_ML_vs_Expert.py``, the standalone page, a thin shim;
* the Mussel Insights dashboard's "ML vs Expert detail" view (``embedded=True``).

All data access goes through ``uoa_data``, no paths or constants live here.
"""

import pandas as pd
import plotly.express as px
import streamlit as st
from uoa_data import (
    CHAPTER_SEC,
    DEFAULT_CONF,
    DEPTH_LABEL,
    EXPERT_CSV,
    EXPERT_TO_ML_CLASS,
    FRAMES_ROOT,
    INTERVAL_SEC,
    RAW_ROOT,
    TREAT_LABEL,
    chapter_from_symlink,
    date_key,
    deployment_peak,
    fake_latlon,
    load_expert_rows,
    load_ml_raw,
    maxn_in_window,
    ml_call,
    presence_flip,
    severity,
)

MAXN_RE = r"max ?n"


def render_body(embedded: bool = False, key_prefix: str = "uoa") -> None:
    """The whole comparison. `embedded` drops the standalone chrome (the caller
    owns the title) and namespaces widget keys via `key_prefix` so the view can
    sit inside another dashboard without key collisions."""

    def k(name: str) -> str:
        return f"{key_prefix}_{name}"

    # ── Load + guard ─────────────────────────────────────────────────────────
    if not EXPERT_CSV.exists():
        st.error(f"Expert CSV not found: `{EXPERT_CSV}`")
        return

    raw_by_key, model_by_dep = load_ml_raw()
    expert_df, maxn_cols = load_expert_rows()

    if not raw_by_key:
        st.warning(
            f"No ML raw CSVs found under `{RAW_ROOT}`. "
            "Download a run's `raw/` folder there (filenames like "
            "`ESK_B_20220118_EV_C_v01_<model>_raw.csv`)."
        )
        return

    ml_deps = sorted(model_by_dep.keys(), key=date_key)
    models = sorted(set(model_by_dep.values()))
    days = sorted({d.split("_")[2] for d in ml_deps if len(d.split("_")) > 2})

    # Inline rather than in the sidebar: embedded mode must not fight the host
    # dashboard's sidebar, and the filter reads fine next to the data it scopes.
    fcols = st.columns([2, 1, 1, 2])
    with fcols[0]:
        sel_days = st.multiselect(
            "Day(s) to show",
            days,
            default=days,
            key=k("days"),
            help="Filter every section below to these deployment dates "
            "(parsed from the deployment_id). Narrow to one day to declutter.",
        )
    ml_deps = [d for d in ml_deps if d.split("_")[2] in sel_days]
    with fcols[1]:
        st.metric("Deployments with ML", len(ml_deps))
    with fcols[2]:
        st.metric("Chapter raw CSVs", len(raw_by_key))
    with fcols[3]:
        st.write("**Models:**", ", ".join(models))

    st.caption(
        "Expert annotations slice each deployment into **6-min intervals** "
        "(interval 1 = recording min 6–12; first 6 min are bait soak). ML ran "
        "per **GoPro chapter** (~11.8 min). Each interval is matched to its chapter "
        "+ start-second via `video_symlink` / `start_sec_in_chapter`, then ML MaxN "
        "is computed in that exact 6-min window (spilling into the next chapter "
        "when it overruns ~707 s). Per-species comparison only where the model has "
        "a matching class (Snapper, Spotty); otherwise **fish (any)**."
    )

    conf = st.slider(
        "ML confidence threshold",
        0.15,
        0.95,
        DEFAULT_CONF,
        0.05,
        key=k("conf"),
        help="Detections below this confidence are excluded. Raw CSVs hold "
        "everything ≥ 0.15 (inference conf), so any value ≥ 0.15 re-filters.",
    )

    # Only compare deployments that have BOTH ML data and an expert row
    expert_ml = expert_df[expert_df["deployment_id"].isin(ml_deps)].copy()
    if expert_ml.empty:
        st.warning(
            "None of the ML deployments have matching rows in the expert CSV. "
            f"ML deployments: {ml_deps[:5]}…"
        )
        return

    # Which per-species comparisons are possible given the loaded model(s)
    ml_class_names = set()
    for df in raw_by_key.values():
        if df is not None and not df.empty:
            ml_class_names.update(df["class"].dropna().unique())
    active_species = {
        h: c for h, c in EXPERT_TO_ML_CLASS.items() if c in ml_class_names
    }

    # ── Section 1: per-interval ──────────────────────────────────────────────

    st.divider()
    st.header("1. Per 6-min interval")

    rows = []
    for _, r in expert_ml.iterrows():
        dep = r["deployment_id"]
        chap = chapter_from_symlink(r.get("video_symlink"))
        start = r.get("start_sec_in_chapter")
        if chap is None or pd.isna(start):
            continue
        start = float(start)

        # For an ANNOTATED deployment a blank species cell means "expert looked,
        # saw none" → 0. Only genuinely unannotated rows (unannotated=True)
        # stay None, no expert has looked, so there's nothing to compare.
        is_unann = str(r.get("unannotated")).strip().lower() == "true"

        expert_vals = [float(r[c]) for c in maxn_cols if pd.notna(r[c])]
        expert_fish = (
            None if is_unann else (int(max(expert_vals)) if expert_vals else 0)
        )
        ml_fish = maxn_in_window(raw_by_key, dep, chap, start, conf)

        row = {
            "deployment_id": dep,
            "day": dep.split("_")[2] if len(dep.split("_")) > 2 else "?",
            "interval": (
                int(float(r["interval_idx"]))
                if pd.notna(r.get("interval_idx"))
                else None
            ),
            "chapter": chap,
            "start_sec": round(start, 1),
            "fish_expert": expert_fish,
            "fish_ml": ml_fish,
            "fish_call": ml_call(expert_fish, ml_fish),
            "fish_flip": presence_flip(expert_fish, ml_fish),
            "fish_abs_diff": (
                None if expert_fish is None else abs(expert_fish - ml_fish)
            ),
            "fish_severity": severity(expert_fish, ml_fish),
        }
        for hcol, mlclass in active_species.items():
            label = hcol.replace(" MaxN", "").lower()
            h = None if is_unann else (int(float(r[hcol])) if pd.notna(r[hcol]) else 0)
            m = maxn_in_window(raw_by_key, dep, chap, start, conf, class_name=mlclass)
            row[f"{label}_expert"] = h
            row[f"{label}_ml"] = m
            row[f"{label}_call"] = ml_call(h, m)
            row[f"{label}_flip"] = presence_flip(h, m)
            row[f"{label}_abs_diff"] = None if h is None else abs(h - m)
            row[f"{label}_severity"] = severity(h, m)
        rows.append(row)

    interval_df = pd.DataFrame(rows)

    dep_choices = ["(all)"] + sorted(
        interval_df["deployment_id"].unique(), key=date_key
    )
    chosen = st.selectbox("Filter deployment", dep_choices, key=k("dep"))
    view = (
        interval_df
        if chosen == "(all)"
        else interval_df[interval_df.deployment_id == chosen]
    )
    view = view.sort_values("fish_severity", ascending=False, na_position="last")

    st.caption(
        "Sorted by **`fish_severity`** = (count missed) × (fraction missed), a busy "
        "scene called empty outranks a small undercount. `fish_call`: **−** too few, "
        "**+** too many, **=** exact. `fish_flip` **⚠️** = presence/absence "
        "disagreement (one says fish, the other says none). `fish_abs_diff` = raw "
        "count error."
    )
    st.dataframe(view, hide_index=True, width="stretch")

    # Agreement scatter (fish-any), only rows where expert annotated
    st.subheader("Fish (any) MaxN, expert vs ML")
    scat = interval_df.dropna(subset=["fish_expert"])
    if scat.empty:
        st.info("No expert-annotated intervals among the loaded ML deployments.")
    else:
        fig = px.scatter(
            scat,
            x="fish_expert",
            y="fish_ml",
            color="day",
            hover_data=["deployment_id", "interval"],
            labels={"fish_expert": "Expert MaxN", "fish_ml": "ML MaxN", "day": "Day"},
            title="Per-interval fish MaxN (points on the diagonal = agreement)",
        )
        lim = max(1, int(scat[["fish_expert", "fish_ml"]].max().max()))
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=lim,
            y1=lim,
            line=dict(dash="dash", color="grey"),
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, key=k("scatter"))

    # ── Section 2: per deployment (peak) ─────────────────────────────────────

    st.divider()
    st.header("2. Per deployment, peak")
    st.caption(
        "One row per deployment. **`expert_peak`** = highest single-species MaxN "
        "across intervals; **`expert_total_peak`** = highest Σ-species MaxN of any "
        "interval, the quantity comparable to the ML peak, since ML counts every "
        "fish in a frame regardless of species. The error columns compare ML "
        "against `expert_total_peak` for that reason: judging an all-species count "
        "against a single-species peak builds in a fake ML overcount. Sorted by "
        "`fish_severity_peak`. `fish_call_peak`: **−** too few, **+** too many, "
        "**=** exact. `fish_flip_peak` **⚠️** = presence/absence disagreement."
    )

    dep_rows = []
    for dep in sorted(expert_ml["deployment_id"].unique(), key=date_key):
        sub = expert_ml[expert_ml["deployment_id"] == dep]
        # Deployment is unannotated only if ALL its rows are flagged unannotated.
        dep_unann = (
            sub["unannotated"].astype(str).str.strip().str.lower().eq("true").all()
        )
        # Per-deployment peak (max single-species MaxN across intervals).
        vals = [
            float(r[c]) for _, r in sub.iterrows() for c in maxn_cols if pd.notna(r[c])
        ]
        expert_peak = None if dep_unann else (int(max(vals)) if vals else 0)
        # Paper's "total abundance" = SUM of species MaxN per 6-min segment;
        # per-deployment value = peak across segments. This is the all-species
        # quantity, so it is what the ML peak is judged against.
        seg_sums = [
            sum(float(r[c]) for c in maxn_cols if pd.notna(r[c]))
            for _, r in sub.iterrows()
        ]
        expert_total_peak = (
            None if dep_unann else (int(max(seg_sums)) if seg_sums else 0)
        )
        ml_peak = deployment_peak(raw_by_key, dep, conf)

        # deployment_id = SITE_HAB_DATE_TIMEBUCKET_TREAT, e.g. ESK_B_20220118_EV_C
        parts = dep.split("_")
        site = parts[0]
        treatment = TREAT_LABEL.get(parts[-1], parts[-1])
        lat, lon = fake_latlon(dep, site, treatment)
        dep_rows.append(
            {
                "deployment_id": dep,
                "site": site,
                "depth": DEPTH_LABEL.get(parts[1], parts[1]) if len(parts) > 1 else "?",
                "treatment": treatment,
                "lat": lat,
                "lon": lon,
                "expert_annotated": "—" if dep_unann else "✅",
                "fish_expert_peak": expert_peak,
                "expert_total_peak": expert_total_peak,
                "fish_ml_peak": ml_peak,
                "fish_call_peak": ml_call(expert_total_peak, ml_peak),
                "fish_flip_peak": presence_flip(expert_total_peak, ml_peak),
                "fish_abs_diff_peak": (
                    None
                    if expert_total_peak is None
                    else abs(expert_total_peak - ml_peak)
                ),
                "fish_severity_peak": severity(expert_total_peak, ml_peak),
                "model": model_by_dep.get(dep, ""),
            }
        )

    dep_df = pd.DataFrame(dep_rows).sort_values(
        "fish_severity_peak", ascending=False, na_position="last"
    )
    # lat/lon feed the map, hidden from the table.
    st.dataframe(dep_df.drop(columns=["lat", "lon"]), hide_index=True, width="stretch")

    plot_df = dep_df.dropna(subset=["expert_total_peak"])
    if not plot_df.empty:
        long = plot_df.melt(
            id_vars=["deployment_id"],
            value_vars=["expert_total_peak", "fish_ml_peak"],
            var_name="Source",
            value_name="Fish MaxN (peak)",
        )
        long["Source"] = long["Source"].map(
            {"expert_total_peak": "Expert (Σ species)", "fish_ml_peak": "ML"}
        )
        fig2 = px.bar(
            long,
            x="deployment_id",
            y="Fish MaxN (peak)",
            color="Source",
            barmode="group",
            title="Peak fish MaxN per deployment, expert (all species) vs ML",
        )
        fig2.update_layout(height=420, xaxis_title=None)
        st.plotly_chart(fig2, key=k("dep_bar"))

    # ── Map: deployment locations (synthetic coordinates) ────────────────────

    st.divider()
    st.header("🗺️ Deployment map")
    st.warning(
        "**Coordinates are fake.** The dataset carries no GPS, and real BUV "
        "positions are withheld (illegal-fishing / poaching risk). Points are "
        "arbitrary anchors in the Hauraki Gulf, clustered per site with controls "
        "nudged east of farms, for layout only, not real locations.",
        icon="📍",
    )

    size_choice = st.radio(
        "Size markers by",
        ["ML peak", "Expert peak", "Severity"],
        horizontal=True,
        key=k("map_size"),
        help="Marker area scales with the chosen per-deployment metric.",
    )
    size_col = {
        "ML peak": "fish_ml_peak",
        "Expert peak": "expert_total_peak",
        "Severity": "fish_severity_peak",
    }[size_choice]

    map_df = dep_df.copy()
    # Expert peak / severity are None on unannotated deployments and the count
    # can be 0, coerce and floor at +1 so every point still renders.
    map_df["_size"] = pd.to_numeric(map_df[size_col], errors="coerce").fillna(0) + 1

    fig_map = px.scatter_map(
        map_df,
        lat="lat",
        lon="lon",
        color="treatment",
        size="_size",
        size_max=28,
        zoom=8.5,
        hover_name="deployment_id",
        hover_data={
            "site": True,
            "depth": True,
            "treatment": True,
            "expert_total_peak": True,
            "fish_ml_peak": True,
            "fish_severity_peak": True,
            "lat": False,
            "lon": False,
            "_size": False,
        },
        color_discrete_map={"Mussel farm": "#EF553B", "Control": "#636EFA"},
        map_style="open-street-map",
        title=f"Deployment locations, marker size = {size_choice} (fake coordinates)",
    )
    fig_map.update_layout(
        height=560, margin=dict(l=0, r=0, t=40, b=0), legend_title_text="Treatment"
    )
    st.plotly_chart(fig_map, key=k("map"))

    # ── Section 3: Mussel farm vs control (the paper's question) ─────────────

    st.divider()
    st.header("3. Mussel farm vs control, the paper's question")
    st.caption(
        "Underwood & Jeffs: mussel farms had **significantly higher fish abundance** "
        "than soft-sediment controls (surface: 3,074 fish in farms vs 50 in controls). "
        "Surface & benthic are analysed **separately** (different species + "
        "visibility). One value per deployment = **peak across its 6-min segments** "
        "(the paper's repeated-measures fix). Expert abundance = Σ species MaxN; ML "
        "abundance = peak fish/frame. **Only Esk & Motukopake have controls**. Rat & "
        "Whanganui are farm-only, so controls are pooled across sites (the paper used "
        "Site as a 6-level factor, not site-paired farm/control)."
    )

    try:
        from scipy import stats as _scipy_stats
    except Exception:
        _scipy_stats = None

    eco = dep_df[dep_df["treatment"].isin(["Mussel farm", "Control"])]

    for depth in ["Surface", "Benthic"]:
        sub = eco[eco["depth"] == depth]
        if sub.empty:
            continue
        st.subheader(depth)
        rows3 = []
        for _, r in sub.iterrows():
            if r["expert_total_peak"] is not None:
                rows3.append(
                    {
                        "treatment": r["treatment"],
                        "source": "Expert",
                        "abundance": r["expert_total_peak"],
                        "deployment_id": r["deployment_id"],
                    }
                )
            rows3.append(
                {
                    "treatment": r["treatment"],
                    "source": "ML",
                    "abundance": r["fish_ml_peak"],
                    "deployment_id": r["deployment_id"],
                }
            )
        long3 = pd.DataFrame(rows3)

        fig3 = px.box(
            long3,
            x="treatment",
            y="abundance",
            color="source",
            points="all",
            category_orders={"treatment": ["Mussel farm", "Control"]},
            hover_data=["deployment_id"],
            title=f"{depth}, peak fish abundance per deployment (farm vs control)",
            labels={"abundance": "Peak fish abundance", "treatment": ""},
        )
        fig3.update_layout(height=420)
        st.plotly_chart(fig3, key=k(f"box_{depth}"))

        # Kruskal-Wallis farm vs control (paper's test, zero-inflated,
        # non-parametric)
        if _scipy_stats is not None:
            lines = []
            for src in ["Expert", "ML"]:
                farm = long3[
                    (long3.source == src) & (long3.treatment == "Mussel farm")
                ]["abundance"].dropna()
                ctrl = long3[(long3.source == src) & (long3.treatment == "Control")][
                    "abundance"
                ].dropna()
                if len(farm) >= 2 and len(ctrl) >= 2:
                    _, p = _scipy_stats.kruskal(farm, ctrl)
                    sig = "✅ significant" if p < 0.05 else "ns"
                    lines.append(
                        f"**{src}**, farm median {farm.median():.0f} (n={len(farm)}) vs "
                        f"control median {ctrl.median():.0f} (n={len(ctrl)}); "
                        f"Kruskal-Wallis p={p:.3f} ({sig})"
                    )
                else:
                    lines.append(
                        f"**{src}**, too few deployments in both groups to test "
                        f"(farm n={len(farm)}, control n={len(ctrl)})"
                    )
            st.markdown("  \n".join(lines))
        else:
            st.info("scipy not available, install it for the Kruskal-Wallis test.")

    st.info(
        "**Partial replication so far.** This reflects only the day(s) loaded. The "
        "paper used **18–20 Jan** (not the 21st) and reports **abundance + Shannon "
        "diversity + evenness**, diversity/evenness need the **species model** days "
        "(19th/20th), not the binary CFD 18th. Load those to complete the picture."
    )

    # ── Section 4: MaxN frame per deployment ─────────────────────────────────

    st.divider()
    st.header("4. MaxN frame per deployment")

    manifests = list(FRAMES_ROOT.rglob("deployment_maxn_manifest.csv"))
    if not manifests:
        st.info(
            "No MaxN frames yet. On NeSI run "
            "`scripts/wip/render_deployment_maxn_frames.py` for a volume, then scp its "
            f"output folder (JPEGs + `deployment_maxn_manifest.csv`) into "
            f"`{FRAMES_ROOT}/<volume>/`. This panel then shows one annotated MaxN frame "
            "per deployment, biggest ML-vs-expert disagreements first."
        )
    else:
        from pathlib import Path

        man = pd.concat(
            [pd.read_csv(m).assign(_dir=str(m.parent)) for m in manifests],
            ignore_index=True,
        )
        sev = dep_df.set_index("deployment_id")
        man = man[man["deployment_id"].isin(sev.index)].copy()
        man["severity"] = man["deployment_id"].map(sev["fish_severity_peak"])
        man["expert_peak"] = man["deployment_id"].map(sev["expert_total_peak"])
        man = man.sort_values("severity", ascending=False, na_position="last")

        st.caption(
            "One annotated frame at each deployment's MaxN peak (busiest frame), "
            "ordered by **biggest ML-vs-expert disagreement first** "
            "(`fish_severity_peak`). 🔴 = ML and expert disagree; ⚠️ = partial "
            "deployment (not all chapters analysed → peak is provisional)."
        )
        only_err = st.checkbox(
            "Show only disagreements (severity > 0)", value=False, key=k("only_err")
        )
        frame_rows = man[man["severity"].fillna(0) > 0] if only_err else man

        items = list(frame_rows.iterrows())
        per_row = 3
        for i in range(0, len(items), per_row):
            for col, (_, r) in zip(st.columns(per_row), items[i : i + per_row]):
                with col:
                    sev_val = r["severity"]
                    badge = "🔴 " if pd.notna(sev_val) and sev_val > 0 else ""
                    if not bool(r.get("complete", True)):
                        badge += "⚠️ "
                    hp = int(r["expert_peak"]) if pd.notna(r["expert_peak"]) else "—"
                    sv = sev_val if pd.notna(sev_val) else "—"
                    caption = (
                        f"{badge}{r['deployment_id']} · expert {hp} / "
                        f"ML {int(r['maxn_count'])} · sev {sv}"
                    )
                    fname = str(r["frame_file"]) if pd.notna(r["frame_file"]) else ""
                    fpath = Path(r["_dir"]) / fname if fname else None
                    if fpath and fpath.exists():
                        st.image(str(fpath), caption=caption, use_container_width=True)
                    else:
                        st.markdown(
                            f"**{caption}**  \n_(no detections, no MaxN frame)_"
                        )

    # ── Methodology ──────────────────────────────────────────────────────────

    with st.expander("Methodology & caveats", expanded=False):
        st.markdown(
            f"""
**Time mapping**, each expert interval row carries `video_symlink`
(→ deployment + chapter) and `start_sec_in_chapter`. The 6-min window is
`[start, start+{INTERVAL_SEC}s)`; when it overruns the ~{CHAPTER_SEC}s chapter
length the remainder is read from the next chapter's raw CSV.

**ML MaxN** = max over frames in the window of (detections in that frame, conf ≥
threshold). Fish-any counts all classes; per-species counts one class.

**Expert MaxN**, `fish (any)` = max across all `… MaxN` columns for the row;
per-species taken directly from `Snapper MaxN` / `Spotty MaxN`. Per deployment,
`expert_total_peak` (the highest Σ-species interval) is the all-species quantity
the ML peak is judged against; `fish_expert_peak` (highest single-species MaxN)
is kept for reference.

**Per-species only where the model has the class.** Loaded models:
`{", ".join(models)}`. Active per-species comparisons:
`{", ".join(active_species) or "none (fish-any only, e.g. CFD binary)"}`.

**None vs 0 (the `unannotated` rule)**, for an **annotated** deployment a blank
species cell means "a expert looked and saw none" → counted as **0**. Only rows
flagged `unannotated=True` (placeholder rows where no expert ever looked) stay
**None**, no expert count to compare against, so they're excluded from every
error column and the scatter.

**Error columns**, `fish_call` is the direction (**−** ML too few, **+** too
many, **=** exact); `fish_abs_diff` is the raw count error; **`fish_severity` =
|expert − ml|² / max(expert, ml)** = (count missed) × (fraction missed), so a busy
scene called empty (20→0 ⇒ 20) outranks a small undercount on a busy scene
(100→80 ⇒ 4) and a tiny miss (1→0 ⇒ 1). `fish_flip` **⚠️** marks a
presence/absence disagreement, worth a look because a single individual *could*
be a notable species, though the binary CFD model can't tell which.

**Caveats**: bait-soak (first 6 min) excluded from expert annotation; the legacy
+6-min offset is already applied in `start_sec_in_chapter`. ML inference conf
floor was 0.15. The diagonal on the scatter is exact agreement.
"""
        )
