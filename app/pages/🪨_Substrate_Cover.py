"""
Substrate Cover (CMECS), percent-cover of seabed substrate from Biigle.

Downloads a Biigle volume's annotation report, measures every CMECS substrate
shape (magic-wand Polygon OR traced LineString) as a closed polygon area, and
reports per-image cover percentages:

- pct_of_annotated (primary): each substrate's share of the annotated area on an
  image. Categories sum to 100%, the CMECS composition.
- pct_of_image: each substrate's share of the whole image (can sum to <100% when
  seabed is left unannotated, or >100% when outlines overlap).

Substrate is identified by label-tree membership (config.biigle_substrate_label_tree_id),
so fish-size LineStrings (species / "Scale bar" labels) are never miscounted.
"""

import sys
from pathlib import Path

# Allow launching this page directly (streamlit run pages/_advanced/<this>.py):
# Streamlit only puts the page's own folder on sys.path, so the shared `utils`
# module that lives in app/ wouldn't resolve. parents[1] is the app/ dir: this
# page sits in `pages/`, one level under app/, not in `pages/_advanced/`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from spyfish.biigle.biigle_parser import BiigleParser  # noqa: E402
from spyfish.config.wrapper import config  # noqa: E402

st.set_page_config(page_title="Substrate Cover", page_icon="🪨", layout="wide")
st.title("🪨 Substrate Cover (CMECS)")
# Rendered for every page by the entrypoint now.
st.markdown(
    "Measure percent-cover of seabed substrate from a Biigle volume.  \n"
    "Works for both magic-wand **Polygons** and traced **LineStrings**, each is "
    "measured as the area it encloses, then divided by the image.  \n"
    "Find the Volume ID in the Biigle URL, e.g. `https://biigle.de/volumes/25173` → `25173`."
)

# ── Volume + credentials ─────────────────────────────────────────────────────

volume_id_str = st.text_input(
    "Volume ID",
    placeholder="Enter volume ID (number)",
    help="ID of the Biigle volume you want to measure. Found in the URL on Biigle.",
    key="substrate_volume_id",
).strip()

with st.form("substrate_form"):
    email = st.text_input(
        "Biigle Email",
        value=config.email or "",
        placeholder="you@example.com",
        help="The email you use to sign in to Biigle.",
    ).strip()
    token = st.text_input(
        "Biigle API Token",
        type="password",
        placeholder="Paste your token…",
        help=(
            "Find your token at https://biigle.de/settings/tokens.  \n"
            "Treat this like a password. You can also set BIIGLE_API_TOKEN in your .env file."
        ),
    ).strip()
    submitted = st.form_submit_button("Measure substrate cover", type="primary")

# ── Fetch, measure & display ─────────────────────────────────────────────────

if submitted:
    if not (email and token):
        st.error("Please provide a valid Email and Token.")
        st.stop()
    if not volume_id_str.isdigit():
        st.error("Please provide a numeric Volume ID.")
        st.stop()

    volume_id = int(volume_id_str)
    parser = BiigleParser(email=email, token=token)

    try:
        # Substrate is normally drawn on image volumes; pick the report type from
        # the volume's media_type, defaulting to image if the lookup fails.
        with st.spinner("Looking up volume…"):
            try:
                media_type = parser.biigle_handler.get_volume_info(volume_id).get(
                    "media_type", "image"
                )
            except Exception:
                media_type = "image"
        report_type = (
            config.annotation_report_type_video
            if media_type == "video"
            else config.annotation_report_type_images
        )

        with st.spinner("Creating report and downloading from Biigle…"):
            raw_df = parser.download_volume_annotations(
                volume_id=volume_id, type_id=report_type
            )

        if raw_df.empty:
            st.warning(f"No annotations found for volume {volume_id}.")
            st.stop()

        # Labels that count as substrate (one API fetch for the whole tree).
        # Keep the full label list, its parent_id links let us reconstruct each
        # category's CMECS path without parsing the report's hierarchy string.
        with st.spinner("Fetching substrate label tree…"):
            tree_labels = parser.biigle_handler.get_label_tree_labels(
                config.biigle_substrate_label_tree_id
            )
        substrate_label_ids = {int(lbl["id"]) for lbl in tree_labels}

        substrate_df = parser.process_substrate(raw_df, substrate_label_ids)

        if substrate_df.empty:
            st.warning(
                f"No substrate (CMECS tree {config.biigle_substrate_label_tree_id}) "
                f"annotations found in volume {volume_id}."
            )
            st.stop()

        n_images = substrate_df["image"].nunique()
        n_categories = substrate_df["substrate"].nunique()
        st.success(f"Measured substrate cover for volume {volume_id} ({media_type}).")
        c1, c2, c3 = st.columns(3)
        c1.metric("Images", n_images)
        c2.metric("Substrate categories", n_categories)
        c3.metric("Annotations measured", len(substrate_df))

        # Composition (primary): substrate × image, normalised to 100% per image.
        composition = substrate_df.pivot_table(
            index="image",
            columns="substrate",
            values="pct_of_annotated",
            fill_value=0.0,
        )

        # Nest the columns under their CMECS parents (up to 6 levels:
        # Component → Origin/Setting → Class → Subclass → Group → SubGroup/
        # Community). Build each category's path from the label TREE's parent_id
        # chain. NOT by splitting the report's hierarchy string, because CMECS
        # label names contain ">" (e.g. "(>5% gravel ...)"), which a naive split
        # would shatter into phantom levels.
        by_id = {int(lbl["id"]): lbl for lbl in tree_labels}

        def _tree_path(label_id: int) -> tuple:
            parts: list = []
            seen: set = set()
            cur = by_id.get(int(label_id))
            while cur is not None and int(cur["id"]) not in seen:
                seen.add(int(cur["id"]))
                parts.append(str(cur["name"]).strip())
                pid = cur.get("parent_id")
                cur = by_id.get(int(pid)) if pid not in (None, "") else None
            return tuple(reversed(parts))

        sub_mask = pd.to_numeric(raw_df["label_id"], errors="coerce").isin(
            substrate_label_ids
        )
        hier_lookup: dict[str, tuple] = {}
        for leaf, lid in zip(
            raw_df.loc[sub_mask, "label_name"].astype(str).str.strip(),
            pd.to_numeric(raw_df.loc[sub_mask, "label_id"], errors="coerce"),
        ):
            if pd.notna(lid):
                hier_lookup.setdefault(leaf, _tree_path(int(lid)) or (leaf,))

        # Cap at 6 levels (Component + 5 CMECS tiers). Anything deeper, e.g. the
        # boulder size-classes that sit below SubGroup, is folded into the 6th
        # label (joined with a space, no separator) so the size detail rides
        # along with its parent instead of adding a 7th header row.
        MAX_LEVELS = 6

        def _cap(path: tuple) -> tuple:
            if len(path) <= MAX_LEVELS:
                return path
            tail = " ".join(p for p in path[MAX_LEVELS - 1 :] if p)
            return path[: MAX_LEVELS - 1] + (tail,)

        paths = {c: _cap(hier_lookup.get(c, (c,))) for c in composition.columns}
        max_depth = max((len(p) for p in paths.values()), default=1)
        padded = {c: p + ("",) * (max_depth - len(p)) for c, p in paths.items()}
        # Order columns by path so siblings (same parents) sit together.
        ordered = sorted(composition.columns, key=lambda c: padded[c])
        composition = composition[ordered]
        composition.columns = pd.MultiIndex.from_tuples([padded[c] for c in ordered])

        # Expanded tidy table: each CMECS level becomes its own column. Used for
        # both the readable Full table (separate columns don't clip like a deep
        # nested header) and the CSV export, so the download carries every level.
        level_cols = [f"Level_{i + 1}" for i in range(max_depth)]
        level_tuples = substrate_df["substrate"].map(
            lambda s: padded.get(s, (s,) + ("",) * (max_depth - 1))
        )
        export_df = substrate_df.drop(columns=["substrate"]).copy()
        for i, col in enumerate(level_cols):
            export_df[col] = [t[i] if i < len(t) else "" for t in level_tuples]
        export_df = export_df[
            [
                config.drop_id_column,
                "image",
                *level_cols,
                "pct_of_annotated",
                "pct_of_image",
                "area_px",
                "image_area_px",
            ]
        ].rename(columns={"image": "Frame"})

        tab1, tab2, tab3 = st.tabs(["Composition (per image)", "Chart", "Full table"])
        with tab1:
            st.caption(
                "One row per image (sums to 100% across the row, share of "
                "annotated substrate area). Columns are substrate categories, "
                "nested by the CMECS hierarchy: each higher level sits in a "
                "header row above its children."
            )
            # st.dataframe/st.table serialize through Arrow, which has no concept
            # of multi-level column headers and flattens the MultiIndex into one
            # row. Render pandas' own HTML instead, it keeps each CMECS level as
            # its own header row, with parents merged via colspan, inside a
            # scrollable box (handles many image rows + wide/deep headers).
            # Display copy: row label is the deployment (DropID), and the frame
            # name moves to the last column. (composition itself stays numeric so
            # the chart below still works.)
            frame_to_drop = (
                substrate_df.drop_duplicates("image")
                .set_index("image")[config.drop_id_column]
                .reindex(composition.index)
            )
            display_df = composition.copy()
            display_df[("Frame",) + ("",) * (max_depth - 1)] = composition.index
            display_df.index = pd.Index(frame_to_drop.values, name="Deployment")
            table_html = display_df.to_html(
                na_rep="", float_format=lambda v: f"{v:.1f}"
            )
            st.markdown(
                f'<div style="overflow:auto; max-height:600px;">{table_html}</div>',
                unsafe_allow_html=True,
            )
        with tab2:
            st.caption("Substrate composition per image (%).")
            # Altair needs flat column names, join the hierarchy path.
            chart_df = composition.copy()
            chart_df.columns = [
                " > ".join(level for level in col if level)
                for col in composition.columns
            ]
            st.bar_chart(chart_df, stack=True)
        with tab3:
            st.caption(
                "One row per (image, substrate), with each CMECS level in its "
                "own column. pct_of_annotated = share of annotated area; "
                "pct_of_image = share of the whole image."
            )
            st.dataframe(export_df, width="content")

        st.download_button(
            label="⬇️ Download substrate cover (CSV)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"substrate_cover_{volume_id}.csv",
            mime="text/csv",
            width="stretch",
        )

    except Exception as e:
        st.error(f"Error measuring substrate cover: {e}")
        st.stop()
