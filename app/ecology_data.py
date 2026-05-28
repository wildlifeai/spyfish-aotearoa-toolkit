"""Shared ecology data layer for the Streamlit app.

Single source of truth for loading + enriching annotation data. Used by both the
Experiments page (``pages/_advanced/🧪_Experiments.py``) and the Species Search
page (``pages/🔎_Species_Search.py``). Page-specific viz helpers stay in their own
pages — only the page-agnostic loaders and the drop_id/site enrichment live here.
"""

import json
import sqlite3

import pandas as pd
import streamlit as st

from spyfish.config.wrapper import config
from spyfish.database.annotation_manager import AnnotationDatabaseManager

# Source precedence used wherever we collapse to a single "best" annotation per
# (drop, species): expert beats citsci beats ml.
_SOURCE_PRIORITY = {"expert": 0, "citsci": 1, "ml": 2}


@st.cache_data(ttl=300)
def load_maxn() -> pd.DataFrame:
    return AnnotationDatabaseManager().get_maxn_summary()


@st.cache_data(ttl=300)
def search_species_annotations(scientific_name: str) -> pd.DataFrame:
    """Every raw annotation row for one species — cached per species.

    Unlike `get_maxn_summary()` (peak per drop only), this returns every
    time-window observation so the species-search view can list every
    timestamp the species was seen. Cached by argument so each species
    selection is fetched once per session.
    """
    with sqlite3.connect(config.annotations_db_path) as conn:
        return pd.read_sql(
            "SELECT drop_id, scientific_name, time_of_max, time_of_max_seconds, "
            "max_interval, annotated_by, confidence_agreement, external_id "
            "FROM annotations WHERE scientific_name = ? "
            "ORDER BY drop_id, time_of_max_seconds",
            conn,
            params=(scientific_name,),
        )


@st.cache_data(ttl=300)
def load_sites() -> pd.DataFrame:
    with sqlite3.connect(config.db_path) as conn:
        return pd.read_sql(
            "SELECT site_id, site_name, protection_status FROM sites", conn
        )


@st.cache_data(ttl=3600)
def load_common_names() -> dict:
    """scientific_name → 'Common name (Scientific name)' from class_map.json.

    Returns empty dict when the file is missing or for legacy/generic entries.
    """
    path = config.class_map_path
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {
        entry["scientific_name"]: f"{entry['common_name']} ({entry['scientific_name']})"
        for entry in data.values()
        if entry.get("scientific_name")
        and entry.get("common_name")
        and entry["common_name"].lower() not in ("fish", "bait", "unknown")
        and entry["common_name"] != entry["scientific_name"]
    }


def _enrich(df: pd.DataFrame, sites: pd.DataFrame, common_names: dict) -> pd.DataFrame:
    """Parse drop_id segments, join sites, attach display_name. Done once globally."""
    parts = df["drop_id"].str.split("_", expand=True)
    df = df.copy()
    df["reserve_code"] = parts.get(0, pd.Series("", index=df.index)).fillna("")
    df["survey_date"] = pd.to_datetime(parts[1], format="%Y%m%d", errors="coerce")
    df["survey_year"] = df["survey_date"].dt.year
    p3 = parts.get(3, pd.Series("", index=df.index)).fillna("")
    p4 = parts.get(4, pd.Series("", index=df.index)).fillna("")
    df["site_id"] = p3 + "_" + p4
    df["site_id"] = df["site_id"].replace("_", pd.NA)
    df = df.merge(
        sites[["site_id", "site_name", "protection_status"]], on="site_id", how="left"
    )
    df["site_name"] = df["site_name"].fillna(df["site_id"])
    df["protection_status"] = df["protection_status"].fillna("unknown")
    df["display_name"] = df["scientific_name"].map(
        lambda s: common_names.get(s, s) if pd.notna(s) else s
    )
    return df
