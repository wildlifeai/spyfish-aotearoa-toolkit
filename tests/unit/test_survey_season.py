"""Survey season: the July-to-June year every dashboard view reports by."""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px

APP = Path(__file__).resolve().parents[2] / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from doc_report.charting import year_axis  # noqa: E402
from ecology_data import (  # noqa: E402
    add_drop_id_columns,
    season_label,
    unparseable_dates,
)


def _seasons(*drop_ids):
    frame = add_drop_id_columns(pd.DataFrame({"drop_id": list(drop_ids)}))
    return frame.set_index("drop_id")["survey_season"]


def test_season_boundary_is_between_june_and_july():
    seasons = _seasons(
        "KSF_20240630_BUV_KSF_085_01",
        "KSF_20240701_BUV_KSF_085_01",
    )
    assert seasons["KSF_20240630_BUV_KSF_085_01"] == 2023
    assert seasons["KSF_20240701_BUV_KSF_085_01"] == 2024


def test_december_and_january_of_one_summer_share_a_season():
    seasons = _seasons(
        "KSF_20231212_BUV_KSF_085_01",
        "KSF_20240124_BUV_KSF_085_01",
    )
    assert seasons.nunique() == 1
    assert season_label(seasons.iloc[0]) == "2023/24"


def test_season_label_names_both_years():
    assert season_label(2026) == "2026/27"
    assert season_label(1999) == "1999/00"
    assert season_label(float("nan")) is None


def test_plausibility_uses_the_calendar_date_not_the_season():
    # January 1998 is season 1997, below the floor as a season number, but the
    # date itself is the legitimate Poor Knights 1998 survey.
    frame = add_drop_id_columns(
        pd.DataFrame(
            {
                "drop_id": [
                    "PKI_19980115_BUV_PKI_001_01",
                    "KSF_23042026_BUV_KSF_085_01",
                ]
            }
        )
    )
    assert frame.loc[0, "survey_season"] == 1997
    assert unparseable_dates(frame)["drop_id"].tolist() == [
        "KSF_23042026_BUV_KSF_085_01"
    ]


def test_year_axis_labels_every_season_in_the_span():
    fig = px.bar(pd.DataFrame({"season": [2021, 2023], "n": [1, 2]}), x="season", y="n")
    year_axis(fig)
    assert list(fig.layout.xaxis.tickvals) == [2021, 2022, 2023]
    assert list(fig.layout.xaxis.ticktext) == ["2021/22", "2022/23", "2023/24"]
