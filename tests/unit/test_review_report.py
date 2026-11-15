"""Unit tests for the Survey review HTML export (`app/doc_report/review_report.py`).

The file is sent to people without the dashboard, so nobody is beside it to
say a number is wrong: what it prints has to be what the page worked out.
"""

import math
import os
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

APP = Path(__file__).resolve().parents[2] / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from doc_report import review_data, review_report  # noqa: E402

from spyfish.config.wrapper import config  # noqa: E402

SURVEY = "KSF_20240124_BUV"
INSIDE = "KSF_20240124_BUV_KSF_085_01"
OUTSIDE = "KSF_20240124_BUV_KSF_090_01"
UNKNOWN = "KSF_20240124_BUV_KSF_099_01"
COD = "Parapercis colias"
LABELS = {COD: "Blue cod", "Chrysophrys auratus": "Snapper"}
SOURCE_LABELS = {"expert": "BIIGLE expert", "ml": "ML"}
GROUPS = ["Protected", "Unprotected", "Not recorded"]


def _row(drop_id, group, status="ok", sources=("expert",), slate="confirmed"):
    return {
        "drop_id": drop_id,
        "Deployment": drop_id.split("_BUV_")[1],
        "Site": "Te <Awaatu>",
        "Protection": group,
        "protection_status": None if group == "Not recorded" else "Marine Reserve",
        "Slate": slate,
        "Annotated by": ", ".join(SOURCE_LABELS[s] for s in sources),
        "sources": set(sources),
        "Status": status,
        "Why": "",
        "Notes": "",
        "video_presence": "present",
        "depth": 12.5,
        "bad": False,
        "eligible": status == "ok",
    }


def _build(tmp_path, sources=("expert", "ml"), slates=True):
    drop_dir = tmp_path / SURVEY / INSIDE
    review = drop_dir / config.review_dir_name
    review.mkdir(parents=True, exist_ok=True)
    slate = drop_dir / f"{INSIDE}{config.slate_frame_suffix}"
    Image.new("RGB", (1920, 1080), "white").save(slate, "JPEG")
    frame = review / f"{INSIDE}__frame_100.0s__maxn_expert_7_Parapercis_colias.jpg"
    Image.new("RGB", (1920, 1080), "navy").save(frame, "JPEG")
    os.utime(frame, (1000, 1000))
    pd.DataFrame([{"file": frame.name, "seconds": 100.0, "agreement": "none"}]).to_csv(
        review / config.review_manifest_name, index=False
    )
    table = pd.DataFrame(
        [
            _row(INSIDE, "Protected"),
            _row(OUTSIDE, "Unprotected", slate="missing"),
            _row(
                UNKNOWN, "Not recorded", review_report.NOT_IN_DB, (), "no board shown"
            ),
        ]
    ).set_index("drop_id")
    ann = pd.DataFrame(
        [(INSIDE, COD, 7, "expert"), (OUTSIDE, None, 0, "expert")],
        columns=["drop_id", "scientific_name", "max_interval", "source"],
    )
    chosen, per_drop = review_data.best_source_maxn(
        ann, [INSIDE, OUTSIDE], list(sources)
    )
    return review_report.build_html(
        SURVEY,
        table,
        chosen,
        per_drop,
        review_data.collect_frames(tmp_path, str),
        list(sources),
        label=lambda sp: LABELS.get(sp, sp),
        source_labels=SOURCE_LABELS,
        group_order=GROUPS,
        indicators=[COD, "Chrysophrys auratus"],
        no_board={UNKNOWN},
        slates=slates,
    )


def test_page_carries_the_numbers_the_view_worked_out(tmp_path):
    page = _build(tmp_path)

    assert f"<title>{SURVEY} review</title>" in page
    # the deployment not in the pipeline DB is listed but not counted
    assert "<dt>Deployments in survey</dt><dd>2</dd>" in page
    assert "<dt>Slates found</dt><dd>1 of 2</dd>" in page
    # the table links to each deployment's section
    assert f"<a href='#{INSIDE}'>KSF_085_01</a>" in page
    assert f"<h3 id='{INSIDE}'>" in page
    # one annotated deployment a side: blue cod 7 inside, 0 outside
    assert "<b>7.00</b>" in page and "<b>0.00</b>" in page
    assert "Blue cod <b>7</b>" in page
    assert "nothing seen" in page
    assert "Not seen on any annotated deployment of this survey: Snapper." in page
    assert "<svg class='sp'" in page


def test_images_are_inside_the_file_with_their_agreement_colour(tmp_path):
    page = _build(tmp_path)

    assert page.count("src='data:image/jpeg;base64,") == 2  # slate, MaxN frame
    assert "<img class='agree-none'" in page
    assert "Blue cod MaxN 7 at 00:01:40" in page


def test_about_explains_only_what_the_page_shows(tmp_path):
    page = _build(tmp_path)

    assert "<b>BIIGLE expert</b> counts go through" in page
    assert "<b>Frame colour</b>" in page
    assert "<b>ML</b> counts are" not in page
    assert review_data.RANDOM_NOTE not in page
    assert "No slate: the video never shows a board" in page
    assert "No slate found." in page


def test_without_slates_no_board_picture_is_in_the_file(tmp_path):
    # the board has the GPS position on it; the default leaves it out
    page = _build(tmp_path, slates=False)

    assert page.count("src='data:image/jpeg;base64,") == 1
    assert "slaterow" not in page.split("</style>")[1]
    assert "No slate found." not in page
    # found or not is still said, in the table
    assert "<td class='warn'>missing</td>" in page


def test_text_from_the_data_is_escaped(tmp_path):
    page = _build(tmp_path)

    assert "Te &lt;Awaatu&gt;" in page
    assert "<Awaatu>" not in page


def test_box_stats_keeps_a_far_value_out_of_the_whisker():
    stats = review_report.box_stats(pd.Series([0, 0, 0, 0, 0, 0, 0, 5, 5]))

    assert (stats["q1"], stats["med"], stats["q3"]) == (0, 0, 0)
    assert (stats["lo"], stats["hi"]) == (0, 0)
    assert stats["out"] == [(5.0, 2)]  # two deployments at 5: two dots


def test_nice_max():
    assert [review_report.nice_max(v) for v in (1, 3, 39, 50, 51)] == [
        1,
        5,
        50,
        50,
        100,
    ]
    assert math.log1p(review_report.nice_max(0.5)) > 0  # the axis never has zero width
