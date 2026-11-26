"""Generate a fully fake, structurally exact copy of the BUV S3 bucket for testing.

Writes a local tree that mirrors the real bucket root, so the whole thing can be
synced to a test bucket as-is:

    process_files/sample_bucket/
    ├── metadata/sharepoint_lists/   BUV Deployment.csv, BUV Survey Metadata.csv,
    │                                BUV Survey Sites.csv, Marine Reserves.csv,
    │                                BUV Species.csv
    └── media/{SurveyID}/{DropID}/{DropID}.mp4

    python scripts/generate_sample_metadata.py [--metadata-only]
    aws s3 sync process_files/sample_bucket/ s3://<your-test-bucket>/

Everything is fictional except the species subset (real names + AphiaIDs so the
species registry still resolves them). Two made-up reserves give two coordinate
clusters in open ocean — nowhere near a real reserve, but inside the validator's
lat/lon ranges so ingest accepts every row.

Videos are genuinely full-length (~32 min, matching the CSV sampling window) but
only one minute has moving content — fish-ish shapes swimming across an
underwater-toned background at t=120-180s, placed AFTER the largest possible
SamplingStart (120s) so the sampling window always contains it. The rest is a
static frame, which H.264 compresses to ~1-2 MB per file. Real duration means
ffprobe/cv2/clip extraction all behave normally — no container metadata tricks.
"""

import argparse
import csv
import math
import random
import zlib
from pathlib import Path

import cv2
import numpy as np

from spyfish.config.wrapper import config

OUT_DIR = Path("process_files/sample_bucket")
# No typed accessor exists for this one; ingest.py reads it off the mapping too.
IS_BAD_COL = config.csv_mapping["is_bad_deployment_column"]
VIDEO_FPS = 10  # low fps keeps files small; ML stride adapts to actual fps
VIDEO_SIZE = (640, 360)
FISH_START_SECONDS = 120  # after max SamplingStart, so the window always sees it
FISH_END_SECONDS = 180
rng = random.Random(42)  # deterministic: re-running yields identical files

# Two fictional reserves = the two coordinate clusters. Sites scatter ~2 km
# around each centre; deployments land ~150 m off their site's planned spot.
RESERVES = [
    {
        "acronym": "TSB",
        "title": "Testy Beach Marine Reserve",
        "region": "Test Region North",
        "center": (-39.62, 177.85),
        "surveys": [
            ("20230118", "18/01/2023"),
            ("20240110", "10/01/2024"),
            ("20250115", "15/01/2025"),
        ],
        "ranger": "T. Kōura",
    },
    {
        "acronym": "SBY",
        "title": "Samply Bay Marine Reserve",
        "region": "Test Region South",
        "center": (-46.35, 171.20),
        "surveys": [
            ("20230228", "28/02/2023"),
            ("20240215", "15/02/2024"),
            ("20250220", "20/02/2025"),
        ],
        "ranger": "M. Pātiki",
    },
]

SITES_PER_RESERVE = 6  # last two are unprotected control sites outside the reserve

# Real species only (validation-passing AphiaIDs); includes the three
# indicator species from config reporting so dashboards have data to show.
SPECIES = [
    (398546, "Snapper", "Pagrus auratus"),
    (276989, "Blue cod", "Parapercis colias"),
    (382879, "Rock lobster", "Jasus edwardsii"),
    (281786, "Spotty", "Notolabrus celidotus"),
    (281653, "Tarakihi", "Nemadactylus macropterus"),
    (219697, "Barracouta", "Thyrsites atun"),
    (105923, "Spiny dogfish", "Squalus acanthias"),
    (275992, "Giant stargazer", "Kathetostoma giganteum"),
]


def write_csv(rel_key: str, header: list[str], rows: list[dict]) -> None:
    path = OUT_DIR / rel_key
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows):3d} rows → {path}")


def _seafloor_background(frame_rng, w: int, h: int) -> np.ndarray:
    """Murky blue-green gradient with blurred noise — reads as underwater."""
    noise = cv2.GaussianBlur(
        frame_rng.integers(0, 80, (h, w, 3), np.uint8), (51, 51), 0
    )
    depth = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    bg = np.zeros((h, w, 3), np.uint8)
    bg[..., 0] = 90 - 40 * depth  # blue fades with depth
    bg[..., 1] = 110 - 30 * depth  # green
    bg[..., 2] = 60 - 30 * depth  # red lowest — underwater light loss
    return cv2.add(bg, noise)


def _draw_fish(frame, cx, cy, size, color, heading):
    """Ellipse body + triangle tail + eye. Cartoonish, but reads as a fish."""
    x, y = int(cx), int(cy)
    body = (int(size), int(size * 0.45))
    cv2.ellipse(frame, (x, y), body, 0, 0, 360, color, -1)
    tail_x = x - heading * int(size * 1.4)
    tail = np.array(
        [
            (x - heading * int(size * 0.8), y),
            (tail_x, y - int(size * 0.5)),
            (tail_x, y + int(size * 0.5)),
        ]
    )
    cv2.fillPoly(frame, [tail], color)
    eye = (x + heading * int(size * 0.6), y - int(size * 0.12))
    cv2.circle(frame, eye, max(2, int(size * 0.08)), (230, 230, 230), -1)
    cv2.circle(frame, eye, max(1, int(size * 0.04)), (20, 20, 20), -1)


def write_video(rel_key: str, drop_id: str, duration_seconds: int) -> None:
    """Full-length video, tiny file: static seafloor everywhere except one minute
    of swimming fish at FISH_START-FISH_END. H.264 spends bytes only where
    pixels change, so the static head and tail are nearly free."""
    path = OUT_DIR / rel_key
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = VIDEO_SIZE
    frame_rng = np.random.default_rng(zlib.crc32(drop_id.encode()))
    base = _seafloor_background(frame_rng, w, h)

    def label(frame):
        cv2.putText(
            frame,
            drop_id,
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "FAKE TEST FOOTAGE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        return frame

    # Each fish: speed px/frame, size, muted fishy colour, vertical wobble.
    fish = [
        {
            "speed": float(frame_rng.uniform(2.0, 5.0)),
            "size": float(frame_rng.uniform(18, 42)),
            "y": float(frame_rng.uniform(0.25, 0.8)) * h,
            "phase": float(frame_rng.uniform(0, 2 * math.pi)),
            "offset": float(frame_rng.uniform(0, w)),
            "heading": int(frame_rng.choice([-1, 1])),
            "color": tuple(int(c) for c in frame_rng.integers(90, 180, 3)),
        }
        for _ in range(int(frame_rng.integers(3, 6)))
    ]

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"avc1"), VIDEO_FPS, (w, h)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cv2 could not open H.264 writer for {path}")

    static = label(base.copy())
    for _ in range(FISH_START_SECONDS * VIDEO_FPS):
        writer.write(static)
    for i in range((FISH_END_SECONDS - FISH_START_SECONDS) * VIDEO_FPS):
        frame = base.copy()
        for f in fish:
            travel = f["offset"] + f["heading"] * f["speed"] * i
            cx = travel % (w + 200) - 100  # swim off one edge, re-enter the other
            cy = f["y"] + 15 * math.sin(f["phase"] + i / 8)
            _draw_fish(frame, cx, cy, f["size"], f["color"], f["heading"])
        writer.write(label(frame))
    for _ in range((duration_seconds - FISH_END_SECONDS) * VIDEO_FPS):
        writer.write(static)
    writer.release()
    print(
        f"wrote {duration_seconds}s video ({path.stat().st_size / 1e6:.1f} MB) → {path}"
    )


def jitter(value: float, spread: float) -> float:
    return round(value + rng.uniform(-spread, spread), 7)


def build_rows():
    reserves_rows, sites_rows, surveys_rows, deployments_rows = [], [], [], []
    next_id = 90001

    for res_idx, res in enumerate(RESERVES):
        acr = res["acronym"]
        lat0, lon0 = res["center"]

        reserves_rows.append(
            {
                config.reserve_title_column: res["title"],
                config.reserve_acronym_column: acr,
                "MarineReserveID": 9000 + res_idx,
                config.region_column: res["region"],
                "CountryCode": "NZ",
                "Office": "Test Office",
                "Office Contact": res["ranger"],
                "ShortID": "",
            }
        )

        # Sites are stable across years — long-term monitoring resurveys the
        # same locations, so one site row serves every annual survey.
        site_coords = {}
        for site_num in range(1, SITES_PER_RESERVE + 1):
            site_id = f"{acr}_{site_num:03d}"
            is_control = site_num > SITES_PER_RESERVE - 2
            site_coords[site_id] = (jitter(lat0, 0.02), jitter(lon0, 0.02))

            sites_rows.append(
                {
                    config.site_id_column: site_id,
                    config.link_to_marine_reserve_column: res["title"],
                    config.site_name_column: f"Site {site_num}",
                    "SiteCode": f"{site_num}a",
                    "SiteExposure": rng.choice(["Exposed", "Sheltered"]),
                    config.protection_status_column: (
                        "No protection" if is_control else "Type I MPA (Marine Reserve)"
                    ),
                    "IsControlSite": "TRUE" if is_control else "FALSE",
                    "ControlToMR01": res["title"] if is_control else "",
                    config.targeted_latitude_column: site_coords[site_id][0],
                    config.targeted_longitude_column: site_coords[site_id][1],
                    "geodeticDatum": "WGS84",
                    "countryCode": "NZ",
                }
            )

        for survey_date, event_date in res["surveys"]:
            survey_id = f"{acr}_{survey_date}_BUV"

            surveys_rows.append(
                {
                    "SurveyName": f"{res['title']} BUV {survey_date[:4]}",
                    "EncoderName": res["ranger"],
                    "DateEntry": event_date,
                    "OfficeContact": res["ranger"],
                    config.link_to_marine_reserve_column: res["title"],
                    config.reserve_acronym_column: acr,
                    config.survey_id_column: survey_id,
                    "SurveyStartDate": event_date,
                    "SurveyLeaderName": res["ranger"],
                    "SiteSelectionDesign": "Haphazard",
                    "FishMultiSpecies": "True",
                    "IsLongTermMonitoring": "True",
                    "RightsHolder": "DOC",
                    "RecordType": "Fish",
                    "Vessel": "RV Makebelieve",
                    "BaitSpecies": "Pilchard",
                    "BaitAmount": "500.0",
                    "SurveyType": "BUV",
                    "BUVType": "L-Frame",
                    "CameraModel": "GoPro Hero 8",
                }
            )

            for site_num in range(1, SITES_PER_RESERVE + 1):
                site_id = f"{acr}_{site_num:03d}"
                site_lat, site_lon = site_coords[site_id]

                # 1-2 replicates per site; one deliberate bad deployment per
                # survey (site 3, replicate 1) to exercise ingest_status=excluded.
                for replicate in range(1, rng.randint(1, 2) + 1):
                    drop_id = config.validate_drop_id(
                        f"{survey_id}_{site_id}_{replicate:02d}"
                    )
                    is_bad = site_num == 3 and replicate == 1
                    sampling_start = rng.randint(30, 120)
                    duration = sampling_start + 1860
                    row = {
                        config.drop_id_column: drop_id,
                        config.survey_id_column: survey_id,
                        config.site_id_column: site_id,
                        config.latitude_column: jitter(site_lat, 0.0015),
                        config.longitude_column: jitter(site_lon, 0.0015),
                        "EventDate": event_date,
                        "Created By": res["ranger"],
                        "TideLevel": rng.choice(["Low", "High", ""]),
                        "Weather": rng.choice(["<0.1m_Swell", "Calm", ""]),
                        config.replicate_column: replicate,
                        "EventTimeStart": f"{rng.randint(8, 15)}:{rng.choice(['00', '15', '30', '45'])}",
                        config.depth_column: rng.randint(8, 40),
                        "RecordedBy": res["ranger"],
                        IS_BAD_COL: "True" if is_bad else "False",
                        "ID": next_id,
                    }
                    if is_bad:
                        bad = "NO VIDEO BAD DEPLOYMENT"
                        row.update(
                            {
                                "NotesDeployment": "Camera flooded (fake test data)",
                                config.file_name_column: bad,
                                config.csv_video_file_link_column: bad,
                                config.csv_sampling_start_column: 0,
                                config.csv_sampling_end_column: 0,
                            }
                        )
                    else:
                        row.update(
                            {
                                "fps": VIDEO_FPS,
                                "duration": duration,
                                config.file_name_column: f"{drop_id}.mp4",
                                config.csv_video_file_link_column: config.get_video_s3_key(
                                    drop_id
                                ),
                                config.csv_sampling_start_column: sampling_start,
                                config.csv_sampling_end_column: sampling_start + 1800,
                            }
                        )
                    deployments_rows.append(row)
                    next_id += 1

    return reserves_rows, sites_rows, surveys_rows, deployments_rows


def build_error_rows(next_id: int) -> list[dict]:
    """Deployment rows that deliberately fail ingest validation, one per
    failure mode, so the error-review dashboard has real content. None of
    them get a video (video_presence=absent on top of the metadata error)."""

    def base(drop_id, survey, site, **overrides):
        row = {
            config.drop_id_column: drop_id,
            config.survey_id_column: survey,
            config.site_id_column: site,
            config.latitude_column: -39.61,
            config.longitude_column: 177.84,
            "EventDate": "15/01/2025",
            "Created By": "T. Kōura",
            config.replicate_column: drop_id.rsplit("_", 1)[-1].lstrip("0") or "0",
            "RecordedBy": "T. Kōura",
            IS_BAD_COL: "False",
            config.file_name_column: f"{drop_id}.mp4",
            config.csv_video_file_link_column: f"media/{survey}/{drop_id}/{drop_id}.mp4",
            config.csv_sampling_start_column: 60,
            config.csv_sampling_end_column: 1860,
            "NotesDeployment": "fake test data — deliberate ingest error",
        }
        row.update(overrides)
        return row

    rows = [
        # Unknown site: TSB_007 is not in BUV Survey Sites.csv → foreign-key error.
        base("TSB_20250115_BUV_TSB_007_01", "TSB_20250115_BUV", "TSB_007"),
        # sampling_start=0 → "missing sampling window metadata" error.
        base(
            "TSB_20250115_BUV_TSB_001_09",
            "TSB_20250115_BUV",
            "TSB_001",
            **{
                config.csv_sampling_start_column: 0,
                config.csv_sampling_end_column: 1800,
            },
        ),
        # Video shorter than a full BUV deployment → sampling_end too small.
        base(
            "SBY_20250220_BUV_SBY_004_09",
            "SBY_20250220_BUV",
            "SBY_004",
            **{config.csv_sampling_end_column: 900},
        ),
        # Latitude far outside the NZ range → value_range error.
        base(
            "SBY_20250220_BUV_SBY_002_09",
            "SBY_20250220_BUV",
            "SBY_002",
            **{config.latitude_column: -12.3456},
        ),
        # FileName doesn't match {DropID}.mp4 → relationship/template error.
        base(
            "TSB_20250115_BUV_TSB_002_09",
            "TSB_20250115_BUV",
            "TSB_002",
            **{config.file_name_column: "GOPR0042.mp4"},
        ),
        # DropID date is DDMMYYYY (hand-typed) → format error; the row is
        # rejected before it ever reaches the deployments table.
        base("TSB_23042025_BUV_TSB_008_01", "TSB_20250115_BUV", "TSB_008"),
    ]
    for i, r in enumerate(rows):
        r["ID"] = next_id + i
    return rows


DEPLOYMENT_HEADER = [
    config.drop_id_column,
    config.survey_id_column,
    config.site_id_column,
    config.latitude_column,
    config.longitude_column,
    "EventDate",
    "Created By",
    "TideLevel",
    "Weather",
    "UnderwaterVisibility",
    config.replicate_column,
    "EventTimeStart",
    "EventTimeEnd",
    config.depth_column,
    "DepthStrata",
    "NZMHCS_Abiotic",
    "NZMHCS_Biotic",
    "NotesDeployment",
    "RecordedBy",
    IS_BAD_COL,
    "fps",
    "duration",
    config.file_name_column,
    config.csv_video_file_link_column,
    config.csv_sampling_start_column,
    config.csv_sampling_end_column,
    "ID",
]
SITES_HEADER = [
    config.site_id_column,
    config.link_to_marine_reserve_column,
    config.site_name_column,
    "SiteCode",
    "SiteExposure",
    config.protection_status_column,
    "ProtectionStatusDetails",
    "IsControlSite",
    "ControlToMR01",
    "ControlToMR02",
    "ControlToMR03",
    config.targeted_latitude_column,
    config.targeted_longitude_column,
    "geodeticDatum",
    "countryCode",
]
SURVEYS_HEADER = [
    "SurveyName",
    "EncoderName",
    "DateEntry",
    "OfficeContact",
    config.link_to_marine_reserve_column,
    config.reserve_acronym_column,
    config.survey_id_column,
    "SurveyStartDate",
    "ContractorName",
    "ContractNumber",
    "SurveyLeaderName",
    "StratifiedBy",
    "SiteSelectionDesign",
    "SurveyVerbatim",
    "FishMultiSpecies",
    "IsLongTermMonitoring",
    "RightsHolder",
    "RecordType",
    "IsMoreHabitatData",
    "LinkReport01",
    "LinkToOriginalData",
    "Vessel",
    "BaitSpecies",
    "BaitAmount",
    "SurveyType",
    "BUVType",
    "CameraModel",
    "LensModel",
]
RESERVES_HEADER = [
    config.reserve_title_column,
    config.reserve_acronym_column,
    "MarineReserveID",
    config.region_column,
    "CountryCode",
    "Office",
    "Office Contact",
    "ShortID",
]
SPECIES_HEADER = ["AphiaID", "CommonName", "ScientificName", "TaxonRank"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="skip video generation, write only the CSVs",
    )
    args = parser.parse_args()

    reserves_rows, sites_rows, surveys_rows, deployments_rows = build_rows()
    error_rows = build_error_rows(next_id=99001)
    species_rows = [
        {
            "AphiaID": aphia,
            "CommonName": common,
            "ScientificName": scientific,
            "TaxonRank": "Species",
        }
        for aphia, common, scientific in SPECIES
    ]

    write_csv(
        config.s3_sharepoint_deployment_csv,
        DEPLOYMENT_HEADER,
        deployments_rows + error_rows,
    )
    write_csv(config.s3_sharepoint_site_csv, SITES_HEADER, sites_rows)
    write_csv(config.s3_sharepoint_survey_csv, SURVEYS_HEADER, surveys_rows)
    write_csv(config.s3_sharepoint_reserves_csv, RESERVES_HEADER, reserves_rows)
    write_csv(config.s3_sharepoint_species_csv, SPECIES_HEADER, species_rows)

    if not args.metadata_only:
        good = [r for r in deployments_rows if r[IS_BAD_COL] == "False"]
        for i, row in enumerate(good, 1):
            print(f"[{i}/{len(good)}] {row[config.drop_id_column]}")
            write_video(
                row[config.csv_video_file_link_column],
                row[config.drop_id_column],
                row["duration"],
            )

    print(
        f"\nDone. Sync to your test bucket with:\n"
        f"  aws s3 sync {OUT_DIR}/ s3://<your-test-bucket>/"
    )


if __name__ == "__main__":
    main()
