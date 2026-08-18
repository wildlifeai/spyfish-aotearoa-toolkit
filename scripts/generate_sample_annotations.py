"""Populate the annotations DB with fake ML / citsci / expert annotations.

Companion to generate_sample_metadata.py — run AFTER --ingest has loaded the
sample deployments. Writes through the real AnnotationDatabaseManager and then
sync_annotation_counts(), so counts and section statuses land exactly the way
the pipeline would set them (ml_complete / citsci_complete / expert_complete).

Coverage is deliberately uneven to exercise the dashboards:
  - every ok drop gets ML annotations (including 'fish' catch-all rows, which
    should be excluded from abundance figures by non_species_classes)
  - ~2/3 of drops also get citsci annotations
  - ~1/3 also get expert annotations, with counts that sometimes disagree
    with ML/citsci — expert wins, the others stay for auditing
All TimeOfMax values fall inside the fake videos' fish window (120-180s).

    python scripts/generate_sample_annotations.py
"""

import random

from spyfish.config.base import IngestStatus
from spyfish.database.annotation_manager import AnnotationDatabaseManager
from spyfish.database.manager import DatabaseManager

MODEL_NAME = "sample_yolo_v1"  # goes into external_id on ML rows, like ml_runner does
FISH_WINDOW = (125, 175)  # keep a margin inside the videos' 120-180s action minute
rng = random.Random(42)

# Scientific names must match the sample BUV Species.csv so registry lookups work.
# Weights skew toward the indicator species so reserve-effect charts have signal.
SPECIES_POOL = [
    ("Pagrus auratus", 5),
    ("Parapercis colias", 5),
    ("Jasus edwardsii", 3),
    ("Notolabrus celidotus", 3),
    ("Nemadactylus macropterus", 2),
    ("Thyrsites atun", 1),
    ("Squalus acanthias", 1),
    ("Kathetostoma giganteum", 1),
]


def hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def row(drop_id, species, count, source, external_id, confidence):
    t = rng.uniform(*FISH_WINDOW)
    return {
        "drop_id": drop_id,
        "scientific_name": species,
        "time_of_max": hms(t),
        "time_of_max_seconds": round(t, 2),
        "max_interval": count,
        "annotated_by": source,
        "interval_annotation": "30",
        "confidence_agreement": confidence,
        "external_id": external_id,
    }


def main():
    db = DatabaseManager()
    ann_db = AnnotationDatabaseManager()

    drops = sorted(
        drop_id
        for drop_id, d in db.get_all_deployments_map().items()
        if d["ingest_status"] == IngestStatus.OK
    )
    if not drops:
        raise SystemExit(
            "No ok deployments found — run `run_pipeline.py --ingest` first."
        )

    pool = [s for s, w in SPECIES_POOL for _ in range(w)]
    indicators = {s for s, w in SPECIES_POOL if w >= 3}
    annotations = []

    def fake_count(drop_id: str, species: str) -> int:
        """Counts carry a reserve-recovery trend: indicator species at
        protected sites gain ~1 fish per survey year; control sites stay flat.
        Site layout mirrors generate_sample_metadata.py (last 2 sites = control)."""
        parts = drop_id.split("_")
        year, site_num = int(parts[1][:4]), int(parts[4])
        base = rng.randint(1, 3)
        if site_num <= 4 and species in indicators:
            base += (year - 2023) + rng.randint(0, 1)
        return base

    for i, drop_id in enumerate(drops):
        species = sorted(set(rng.sample(pool, k=rng.randint(2, 5))))
        has_citsci = i % 3 != 2  # ~2/3 of drops
        has_expert = i % 3 == 0  # ~1/3 of drops

        # ML sees everything plus the catch-all 'fish' bucket (binary-model
        # style: the bucket count dwarfs the per-species counts).
        for sp in species:
            annotations.append(
                row(
                    drop_id,
                    sp,
                    fake_count(drop_id, sp),
                    "ml",
                    MODEL_NAME,
                    round(rng.uniform(0.4, 0.9), 2),
                )
            )
        annotations.append(
            row(
                drop_id,
                "fish",
                rng.randint(3, 12),
                "ml",
                MODEL_NAME,
                round(rng.uniform(0.4, 0.9), 2),
            )
        )

        if has_citsci:
            # Volunteers see a subset of what ML saw; agreement is a percent.
            for sp in species[: rng.randint(1, len(species))]:
                annotations.append(
                    row(
                        drop_id,
                        sp,
                        fake_count(drop_id, sp),
                        "citsci",
                        str(rng.randint(9_000_000, 9_999_999)),
                        round(rng.uniform(50, 100), 1),
                    )
                )

        if has_expert:
            # Experts confirm most species but sometimes disagree on the count
            # (+1) and occasionally add one nobody else reported.
            for sp in species:
                annotations.append(
                    row(
                        drop_id,
                        sp,
                        fake_count(drop_id, sp) + (1 if rng.random() < 0.3 else 0),
                        "expert",
                        str(rng.randint(1_000_000, 1_999_999)),
                        None,
                    )
                )
            if rng.random() < 0.5:
                extra = rng.choice([s for s, _ in SPECIES_POOL if s not in species])
                annotations.append(
                    row(
                        drop_id,
                        extra,
                        1,
                        "expert",
                        str(rng.randint(1_000_000, 1_999_999)),
                        None,
                    )
                )

    # Idempotent: clear each (drop, source) pair before inserting.
    for drop_id in drops:
        for source in ("ml", "citsci", "expert"):
            ann_db.clear_annotations(drop_id, source)
    ann_db.add_annotations(annotations)
    print(f"Inserted {len(annotations)} annotations across {len(drops)} drops.")

    # Writes counts to deployments AND advances section statuses to *_complete.
    db.sync_annotation_counts(drops)


if __name__ == "__main__":
    main()
