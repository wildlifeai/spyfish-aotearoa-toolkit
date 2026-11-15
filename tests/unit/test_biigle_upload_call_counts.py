"""A survey upload asks S3 and BIIGLE once for what every drop shares.

Before 2026-11-11 each drop of a pooled survey upload listed the whole survey
tree on S3, fetched the project's volume list, and (on a volume whose image IDs
were not already known) fetched /filenames twice. The account had already been
locked once by a per-image loop (2026-10-16), so the counts are pinned here.
"""

import json
from collections import Counter

import pandas as pd
import pytest

from spyfish.biigle import upload_frames
from spyfish.config.wrapper import config

SURVEY = "KSF_20240124_BUV"
N_DROPS = 40
S3_ROOT = "process_files/deployment_data"


def _drop(i: int) -> str:
    return f"{SURVEY}_KSF_{i:03d}_01"


class FakeBiigle:
    """Stands in for BiigleHandler; counts every method that is a request."""

    def __init__(self, volumes=None):
        self.calls: Counter = Counter()
        self.vols = list(volumes or [])
        self.images: dict = {}
        self._next = 1000

    def _register(self, files):
        out = []
        for f in files:
            if f not in self.images:
                self._next += 1
                self.images[f] = self._next
                out.append({"id": self._next, "filename": f})
        return out

    def build_s3_url(self, s3_path):
        return f"disk-1://{s3_path.rstrip('/')}/"

    def get_volumes(self, project_id=None):
        self.calls["get_volumes"] += 1
        return list(self.vols)

    def create_volume_from_s3_files(
        self, volume_name, s3_url, files, project_id, media_type
    ):
        self.calls["create_volume"] += 1
        self._register(files)
        vol_id = 70 + len(self.vols)
        self.vols.append(
            {"id": vol_id, "name": volume_name, "url": s3_url, "created_at": "2026"}
        )
        return {"id": vol_id}

    def add_files_to_volume(self, volume_id, files):
        self.calls["add_files"] += 1
        return self._register(files)

    def get_volume_images(self, volume_id):
        self.calls["get_volume_images"] += 1
        return [{"id": i, "filename": f} for f, i in self.images.items()]

    def upload_image_annotations(self, volume_id, annotations):
        self.calls["upload_annotations"] += 1
        return {"uploaded_count": len(annotations), "details": []}

    def attach_image_labels(self, pairs):
        self.calls["attach_image_labels"] += 1
        return len(pairs)


class FakeS3:
    def __init__(self):
        self.list_prefixes: list = []
        self.uploaded: list = []

    def get_file_paths_set_from_s3(self, prefix="", suffixes=()):
        self.list_prefixes.append(prefix)
        return set()

    def upload_file_to_s3(self, path, key, content_type=None):
        self.uploaded.append(key)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Local frames + COCO for every drop; S3, BIIGLE and the DB are fakes."""
    cfg = type(config)
    monkeypatch.setattr(cfg, "deployment_data_dir", property(lambda self: tmp_path))
    monkeypatch.setattr(cfg, "s3_deployment_data_dir", property(lambda self: S3_ROOT))
    monkeypatch.setattr(
        cfg,
        "get_coco_annotations_path",
        lambda self, d, target="": tmp_path / SURVEY / d / "coco.json",
    )
    monkeypatch.setattr(
        cfg, "selection_reason_label_ids", property(lambda self: {"spot_check": 901})
    )
    monkeypatch.setattr(
        cfg, "selection_reason_include_species", property(lambda self: False)
    )

    frames = {}
    for i in range(N_DROPS):
        drop = _drop(i)
        frames_dir = tmp_path / SURVEY / drop / "frames"
        frames_dir.mkdir(parents=True)
        names = [f"{drop}__frame_{t:04d}.000s.jpg" for t in (10, 20)]
        for name in names:
            (frames_dir / name).write_bytes(b"jpg")
        coco = {
            "images": [
                {"id": k, "file_name": n, "width": 1920, "height": 1080}
                for k, n in enumerate(names)
            ],
            "categories": [{"id": 1, "name": "fish"}],
            "annotations": [
                {"id": k, "image_id": k, "category_id": 1, "bbox": [1, 2, 30, 40]}
                for k in range(len(names))
            ],
        }
        (tmp_path / SURVEY / drop / "coco.json").write_text(json.dumps(coco))
        frames[drop] = pd.DataFrame(
            {
                "FramePath": [str(frames_dir / n) for n in names],
                config.selection_reason_column: ["Blind (False Negative Check)"] * 2,
            }
        )

    s3 = FakeS3()
    monkeypatch.setattr(upload_frames, "S3Handler", lambda: s3)
    monkeypatch.setattr(upload_frames, "DatabaseManager", lambda: _NoDb())
    monkeypatch.setattr(upload_frames, "save_upload_record", lambda *a, **k: None)
    monkeypatch.setattr(upload_frames, "_PROCESS_RUN", upload_frames.UploadRun())

    def use(biigle):
        monkeypatch.setattr(upload_frames, "BiigleHandler", lambda: biigle)
        return biigle

    return frames, s3, use


class _NoDb:
    def update_biigle_volume_id(self, drop_id, volume_id):
        pass


def test_survey_upload_into_a_new_volume(env):
    frames, s3, use = env
    biigle = use(FakeBiigle())

    for drop, df in frames.items():
        upload_frames.upload_frames_to_expert_survey_volume(drop, df)

    # the list that finds no volume, then one fresh list after creating it
    assert biigle.calls["get_volumes"] == 2
    assert biigle.calls["create_volume"] == 1
    assert biigle.calls["add_files"] == N_DROPS - 1
    # only the creating drop has no IDs in hand, and it asks once for both the
    # boxes and the image labels
    assert biigle.calls["get_volume_images"] == 1
    assert biigle.calls["upload_annotations"] == N_DROPS
    assert biigle.calls["attach_image_labels"] == N_DROPS

    # one S3 listing per drop, of that drop's frames folder and nothing wider
    assert s3.list_prefixes == [
        f"{S3_ROOT}/{SURVEY}/{_drop(i)}/frames/" for i in range(N_DROPS)
    ]
    assert len(s3.uploaded) == 2 * N_DROPS


def test_survey_upload_into_an_existing_volume(env):
    frames, s3, use = env
    existing = {
        "id": 55,
        "name": config.expert_survey_volume_name_template.format(survey_id=SURVEY),
        "url": f"disk-1://{S3_ROOT}/{SURVEY}/",
        "created_at": "2026",
    }
    biigle = use(FakeBiigle(volumes=[existing]))

    for drop, df in frames.items():
        upload_frames.upload_frames_to_expert_survey_volume(drop, df)

    assert biigle.calls["get_volumes"] == 1
    assert biigle.calls["create_volume"] == 0
    assert biigle.calls["add_files"] == N_DROPS
    assert biigle.calls["get_volume_images"] == 0


def test_an_explicit_run_is_used_instead_of_the_process_one(env):
    frames, s3, use = env
    biigle = use(FakeBiigle())
    drops = list(frames)[:3]

    for drop in drops:
        upload_frames.upload_frames_to_expert_survey_volume(
            drop, frames[drop], run=upload_frames.UploadRun()
        )

    # a new run per call knows nothing: one list each, plus the one that
    # follows the creation (dropped with its run, so never used)
    assert biigle.calls["get_volumes"] == len(drops)
    assert biigle.calls["create_volume"] == 1


def test_find_or_create_without_a_run_fetches_the_list_itself(env):
    frames, s3, use = env
    biigle = use(FakeBiigle())

    for _ in range(2):
        upload_frames.find_or_create_volume_and_add_frames(
            "KSF_20240124_BUV ML", f"{S3_ROOT}/{SURVEY}/", ["a.jpg"]
        )

    assert biigle.calls["get_volumes"] == 2
    assert biigle.calls["create_volume"] == 1
    assert biigle.calls["add_files"] == 1


def test_per_drop_volume_fetches_filenames_once(env, monkeypatch):
    frames, s3, use = env
    biigle = use(FakeBiigle())
    drop = _drop(0)
    monkeypatch.setattr(
        type(config),
        "get_frames_s3_prefix",
        lambda self, d: f"{S3_ROOT}/{SURVEY}/{d}/frames/",
    )
    monkeypatch.setattr(
        upload_frames,
        "create_biigle_volume",
        lambda d, prefix, files, project_id=None: (
            biigle._register(files) and {"id": 70}
        ),
    )

    upload_frames.upload_frames_to_biigle(drop, frames[drop])

    assert biigle.calls["get_volume_images"] == 1  # was 2: boxes, then labels
    assert biigle.calls["upload_annotations"] == 1
    assert biigle.calls["attach_image_labels"] == 1
    assert s3.list_prefixes == [f"{S3_ROOT}/{SURVEY}/{drop}/frames/"]


def test_a_name_the_volume_does_not_have_is_not_fetched_for_twice(env):
    """The whole-volume map is final: a missing name cannot trigger a re-fetch."""
    frames, s3, use = env
    biigle = use(FakeBiigle())
    biigle.images = {"other.jpg": 1}
    drop = _drop(0)
    coco = json.loads(config.get_coco_annotations_path(drop).read_text())

    ids = upload_frames._image_ids_for_drop(9, coco, frames[drop])
    with pytest.raises(RuntimeError, match="none of the"):
        upload_frames.upload_coco_annotations_to_biigle(
            9, coco, filename_to_biigle_id=ids
        )
    upload_frames.attach_selection_reason_labels(
        9, frames[drop], filename_to_biigle_id=ids
    )

    assert biigle.calls["get_volume_images"] == 1


def test_listing_prefix_is_the_deepest_shared_folder():
    f = upload_frames._listing_prefix
    assert f(["a/b/frames/x1.jpg", "a/b/frames/x2.jpg"]) == "a/b/frames/"
    assert f(["a/b/frames/x1.jpg"]) == "a/b/frames/"
    assert f(["a/b1/frames/x.jpg", "a/b2/frames/x.jpg"]) == "a/"
    assert f(["a/b/x.jpg", "a/b/frames/x.jpg"]) == "a/b/"
