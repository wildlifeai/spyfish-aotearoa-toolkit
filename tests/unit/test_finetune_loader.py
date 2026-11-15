"""The BioCLIP fine-tune's DataLoaders: worker count from config, a dataset
that pickles into worker processes, and augmentation that repeats run to run.

No model and no open_clip: only the dataset, the loader settings and the
training transform are exercised.
"""

import pickle

import numpy as np
import pytest
from PIL import Image

from spyfish.config.wrapper import config
from spyfish.ml.classifier.finetune import (
    CropDataset,
    build_train_transform,
    loader_kwargs,
    parse_args,
)

MEAN, STD = (0.48, 0.46, 0.41), (0.27, 0.26, 0.28)


@pytest.fixture
def crop_files(tmp_path):
    rng = np.random.default_rng(3)
    paths = []
    for i in range(12):
        path = tmp_path / f"crop_{i}.jpg"
        Image.fromarray(
            rng.integers(0, 256, (40 + i, 60 + 2 * i, 3)).astype(np.uint8)
        ).save(path)
        paths.append(path)
    return paths


def _train_dataset(paths):
    tf = build_train_transform(config.classifier_finetune.augmentation, MEAN, STD)
    return CropDataset(paths, np.arange(len(paths)) % 3, tf)


def test_workers_is_a_required_non_negative_config_key(monkeypatch):
    recipe = config.classifier_finetune
    assert isinstance(recipe.workers, int) and recipe.workers >= 0
    assert parse_args([]).workers == recipe.workers
    assert parse_args(["--workers", "0"]).workers == 0

    training = dict(config._classifier_training)
    finetune = dict(training["finetune"])

    def with_finetune(ft):
        monkeypatch.setattr(
            type(config),
            "_classifier_training",
            property(lambda s: {**training, "finetune": ft}),
        )

    with_finetune({**finetune, "workers": -1})
    with pytest.raises(ValueError, match="workers"):
        config.classifier_finetune
    finetune.pop("workers")
    with_finetune(finetune)
    with pytest.raises(KeyError):
        config.classifier_finetune


def test_no_workers_adds_nothing_to_the_loader():
    """`workers: 0` must stay the run it always was: no generator, so the
    training process draws the same random numbers as before."""
    assert loader_kwargs(0, "cpu", 42) == {"num_workers": 0, "pin_memory": False}
    assert loader_kwargs(0, "cuda", 42) == {"num_workers": 0, "pin_memory": True}


def test_workers_are_persistent_seeded_and_pinned_only_on_cuda():
    import torch

    kwargs = loader_kwargs(6, "cuda", 42)
    assert kwargs["num_workers"] == 6
    assert kwargs["persistent_workers"] is True
    assert kwargs["pin_memory"] is True
    assert loader_kwargs(6, "mps", 42)["pin_memory"] is False
    # Seeded from the given seed: two loaders draw the same base seed.
    draw = (
        lambda g: torch.empty((), dtype=torch.int64).random_(generator=g).item()
    )  # noqa: E731
    assert draw(kwargs["generator"]) == draw(loader_kwargs(2, "cpu", 42)["generator"])
    assert draw(loader_kwargs(2, "cpu", 42)["generator"]) != draw(
        loader_kwargs(2, "cpu", 43)["generator"]
    )


def test_loader_is_built_with_the_given_worker_count(crop_files):
    from torch.utils.data import DataLoader

    loader = DataLoader(
        _train_dataset(crop_files), batch_size=4, **loader_kwargs(3, "cpu", 42)
    )
    assert loader.num_workers == 3 and loader.persistent_workers is True


def test_dataset_pickles_for_spawned_workers(crop_files):
    """macOS starts workers with spawn: the dataset and its transform cross
    into them by pickle, so neither may be a local class, lambda or open file."""
    ds = _train_dataset(crop_files)
    clone = pickle.loads(pickle.dumps(ds))
    assert len(clone) == len(ds) == 12
    x, label = clone[5]
    assert tuple(x.shape) == (3, 224, 224) and label == 2 and isinstance(label, int)


def test_dataset_refuses_non_integer_class_ids(crop_files):
    with pytest.raises(TypeError, match="integers"):
        CropDataset(crop_files, np.array([0.0, np.nan] * 6), None)


def _epochs(paths, workers, seed, n_epochs=2):
    """Every batch of `n_epochs` passes over the crops, as (pixels, labels)."""
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    loader = DataLoader(
        _train_dataset(paths), batch_size=4, **loader_kwargs(workers, "cpu", seed)
    )
    xs, ys = [], []
    for _ in range(n_epochs):
        for x, y in loader:
            xs.append(x)
            ys.append(y)
    # torch on macOS waits 5 s per worker for a clean exit; the batches are in
    # hand, so kill the workers and keep the suite fast.
    for worker in getattr(getattr(loader, "_iterator", None), "_workers", []):
        worker.kill()
    return torch.cat(xs), torch.cat(ys)


# Killing the workers (see _epochs) makes the loader's own __del__ complain.
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_augmentation_repeats_run_to_run_with_and_without_workers(crop_files):
    """Same seed, same worker count: the same augmented pixels, across epochs
    too. A different seed gives different ones, so the check is not vacuous."""
    import torch

    for workers in (0, 2):
        x1, y1 = _epochs(crop_files, workers, seed=42)
        x2, y2 = _epochs(crop_files, workers, seed=42)
        assert torch.equal(x1, x2) and torch.equal(y1, y2), workers
        assert y1.dtype == torch.int64 and y1.tolist() == [0, 1, 2] * 4 * 2
        # second epoch is augmented afresh, not a replay of the first
        assert not torch.equal(x1[:12], x1[12:])
    # (workers=0 here: each worker start costs seconds under macOS spawn)
    x3, _ = _epochs(crop_files, 0, seed=43)
    assert not torch.equal(_epochs(crop_files, 0, seed=42)[0], x3)
