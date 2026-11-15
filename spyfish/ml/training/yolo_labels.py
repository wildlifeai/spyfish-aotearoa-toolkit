"""Reading and rewriting the files of a YOLO dataset: label ``.txt`` and ``data.yaml``.

One place for the small parsers every training step needs, so "what counts as a
label row" and "what order are the class names in" have one answer.

A label file holds one box per line: ``class_id cx cy w h`` (five fields). A
``data.yaml`` names the classes under ``names``, either as a list (what
``generate_data_yaml`` writes) or as an ``{id: name}`` mapping (what Ultralytics
and the per-model test views write).

No config access and no Ultralytics import: everything here is a pure function
of the paths it is given.
"""

from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional

import yaml

# The split directories of an assembled dataset, in the order they are reported.
SPLITS = ("train", "val", "test")

# Fields in a detection label row: class id + normalised cx, cy, w, h.
BOX_FIELDS = 5

DATA_YAML_FILENAME = "data.yaml"


# ---------------------------------------------------------------------------
# Label files
# ---------------------------------------------------------------------------


def iter_class_ids(label_path: Path, boxes_only: bool = False) -> Iterator[int]:
    """Yield the class id of every row of a YOLO label file, in file order.

    Blank lines and rows whose first field is not an integer are skipped. An id
    is yielded once per row, so ``set(iter_class_ids(p))`` is the classes in the
    frame and ``Counter(iter_class_ids(p))`` its boxes per class.

    ``boxes_only=True`` also skips rows that do not have exactly five fields,
    for callers that must not count a truncated or segmentation row as a box.
    """
    for line in Path(label_path).read_text().splitlines():
        parts = line.split()
        if not parts or (boxes_only and len(parts) != BOX_FIELDS):
            continue
        try:
            yield int(parts[0])
        except ValueError:
            continue


def read_label_rows(label_path: Path, boxes_only: bool = True) -> List[List[str]]:
    """The rows of a YOLO label file, each as its whitespace-split fields.

    Fields stay strings so a row can be written back byte for byte. By default
    only well-formed box rows (five fields) are returned; ``boxes_only=False``
    returns every non-blank row, whatever its length.
    """
    rows = []
    for line in Path(label_path).read_text().splitlines():
        parts = line.split()
        if not parts or (boxes_only and len(parts) != BOX_FIELDS):
            continue
        rows.append(parts)
    return rows


def remap_label_file(
    label_path: Path,
    id_map: Mapping[int, Optional[int]],
    dst: Optional[Path] = None,
    missing: str = "drop",
    trailing_newline: bool = False,
) -> Counter:
    """Rewrite a YOLO label file through ``id_map`` (old class id -> new).

    Only five-field rows are carried over; anything else is left out of the
    output. A row whose id maps to ``None`` is dropped. A row whose id is not
    in ``id_map`` at all follows ``missing``:

    * ``"drop"``: the row is dropped (flattening per-drop labels into the
      unified id space, where an unknown id has no meaning);
    * ``"keep"``: the row keeps its id (``id_map`` lists only the ids that move);
    * ``"raise"``: ``KeyError`` (``id_map`` is meant to cover every id in use, so
      a gap is a corrupt dataset rather than something to paper over).

    A first field that is not an integer raises ``ValueError`` in every mode.

    The result is written to ``dst``, or back to ``label_path`` when ``dst`` is
    None. It is a plain write, not a replace: assembled datasets symlink their
    labels to the staged tree, and writing through the link is what the callers
    rely on. Rows are joined with newlines, with a final newline only when
    ``trailing_newline`` is set and there is at least one row.

    Returns the boxes read per SOURCE class id (every five-field row, kept or
    not), so a caller can report how many boxes each mapping decision touched.
    """
    if missing not in ("drop", "keep", "raise"):
        raise ValueError(f"missing must be 'drop', 'keep' or 'raise', not {missing!r}")
    seen: Counter = Counter()
    out: List[str] = []
    for parts in read_label_rows(label_path):
        old_id = int(parts[0])
        seen[old_id] += 1
        if old_id in id_map:
            new_id = id_map[old_id]
        elif missing == "keep":
            new_id = old_id
        elif missing == "raise":
            raise KeyError(old_id)
        else:
            new_id = None
        if new_id is None:
            continue
        out.append(f"{new_id} " + " ".join(parts[1:]))
    text = "\n".join(out)
    if trailing_newline and out:
        text += "\n"
    Path(dst if dst is not None else label_path).write_text(text)
    return seen


# ---------------------------------------------------------------------------
# data.yaml class names
# ---------------------------------------------------------------------------


def read_class_names_by_id(data_yaml: Path, required: bool = False) -> Dict[int, str]:
    """``{class id: name}`` from a ``data.yaml``, whether ``names`` is a list or
    a mapping.

    A ``data.yaml`` with no ``names`` key gives ``{}``, or ``KeyError`` when
    ``required`` is set.
    """
    data = yaml.safe_load(Path(data_yaml).read_text())
    names = data["names"] if required else data.get("names", [])
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    return dict(enumerate(names))


def read_class_names(data_yaml: Path, required: bool = False) -> List[str]:
    """Class names of a ``data.yaml`` in class-id order, whether ``names`` is a
    list or a mapping. Index in the returned list = class id, which holds for
    every dataset YOLO accepts (ids are 0..nc-1 with no gaps)."""
    by_id = read_class_names_by_id(data_yaml, required=required)
    return [by_id[k] for k in sorted(by_id)]


def write_class_names(data_yaml: Path, names: List[str]) -> None:
    """Replace ``names`` (as a list) and ``nc`` in an existing ``data.yaml``,
    keeping every other key and the key order as they are."""
    data_yaml = Path(data_yaml)
    data = yaml.safe_load(data_yaml.read_text())
    data["names"] = list(names)
    data["nc"] = len(names)
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False))
