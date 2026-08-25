"""Create the "Pick" selection-reason label hierarchy in the Biigle workflow tree.

One-off setup for the selection-reason image labels. Creates a parent label
"Pick" with one child per selection bucket, then prints the IDs to paste into
`biigle.selection_reason_labels` in config.yaml.

The labels go in the WORKFLOW tree (config.biigle_workflow_label_tree_id, 3375)
rather than the species tree, for two reasons: the species tree is what experts
pick from and must stay clean, and the expert sync already drops workflow-tree
labels that are not in `workflow_tree_keep_labels`, so these can never be
mistaken for a sighting.

Two Biigle constraints make this worth running deliberately rather than as part
of the pipeline: a label's parent_id cannot later be set back to null, and a
label with children cannot be deleted. Creating the hierarchy is easy; undoing
it is not. Hence --apply is required and the default is a dry run.

Renaming a label afterwards is safe and needs no code change: config stores
only the IDs.

Usage (from project root):
    python scripts/create_biigle_pick_labels.py             # dry run, shows plan
    python scripts/create_biigle_pick_labels.py --apply     # actually create
    python scripts/create_biigle_pick_labels.py --apply --tree-id 1234
"""

import argparse
import logging

from spyfish.biigle.biigle_handler import BiigleHandler
from spyfish.config.wrapper import config

PARENT_NAME = "Pick"
PARENT_COLOR = "6c757d"  # muted grey: workflow metadata, not a sighting

# Canonical key (spyfish/biigle/selection_reason.py) → (child label name, colour).
# Colours group by kind: counting buckets blue, uncertainty amber, coverage green.
CHILDREN: dict = {
    "maxn_peak": ("MaxN peak", "0d6efd"),
    "ml_peak": ("ML peak", "6610f2"),
    "uncertain_id": ("Uncertain ID", "fd7e14"),
    "fish_variety": ("Fish variety", "20c997"),
    "spot_check": ("Spot check", "198754"),
    "video_start": ("Video start", "adb5bd"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually create the labels. Without this the script only prints "
        "the plan and what already exists.",
    )
    parser.add_argument(
        "--tree-id",
        type=int,
        default=None,
        help="Label tree to create in. Defaults to the workflow tree from config.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    tree_id = args.tree_id or config.biigle_workflow_label_tree_id
    handler = BiigleHandler()

    existing = handler.get_label_tree_labels(tree_id)
    by_name = {str(lbl.get("name")): lbl for lbl in existing}
    logging.info(f"Label tree {tree_id} currently holds {len(existing)} label(s).")

    # Reuse an existing parent so a re-run tops up missing children instead of
    # creating a second "Pick" tree alongside the first.
    parent = by_name.get(PARENT_NAME)
    if parent:
        logging.info(f"Parent {PARENT_NAME!r} already exists (id={parent['id']}).")
    elif args.apply:
        parent = handler.create_label(tree_id, PARENT_NAME, PARENT_COLOR)
    else:
        logging.info(f"WOULD CREATE parent {PARENT_NAME!r} in tree {tree_id}")

    parent_id = parent["id"] if parent else None
    resolved: dict = {}

    for key, (name, color) in CHILDREN.items():
        found = by_name.get(name)
        if found:
            logging.info(f"  {key:<13} {name!r} already exists (id={found['id']})")
            resolved[key] = found["id"]
            continue
        if not args.apply:
            logging.info(f"  {key:<13} WOULD CREATE {name!r} under {PARENT_NAME!r}")
            continue
        created = handler.create_label(tree_id, name, color, parent_id=parent_id)
        resolved[key] = created["id"]

    if not args.apply:
        logging.info("\nDry run, nothing was created. Re-run with --apply.")
        return

    logging.info("\nPaste into config.yaml under biigle.selection_reason_labels:\n")
    for key, (name, _) in CHILDREN.items():
        label_id = resolved.get(key)
        value = label_id if label_id is not None else "null"
        logging.info(f"    {key}: {value}".ljust(30) + f"# Pick > {name}")


if __name__ == "__main__":
    main()
