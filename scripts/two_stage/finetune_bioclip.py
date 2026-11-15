"""Fine-tune the tail of BioCLIP 2 on the crops (CLI for spyfish.ml.classifier.finetune).

    python scripts/two_stage/finetune_bioclip.py                       # recipe from training.classifier.finetune
    python scripts/two_stage/finetune_bioclip.py --epochs 15 --unfreeze 6
    python scripts/two_stage/finetune_bioclip.py --all-train --out final   # the checkpoint that ships

Writes best.pt / final.pt, classes.json (which species became classes and why),
finetune_report.md and finetune_predictions.csv under <two_stage_dir>/<out>/.
Copy the checkpoint into paths.sub_dirs.classifier with a model-name stamp to
put it into the pipeline (ml_inference.classifier).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from spyfish.ml.classifier.finetune import main  # noqa: E402

if __name__ == "__main__":
    main()
