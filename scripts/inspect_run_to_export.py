from pathlib import Path
import json
import torch
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument(
    "--run-dir",
    type=Path,
    required=True,
    help="Path to the copied training run directory",
)
args = parser.parse_args()

run = args.run_dir.expanduser().resolve()

if not run.is_dir():
    parser.error(f"Run directory does not exist: {run}")
    
with (run / "config.json").open() as f:
    config = json.load(f)

checkpoint = torch.load(
    run / "checkpoints" / "best.pt",
    map_location="cpu",
    weights_only=False,
)

print("Config model:", config.get("model"))
print("Checkpoint keys:", sorted(checkpoint.keys()))
print("Model kwargs:", checkpoint.get("model_kwargs"))
print("Feature normalization:", checkpoint.get("feature_norm"))
print("Valid labels:", checkpoint.get("valid_labels"))
