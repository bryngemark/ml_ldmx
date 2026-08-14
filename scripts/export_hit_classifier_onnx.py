#!/usr/bin/env python3
"""Export a saved ECalTpadTransformer baseline run to ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from ml_ldmx.export import CANONICAL_FEATURES, ECalTpadTransformerONNXWrapper
from ml_ldmx.models import ECalTpadTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a saved ml_ldmx ECalTpadTransformer run to ONNX."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        default="best.pt",
        help="Checkpoint filename below <run-dir>/checkpoints, or an explicit path.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--metadata-output", type=Path, default=None)
    parser.add_argument("--example-tokens", type=int, default=32)
    parser.add_argument("--opset-version", type=int, default=18)
    parser.add_argument(
        "--legacy-exporter",
        action="store_true",
        help="Use the TorchScript exporter if the modern dynamo exporter fails.",
    )
    parser.add_argument("--report", action="store_true")
    return parser.parse_args()


def resolve_checkpoint(run_dir: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_file():
        return candidate
    candidate = run_dir / "checkpoints" / value
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Could not find checkpoint: {value}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def checkpoint_arg(checkpoint: dict[str, Any], config: dict[str, Any], name: str, default: Any) -> Any:
    checkpoint_args = checkpoint.get("args") or {}
    if name in checkpoint_args:
        return checkpoint_args[name]
    return config.get(name, default)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing run config: {config_path}")

    checkpoint_path = resolve_checkpoint(run_dir, args.checkpoint).resolve()
    config = load_json(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected checkpoint to contain a dictionary.")

    model_name = checkpoint_arg(checkpoint, config, "model", None)
    if model_name != "ECalTpadTransformer":
        raise ValueError(
            "This exporter currently targets ECalTpadTransformer; "
            f"the saved run reports {model_name!r}."
        )

    model_kwargs = checkpoint.get("model_kwargs")
    if not isinstance(model_kwargs, dict):
        raise KeyError("Checkpoint does not contain model_kwargs.")
    if int(model_kwargs.get("in_dim", -1)) != len(CANONICAL_FEATURES):
        raise ValueError(
            f"Expected an 8-feature context model, got model_kwargs={model_kwargs}."
        )

    model = ECalTpadTransformer(**model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    ecal_transform = str(
        checkpoint_arg(checkpoint, config, "ecal_energy_transform", "raw")
    )
    tpad_transform = str(
        checkpoint_arg(checkpoint, config, "tpad_pe_transform", "raw")
    )
    wrapper = ECalTpadTransformerONNXWrapper(
        model,
        ecal_energy_transform=ecal_transform,
        tpad_pe_transform=tpad_transform,
        feature_norm=checkpoint.get("feature_norm"),
    ).eval()

    output_path = (args.output or (run_dir / "export" / "model.onnx")).resolve()
    metadata_path = (
        args.metadata_output or output_path.with_name("model_metadata.json")
    ).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    if args.example_tokens <= 1:
        raise ValueError("--example-tokens must be greater than one.")
    example = torch.zeros((args.example_tokens, len(CANONICAL_FEATURES)), dtype=torch.float32)
    split = max(1, args.example_tokens - 2)
    example[:split, 0] = 1.0
    example[:split, 2:5] = torch.randn((split, 3)) * 50.0
    example[:split, 5] = torch.rand(split) * 1000.0
    example[split:, 1] = 1.0
    example[split:, 6] = torch.randn(args.example_tokens - split) * 20.0
    example[split:, 7] = torch.rand(args.example_tokens - split) * 100.0

    if args.legacy_exporter:
        torch.onnx.export(
            wrapper,
            (example,),
            str(output_path),
            input_names=["tokens"],
            output_names=["logits"],
            dynamic_axes={
                "tokens": {0: "num_tokens"},
                "logits": {0: "num_tokens"},
            },
            opset_version=args.opset_version,
            do_constant_folding=True,
            dynamo=False,
        )
        exporter = "torchscript"
    else:
        onnx_program = torch.onnx.export(
            wrapper,
            (example,),
            input_names=["tokens"],
            output_names=["logits"],
            dynamic_shapes=({0: "num_tokens"},),
            opset_version=args.opset_version,
            dynamo=True,
            report=args.report,
            artifacts_dir=str(output_path.parent),
            verify=False,
        )
        onnx_program.save(str(output_path), external_data=False)
        exporter = "dynamo"

    valid_labels = [int(value) for value in checkpoint.get("valid_labels", ())]
    out_dim = int(model_kwargs["out_dim"])
    if valid_labels and len(valid_labels) != out_dim:
        raise ValueError("valid_labels length does not match model output dimension.")

    metadata = {
        "format_version": 1,
        "model_family": "ECalTpadTransformer",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "exporter": exporter,
        "opset_version": args.opset_version,
        "input": {
            "name": "tokens",
            "dtype": "float32",
            "shape": ["num_tokens", len(CANONICAL_FEATURES)],
            "features": list(CANONICAL_FEATURES),
            "raw_framework_values": True,
        },
        "preprocessing": {
            "ecal_energy_transform": ecal_transform,
            "tpad_pe_transform": tpad_transform,
            "feature_norm": checkpoint.get("feature_norm"),
            "embedded_in_onnx": True,
        },
        "output": {
            "name": "logits",
            "dtype": "float32",
            "shape": ["num_tokens", out_dim],
            "scope": "all_tokens",
            "consume_rows_where": "input.is_ecal == 1",
            "class_index_to_classification": "argmax(logits) + 1",
            "confidence": "max(softmax(logits))",
            "valid_training_labels": valid_labels,
        },
        "model_kwargs": model_kwargs,
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Wrote ONNX model: {output_path}")
    print(f"Wrote metadata:   {metadata_path}")


if __name__ == "__main__":
    main()
