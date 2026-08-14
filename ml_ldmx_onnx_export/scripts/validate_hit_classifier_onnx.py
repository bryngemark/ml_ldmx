#!/usr/bin/env python3
"""Compare PyTorch and ONNX Runtime logits for an exported baseline run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from ml_ldmx.export import CANONICAL_FEATURES, ECalTpadTransformerONNXWrapper
from ml_ldmx.models import ECalTpadTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, default=None)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument(
        "--input-npy",
        type=Path,
        action="append",
        default=[],
        help="Optional raw [num_tokens,8] arrays; may be repeated.",
    )
    parser.add_argument("--token-counts", type=int, nargs="+", default=[3, 17, 64, 257])
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    return parser.parse_args()


def synthetic_tokens(count: int, seed: int) -> np.ndarray:
    if count < 2:
        raise ValueError("Each validation event needs at least two tokens.")
    rng = np.random.default_rng(seed)
    x = np.zeros((count, 8), dtype=np.float32)
    n_ecal = max(1, count - max(1, count // 8))
    x[:n_ecal, 0] = 1.0
    x[:n_ecal, 2:5] = rng.normal(0.0, 100.0, size=(n_ecal, 3))
    x[:n_ecal, 5] = rng.lognormal(mean=0.0, sigma=3.0, size=n_ecal)
    x[n_ecal:, 1] = 1.0
    x[n_ecal:, 6] = rng.normal(0.0, 30.0, size=count - n_ecal)
    x[n_ecal:, 7] = rng.lognormal(mean=1.0, sigma=2.0, size=count - n_ecal)
    return x


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        checkpoint_path = run_dir / "checkpoints" / args.checkpoint
    onnx_path = (args.onnx or (run_dir / "export/model.onnx")).resolve()

    with (run_dir / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args") or {}
    model_kwargs = checkpoint["model_kwargs"]
    model = ECalTpadTransformer(**model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    wrapper = ECalTpadTransformerONNXWrapper(
        model,
        ecal_energy_transform=checkpoint_args.get(
            "ecal_energy_transform", config.get("ecal_energy_transform", "raw")
        ),
        tpad_pe_transform=checkpoint_args.get(
            "tpad_pe_transform", config.get("tpad_pe_transform", "raw")
        ),
        feature_norm=checkpoint.get("feature_norm"),
    ).eval()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    arrays = [np.load(path).astype(np.float32, copy=False) for path in args.input_npy]
    arrays.extend(synthetic_tokens(count, seed=1000 + i) for i, count in enumerate(args.token_counts))
    if not arrays:
        raise ValueError("No validation inputs were supplied.")

    worst_abs = 0.0
    worst_rel = 0.0
    for index, array in enumerate(arrays):
        if array.ndim != 2 or array.shape[1] != len(CANONICAL_FEATURES):
            raise ValueError(f"Input {index} has invalid shape {array.shape}; expected [T,8].")
        with torch.no_grad():
            torch_logits = wrapper(torch.from_numpy(array)).cpu().numpy()
        ort_logits = session.run([output_name], {input_name: array})[0]
        if torch_logits.shape != ort_logits.shape:
            raise AssertionError(
                f"Shape mismatch for input {index}: torch={torch_logits.shape}, ONNX={ort_logits.shape}"
            )
        abs_error = np.abs(torch_logits - ort_logits)
        rel_error = abs_error / np.maximum(np.abs(torch_logits), 1e-12)
        worst_abs = max(worst_abs, float(abs_error.max(initial=0.0)))
        worst_rel = max(worst_rel, float(rel_error.max(initial=0.0)))
        np.testing.assert_allclose(ort_logits, torch_logits, rtol=args.rtol, atol=args.atol)
        print(
            f"PASS input={index} tokens={array.shape[0]} classes={torch_logits.shape[1]} "
            f"max_abs={abs_error.max():.3e} max_rel={rel_error.max():.3e}"
        )

    print(f"All {len(arrays)} inputs passed; worst_abs={worst_abs:.3e}, worst_rel={worst_rel:.3e}")


if __name__ == "__main__":
    main()
