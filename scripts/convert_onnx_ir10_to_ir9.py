#!/usr/bin/env python3
"""Convert a compatible ONNX IR-10 model to IR 9.

The converter is conservative:

1. It refuses models that use IR-10-only computational/schema features.
2. It permits IR-10 metadata_props on GraphProto, NodeProto, and
   ValueInfoProto.
3. It writes every stripped metadata entry to a JSON sidecar.
4. It clears only those metadata_props fields.
5. It sets ModelProto.ir_version to 9.
6. It runs the ONNX checker.
7. Unless disabled, it compares ONNX Runtime outputs from the original and
   converted models for several variable token counts.

It does not change the model's operator-set imports.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import onnx
from google.protobuf.message import Message


IR9 = 9

IR10_ONLY_TENSOR_TYPES = {
    21: "UINT4",
    22: "INT4",
}

# These message types gained metadata_props in IR 10. The metadata is
# annotation-only and may be stripped for an IR-9 deployment copy.
STRIPPABLE_IR10_METADATA_TYPES = {
    "GraphProto",
    "NodeProto",
    "ValueInfoProto",
}

# Metadata on these additional message types is also an IR-10 feature, but this
# converter does not strip it automatically because it has not been observed in
# this export and deserves separate review.
UNSUPPORTED_IR10_METADATA_TYPES = {
    "TensorProto",
    "FunctionProto",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an IR-9 deployment copy of a compatible ONNX IR-10 model."
        )
    )
    parser.add_argument("input", type=Path, help="Input ONNX model (IR 10).")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output model path. Default: <input stem>.ir9.onnx",
    )
    parser.add_argument(
        "--metadata-sidecar",
        type=Path,
        help=(
            "JSON file receiving stripped IR-10 metadata. "
            "Default: <input stem>.ir10_metadata.json beside the output."
        ),
    )
    parser.add_argument(
        "--token-counts",
        type=int,
        nargs="+",
        default=[1, 2, 7, 32, 128],
        help="Token counts used for ONNX Runtime output validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed used for validation inputs.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for output comparison.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for output comparison.",
    )
    parser.add_argument(
        "--skip-runtime-validation",
        action="store_true",
        help="Only inspect, rewrite, and run the ONNX checker.",
    )
    return parser.parse_args()


def default_output_path(input_path: Path) -> Path:
    suffix = input_path.suffix or ".onnx"
    return input_path.with_name(f"{input_path.stem}.ir9{suffix}")


def default_sidecar_path(input_path: Path, output_path: Path) -> Path:
    return output_path.parent / f"{input_path.stem}.ir10_metadata.json"


def iter_messages(
    message: Message, path: str = "model"
) -> Iterable[tuple[str, Message]]:
    """Yield a protobuf message and all populated nested protobuf messages."""
    yield path, message

    for field, value in message.ListFields():
        if field.message_type is None:
            continue

        if field.is_repeated:
            for index, child in enumerate(value):
                yield from iter_messages(child, f"{path}.{field.name}[{index}]")
        else:
            yield from iter_messages(value, f"{path}.{field.name}")


def collect_strippable_metadata(
    model: onnx.ModelProto,
) -> tuple[list[dict[str, object]], list[tuple[str, Message]]]:
    """Collect metadata entries and the owning messages that may be cleared."""
    records: list[dict[str, object]] = []
    owners: list[tuple[str, Message]] = []

    for path, message in iter_messages(model):
        message_name = message.DESCRIPTOR.name
        if message_name not in STRIPPABLE_IR10_METADATA_TYPES:
            continue

        metadata_field = message.DESCRIPTOR.fields_by_name.get("metadata_props")
        if metadata_field is None or len(message.metadata_props) == 0:
            continue

        owners.append((path, message))
        records.append(
            {
                "path": path,
                "message_type": message_name,
                "entries": [
                    {"key": entry.key, "value": entry.value}
                    for entry in message.metadata_props
                ],
            }
        )

    return records, owners


def inspect_non_strippable_ir10_features(
    model: onnx.ModelProto,
) -> list[str]:
    """Return reasons why lowering this model to IR 9 would be unsafe."""
    problems: list[str] = []

    for path, message in iter_messages(model):
        message_name = message.DESCRIPTOR.name

        if message_name in UNSUPPORTED_IR10_METADATA_TYPES:
            metadata_field = message.DESCRIPTOR.fields_by_name.get(
                "metadata_props"
            )
            if metadata_field is not None and len(message.metadata_props) > 0:
                problems.append(
                    f"{path} ({message_name}) contains IR-10 metadata_props "
                    "that this converter does not strip automatically"
                )

        overload_field = message.DESCRIPTOR.fields_by_name.get("overload")
        if overload_field is not None and getattr(message, "overload", ""):
            problems.append(
                f"{path} ({message_name}) uses IR-10 function overload "
                f"{getattr(message, 'overload')!r}"
            )

        if message_name == "FunctionProto":
            value_info_field = message.DESCRIPTOR.fields_by_name.get(
                "value_info"
            )
            if value_info_field is not None and len(message.value_info) > 0:
                problems.append(
                    f"{path} contains FunctionProto.value_info, added in IR 10"
                )

        for field, value in message.ListFields():
            if field.message_type is not None:
                continue

            if field.name not in {"data_type", "elem_type"}:
                continue

            values: Sequence[int]
            if field.is_repeated:
                values = value
            else:
                values = [value]

            for type_value in values:
                if type_value in IR10_ONLY_TENSOR_TYPES:
                    problems.append(
                        f"{path}.{field.name} uses "
                        f"{IR10_ONLY_TENSOR_TYPES[type_value]} "
                        f"(tensor type {type_value}), added in IR 10"
                    )

    return problems


def clear_metadata(owners: Sequence[tuple[str, Message]]) -> None:
    for _path, message in owners:
        del message.metadata_props[:]


def opset_summary(model: onnx.ModelProto) -> list[tuple[str, int]]:
    return [
        (entry.domain or "ai.onnx", int(entry.version))
        for entry in model.opset_import
    ]


def print_model_summary(label: str, model: onnx.ModelProto) -> None:
    print(f"{label} IR version: {model.ir_version}")
    print(f"{label} opsets: {opset_summary(model)}")
    print(f"{label} graph inputs: {[value.name for value in model.graph.input]}")
    print(
        f"{label} graph outputs: {[value.name for value in model.graph.output]}"
    )


def dimension_value(
    dimension: onnx.TensorShapeProto.Dimension,
) -> int | None:
    if dimension.HasField("dim_value"):
        return int(dimension.dim_value)
    return None


def make_canonical_tokens(
    num_tokens: int, rng: np.random.Generator
) -> np.ndarray:
    """Build canonical ml_ldmx ECal+TriggerPad token rows."""
    if num_tokens <= 0:
        raise ValueError("Token counts must be positive.")

    # [is_ecal, is_tpad, ecal_x, ecal_y, ecal_z, ecal_energy,
    #  tpad_centroid, tpad_pe]
    tokens = np.zeros((num_tokens, 8), dtype=np.float32)

    num_tpad = 0 if num_tokens == 1 else max(1, num_tokens // 4)
    num_ecal = num_tokens - num_tpad

    tokens[:num_ecal, 0] = 1.0
    tokens[:num_ecal, 2] = rng.normal(0.0, 100.0, size=num_ecal)
    tokens[:num_ecal, 3] = rng.normal(0.0, 100.0, size=num_ecal)
    tokens[:num_ecal, 4] = rng.uniform(200.0, 800.0, size=num_ecal)
    tokens[:num_ecal, 5] = rng.lognormal(0.0, 1.0, size=num_ecal)

    if num_tpad:
        tokens[num_ecal:, 1] = 1.0
        tokens[num_ecal:, 6] = rng.uniform(-500.0, 500.0, size=num_tpad)
        tokens[num_ecal:, 7] = rng.lognormal(2.0, 0.8, size=num_tpad)

    return tokens


def make_generic_input(
    value_info: onnx.ValueInfoProto,
    token_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    tensor_type = value_info.type.tensor_type
    if tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise RuntimeError(
            f"Validation only supports FLOAT inputs; {value_info.name!r} "
            f"uses element type {tensor_type.elem_type}."
        )

    dimensions = list(tensor_type.shape.dim)
    if not dimensions:
        raise RuntimeError(
            f"Input {value_info.name!r} has no declared tensor dimensions."
        )

    shape: list[int] = []
    for index, dimension in enumerate(dimensions):
        fixed = dimension_value(dimension)
        if fixed is not None:
            shape.append(fixed)
        elif index == 0:
            shape.append(token_count)
        else:
            raise RuntimeError(
                f"Cannot synthesize input {value_info.name!r}: dynamic "
                f"dimension {index} is not the token dimension."
            )

    return rng.normal(size=shape).astype(np.float32)


def build_validation_feeds(
    model: onnx.ModelProto,
    token_count: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    inputs = list(model.graph.input)
    if len(inputs) != 1:
        raise RuntimeError(
            "Automatic validation currently expects exactly one model input; "
            f"found {[value.name for value in inputs]}."
        )

    input_info = inputs[0]
    tensor_type = input_info.type.tensor_type
    dimensions = list(tensor_type.shape.dim)

    if (
        tensor_type.elem_type == onnx.TensorProto.FLOAT
        and len(dimensions) == 2
        and dimension_value(dimensions[1]) == 8
    ):
        values = make_canonical_tokens(token_count, rng)
    else:
        values = make_generic_input(input_info, token_count, rng)

    return {input_info.name: values}


def validate_with_onnxruntime(
    original_path: Path,
    converted_path: Path,
    model: onnx.ModelProto,
    token_counts: Sequence[int],
    seed: int,
    rtol: float,
    atol: float,
) -> None:
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Runtime validation requires onnxruntime. Install it with "
            "`python -m pip install onnxruntime` or use "
            "`--skip-runtime-validation`."
        ) from error

    print(f"ONNX Runtime Python version: {ort.__version__}")

    providers = ["CPUExecutionProvider"]
    original_session = ort.InferenceSession(
        str(original_path), providers=providers
    )
    converted_session = ort.InferenceSession(
        str(converted_path), providers=providers
    )

    original_outputs = [
        output.name for output in original_session.get_outputs()
    ]
    converted_outputs = [
        output.name for output in converted_session.get_outputs()
    ]

    if original_outputs != converted_outputs:
        raise RuntimeError(
            "Output-name mismatch after conversion: "
            f"{original_outputs} versus {converted_outputs}"
        )

    rng = np.random.default_rng(seed)

    for token_count in token_counts:
        feeds = build_validation_feeds(model, token_count, rng)
        original_values = original_session.run(original_outputs, feeds)
        converted_values = converted_session.run(converted_outputs, feeds)

        if len(original_values) != len(converted_values):
            raise RuntimeError(
                f"Output-count mismatch for {token_count} tokens."
            )

        largest_absolute_difference = 0.0

        for output_name, original, converted in zip(
            original_outputs, original_values, converted_values
        ):
            if original.shape != converted.shape:
                raise RuntimeError(
                    f"Shape mismatch for output {output_name!r} with "
                    f"{token_count} tokens: {original.shape} versus "
                    f"{converted.shape}"
                )

            if original.size:
                difference = float(
                    np.max(
                        np.abs(
                            original.astype(np.float64)
                            - converted.astype(np.float64)
                        )
                    )
                )
                largest_absolute_difference = max(
                    largest_absolute_difference, difference
                )

            np.testing.assert_allclose(
                converted,
                original,
                rtol=rtol,
                atol=atol,
                err_msg=(
                    f"Converted output {output_name!r} differs with "
                    f"{token_count} tokens"
                ),
            )

        print(
            f"Validated {token_count:>5} tokens; "
            f"maximum absolute difference = "
            f"{largest_absolute_difference:.3e}"
        )


def write_metadata_sidecar(
    sidecar_path: Path,
    input_path: Path,
    output_path: Path,
    model: onnx.ModelProto,
    metadata_records: Sequence[dict[str, object]],
) -> None:
    document = {
        "format_version": 1,
        "source_model": str(input_path),
        "converted_model": str(output_path),
        "source_ir_version": int(model.ir_version),
        "target_ir_version": IR9,
        "opsets": [
            {"domain": domain, "version": version}
            for domain, version in opset_summary(model)
        ],
        "stripped_field": "metadata_props",
        "stripped_message_types": sorted(
            STRIPPABLE_IR10_METADATA_TYPES
        ),
        "records": list(metadata_records),
    }

    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()

    input_path = args.input.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else default_output_path(input_path)
    )
    sidecar_path = (
        args.metadata_sidecar.expanduser().resolve()
        if args.metadata_sidecar is not None
        else default_sidecar_path(input_path, output_path)
    )

    if not input_path.is_file():
        print(f"Input model does not exist: {input_path}", file=sys.stderr)
        return 2

    if input_path == output_path:
        print("Input and output paths must differ.", file=sys.stderr)
        return 2

    model = onnx.load(str(input_path))
    print_model_summary("Input", model)

    if model.ir_version > 10:
        print(
            f"Refusing to convert IR {model.ir_version}; this script only "
            "handles the IR-10-to-IR-9 case.",
            file=sys.stderr,
        )
        return 2

    if model.ir_version < 10:
        print(
            f"The input is already IR {model.ir_version}; no downgrade is "
            "needed.",
            file=sys.stderr,
        )
        return 2

    minimum_ir = onnx.helper.find_min_ir_version_for(
        model.opset_import, ignore_unknown=False
    )
    print(f"Minimum IR implied by opset imports: {minimum_ir}")

    if minimum_ir > IR9:
        print(
            f"Opset imports require IR {minimum_ir}, so this model cannot be "
            "represented as IR 9.",
            file=sys.stderr,
        )
        return 1

    problems = inspect_non_strippable_ir10_features(model)
    if problems:
        print(
            "Refusing to downgrade because the model uses non-strippable "
            "IR-10-only content:",
            file=sys.stderr,
        )
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    metadata_records, metadata_owners = collect_strippable_metadata(model)
    metadata_entry_count = sum(
        len(record["entries"]) for record in metadata_records
    )

    print(
        f"Found {metadata_entry_count} metadata entries on "
        f"{len(metadata_records)} graph objects."
    )

    write_metadata_sidecar(
        sidecar_path=sidecar_path,
        input_path=input_path,
        output_path=output_path,
        model=model,
        metadata_records=metadata_records,
    )
    print(f"Saved stripped metadata: {sidecar_path}")

    clear_metadata(metadata_owners)
    model.ir_version = IR9

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))

    # Reload the serialized artifact and validate the file actually written.
    converted = onnx.load(str(output_path))
    onnx.checker.check_model(converted, full_check=True)

    remaining_metadata, _owners = collect_strippable_metadata(converted)
    if remaining_metadata:
        raise RuntimeError(
            "Converted model still contains strippable IR-10 metadata."
        )

    print_model_summary("Output", converted)
    print(f"ONNX checker passed: {output_path}")

    if not args.skip_runtime_validation:
        validate_with_onnxruntime(
            original_path=input_path,
            converted_path=output_path,
            model=converted,
            token_counts=args.token_counts,
            seed=args.seed,
            rtol=args.rtol,
            atol=args.atol,
        )
        print("Original and converted ONNX Runtime outputs match.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
