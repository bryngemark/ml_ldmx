# ECalTpadTransformer ONNX export

Copy these files into the matching paths of `ml_ldmx`.

## Dependencies

```bash
python -m pip install --upgrade onnx onnxscript onnxruntime
python -m pip install -e .
```

The modern exporter requires a recent PyTorch release. The scripts prefer the
`dynamo=True` exporter and include `--legacy-exporter` as a fallback.

## Export

```bash
python scripts/export_hit_classifier_onnx.py \
  --run-dir outputs/hit_classifier_baseline/my_run
```

This reads `config.json` and `checkpoints/best.pt`, reconstructs the saved
`ECalTpadTransformer`, embeds the saved raw/log1p transforms and feature
normalization, and writes:

```text
<run-dir>/export/model.onnx
<run-dir>/export/model_metadata.json
```

The ONNX contract is:

```text
input  tokens: float32 [num_tokens, 8]
output logits: float32 [num_tokens, N]
```

Feature order:

```text
[is_ecal, is_tpad, ecal_x, ecal_y, ecal_z,
 ecal_energy, tpad_centroid, tpad_pe]
```

Inputs are raw framework values. Preprocessing is embedded in the model. The
output includes TriggerPad rows because this exports more reliably than dynamic
boolean slicing; deployment should retain logits only where `is_ecal == 1`.

## Validate

```bash
python scripts/validate_hit_classifier_onnx.py \
  --run-dir outputs/hit_classifier_baseline/my_run
```

The validator compares raw logits from PyTorch and ONNX Runtime for several
variable token counts. Real raw token arrays can also be supplied:

```bash
python scripts/validate_hit_classifier_onnx.py \
  --run-dir outputs/hit_classifier_baseline/my_run \
  --input-npy event_001.npy \
  --input-npy event_002.npy
```

Do not validate only the predicted class. Comparing logits catches meaningful
numerical or preprocessing differences that an argmax can hide.

## Notes

- `classification = argmax(logits) + 1` matches the deployment convention.
- `confidence = max(softmax(logits))` should be computed in C++.
- `valid_training_labels` is recorded in metadata for auditability; canonical-y
  class indices are still exported as columns `0..N-1`.
- The exporter intentionally targets `ECalTpadTransformer`. The wrapper pattern
  can later be generalized to the ECal-only Transformer.
