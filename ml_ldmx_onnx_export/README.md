# ECalTpadTransformer ONNX export

## Dependencies (can be dealt with in a python venv)

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

python scripts/convert_onnx_ir10_to_ir9.py outputs/hit_classifier_baseline/my_run-export/model.onnx
```

where the conversion step has to happen to respect certain field conventions in the older version IR9, which is what `ldmx-sw`ONNX is currently using.

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


## A concrete example

Get some model to where you want to run, like
```bash
cp mldmx/outputs/cosmos_baselines/tpad_transformer_3e_1M_h128_3layers_l4_lr3e4 outputs/.
```

Then run like
```bash
> python scripts/inspect_run_to_export.py --run-dir outputs/tpad_transformer_3e_1M_h128_3layers_l4_lr3e4/
Config model: ECalTpadTransformer
Checkpoint keys: ['args', 'best_val_loss', 'epoch', 'feature_norm', 'history', 'model_kwargs', 'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'splits', 'valid_labels']
Model kwargs: {'in_dim': 8, 'd_model': 128, 'nhead': 8, 'num_layers': 3, 'dim_feedforward': 512, 'dropout': 0.1, 'out_dim': 3}
Feature normalization: {'first_continuous_col': 2, 'mean': [-1.805290699005127, 0.011602000333368778, 380.58233642578125, 67.23377990722656, 0.17699654400348663, 1.2975337505340576], 'std': [29.26585578918457, 3\
4.63003921508789, 71.90702056884766, 104.3809585571289, 2.356607437133789, 15.580365180969238]}
Valid labels: (1, 2, 3)

```
If this looks like the expected model, proceed with

```bash
> python scripts/export_hit_classifier_onnx.py --run-dir outputs/tpad_transformer_3e_1M_h128_3layers_l4_lr3e4/

> python scripts/inspect_run_to_export.py --run-dir outputs/tpad_transformer_3e_1M_h128_3layers_l4_lr3e4/
python scripts/convert_onnx_ir10_to_ir9.py outputs/tpad_transformer_3e_1M_h128_3layers_l4_lr3e4/export/model.onnx

```
and point to that model.onnx in a dedicated model import `ldmx-sw` configuration file.
As a courtesy, a minimal example snippet (that works at the time of writing, but these things are always moving, NB):

```python
classifier = EcalHitOriginClassifier()
classifier.ecal_hit_collection = "EcalRecHits"
classifier.ecal_hit_pass_name = ""
classifier.trigger_pad_collection = "TriggerPadTracks"
classifier.trigger_pad_pass_name = ""

classifier.model_path = "/absolute/path/to/model.onnx"
classifier.input_name = "tokens"
classifier.output_name = "logits"

classifier.use_trigger_pad_context = True
classifier.output_includes_context_tokens = True

# Preprocessing is already embedded by the supplied exporter.
classifier.apply_log1p_ecal_energy = False
classifier.apply_log1p_trigger_pad_pe = False
classifier.feature_means = []
classifier.feature_stds = []

p.sequence = [classifier]
```


## Notes

- `classification = argmax(logits) + 1` matches the deployment convention.
- `confidence = max(softmax(logits))` should be computed in C++.
- `valid_training_labels` is recorded in metadata for auditability; canonical-y
  class indices are still exported as columns `0..N-1`.
- The exporter intentionally targets `ECalTpadTransformer`. The wrapper pattern
  can later be generalized to the ECal-only Transformer.
