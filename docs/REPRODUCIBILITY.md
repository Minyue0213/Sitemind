# Reproducing SiteMind

This guide separates the dependency-light core, GOOSE-Ex perception, calibrated ALICE fusion and sequence planning pipelines. Large datasets, generated artifacts and model checkpoints are not stored in Git; checkpoints and the preserved experiment bundle are available from GitHub Releases.

## 1. Core package and tests

Requirements: Python 3.9–3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest
```

Run the self-contained synthetic risk-fusion and planning example:

```bash
python scripts/run_synthetic_demo.py
```

## 2. Expected local directories

The following paths are ignored by Git:

```text
data/          Downloaded and extracted datasets
models/        Official and fine-tuned checkpoints
artifacts/     Predictions, intermediate arrays, metrics and visualizations
outputs/       Optional experiment outputs
```

Do not commit raw GOOSE data or pretrained weights. Download them from the [official setup page](https://goose-dataset.de/docs/setup/).

## 3. GOOSE-Ex semantic perception

The completed SiteMind run used:

- 3,989 GOOSE-Ex training pairs;
- 407 GOOSE-Ex validation pairs;
- 64 semantic classes;
- 512 × 512 inputs;
- 15 epochs, batch size 8;
- AdamW, learning rate `1e-4`, weight decay `1e-4`;
- equal-weight cross-entropy and Generalized Dice loss;
- automatic mixed precision on an RTX 3090.

The formal result did **not** use the optional asymmetric safety loss: `--safety-loss-weight` was `0`. Safety metrics were used for checkpoint selection, followed by an independently evaluated entropy gate.

The training environment is intentionally isolated from the lightweight package because SuperGradients 3.7.1 is tied to an older PyTorch/CUDA stack. Start from a compatible CUDA image and install [`requirements-autodl.txt`](../requirements-autodl.txt).

```bash
python scripts/train_ppliteseg.py \
  data/processed/gooseEx_2d_train \
  data/processed/gooseEx_2d_val \
  --checkpoint models/ppliteseg_class_512.pth \
  --output artifacts/training/ppliteseg_gooseex_ce_dice \
  --epochs 15 \
  --batch-size 8 \
  --amp
```

Run inference and evaluation with the resulting checkpoint and the corresponding prediction directory:

```bash
python scripts/run_ppliteseg.py \
  data/processed/gooseEx_2d_val/images/val \
  --checkpoint models/best_safety.pth \
  --output artifacts/predictions_val_finetuned_safety

python scripts/evaluate_predictions.py \
  data/processed/gooseEx_2d_val \
  artifacts/predictions_val_finetuned_safety \
  --output artifacts/evaluation_val_finetuned_safety
```

Use each script's `--help` output as the source of truth if the downloaded directory layout differs.

## 4. LiDAR geometry evaluation

After extracting the official GOOSE-Ex 3D validation archive:

```bash
python scripts/evaluate_geometry_planning.py \
  data/processed/gooseEx_3d_val \
  --endpoint-mode component \
  --output-dir artifacts/geometry_planning_eval_component
```

The geometry configuration is in [`configs/geometry.yaml`](../configs/geometry.yaml). Thresholds are prototype parameters and should be calibrated for the target machine and worksite.

## 5. Calibrated ALICE frame fusion

Install the ROS bag reader first:

```bash
python -m pip install -e '.[dev,ros]'
```

Extract a synchronized camera/point-cloud frame with calibration from the official ALICE reference bag:

```bash
python scripts/extract_alice_ros_frame.py \
  /path/to/alice_sequence02.bag \
  TARGET_CAMERA_TIMESTAMP_NS \
  --output artifacts/alice_seq02_raw
```

Run the segmentation model on the extracted image, then build calibrated fusion layers:

```bash
python scripts/build_calibrated_fusion_layers.py \
  artifacts/alice_seq02_raw/calibrated_frame.npz \
  artifacts/alice_seq02_raw/camera.png \
  /path/to/prediction.png \
  /path/to/uncertainty.png \
  /path/to/goose_label_mapping.csv \
  --prediction-source-image /path/to/model_input.png \
  --output-dir artifacts/alice_seq02_fusion
```

The projection path uses the recorded camera intrinsics, `plumb_bob` distortion and TF extrinsics. It performs depth-based occlusion handling before projecting sampled semantic risk into the LiDAR BEV.

## 6. Sequence fusion, temporal memory and worksite-goal planning

Extract consecutive frames and retain their world poses:

```bash
python scripts/extract_alice_sequence.py \
  /path/to/alice_sequence02.bag \
  --count 231 \
  --reference-frame e2_map \
  --output artifacts/alice_seq02_sequence_full
```

After PP-LiteSeg predictions have been generated for the extracted images:

```bash
python scripts/run_ppliteseg.py \
  artifacts/alice_seq02_sequence_full \
  --checkpoint models/best_safety.pth \
  --output artifacts/alice_seq02_sequence_full_predictions

python scripts/build_sequence_fusion.py \
  artifacts/alice_seq02_sequence_full \
  artifacts/alice_seq02_sequence_full_predictions \
  data/processed/gooseEx_2d_val/goose_label_mapping.csv \
  --output artifacts/alice_seq02_sequence_full_fusion

python scripts/build_temporal_sequence.py \
  artifacts/alice_seq02_sequence_full_fusion \
  --window-size 7 \
  --output artifacts/alice_seq02_sequence_full_temporal
```

Render the frozen demonstration goal used by the project:

```bash
python scripts/render_sequence_preview.py \
  artifacts/alice_seq02_sequence_full_temporal \
  artifacts/alice_seq02_sequence_full \
  artifacts/alice_seq02_sequence_full_fusion \
  artifacts/alice_seq02_sequence_full/sequence.json \
  --goal-frame 19 \
  --goal-pixel 286 1063 \
  --start-frame 19 \
  --count 149 \
  --fps 5 \
  --output artifacts/sitemind_next_worksite_demo.gif
```

The selected pixel must have a valid, non-occluded measured ground correspondence. It is converted once to a world point and transformed back into the current local frame on every subsequent observation.

## 7. Validation boundaries

- The 407-frame GOOSE-Ex validation split supports held-out 2D perception and independent 3D geometry evaluation.
- The compact validation downloads do not provide the per-recording extrinsics required to fuse every validation image with its point cloud.
- The ALICE Sequence02 raw bag supports genuine calibrated and temporal fusion, but its annotated showcase frames belong to the training split.
- Sequence02 therefore demonstrates engineering integration; it is not reported as held-out fusion generalization.
- The route is a risk-grid reference path, not a track-aware command trajectory.

## 8. Restore the preserved experiment outputs

The generated outputs from the documented run are preserved as a versioned Release asset. Follow [`EXPERIMENT_ARCHIVE.md`](EXPERIMENT_ARCHIVE.md) to download, verify and extract it. The exact upstream inputs and their SHA-256 checksums are recorded in [`DATA_MANIFEST.md`](DATA_MANIFEST.md).
