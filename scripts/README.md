# Script index

All scripts expose command-line help:

```bash
python scripts/<name>.py --help
```

## Minimal demo

| Script | Purpose |
|---|---|
| `run_synthetic_demo.py` | Self-contained semantic/geometric risk fusion and A* example. |

## Data inspection and extraction

| Script | Purpose |
|---|---|
| `inspect_goose.py` | Validate GOOSE 2D pairs and label mappings. |
| `extract_alice_ros_frame.py` | Extract a synchronized calibrated frame from an ALICE ROS1 bag. |
| `extract_alice_sequence.py` | Extract a camera/LiDAR sequence with calibration and poses. |
| `audit_alice_pose.py` | Audit sequence motion and TF relationships. |
| `audit_vehicle_origin.py` | Check the chassis, upper-carriage and point-cloud reference frames. |

## Perception training and evaluation

| Script | Purpose |
|---|---|
| `run_ppliteseg.py` | Run PP-LiteSeg inference and write labels, confidence and uncertainty. |
| `train_ppliteseg.py` | Fine-tune the 64-class model on GOOSE-Ex. |
| `evaluate_predictions.py` | Compute segmentation, calibration and false-safe metrics. |
| `sweep_uncertainty_abstention.py` | Evaluate safety/coverage trade-offs for entropy thresholds. |
| `render_model_comparison.py` | Render baseline, fine-tuned and abstention examples. |
| `render_evaluation_scorecard.py` | Build a compact result scorecard. |
| `render_evaluation_examples.py` | Render selected evaluation frames. |

## Geometry, calibration and fusion

| Script | Purpose |
|---|---|
| `render_goose3d_geometry.py` | Build and visualize a real LiDAR terrain-risk grid. |
| `evaluate_geometry_planning.py` | Batch-evaluate geometry risk and local planning. |
| `build_calibrated_fusion_layers.py` | Project image semantics into a LiDAR BEV using recorded calibration. |
| `render_calibrated_fusion_demo.py` | Render the six-stage calibrated fusion explanation. |
| `build_sequence_fusion.py` | Build calibrated semantic/geometric BEV layers for a sequence. |

## Temporal mapping and planning

| Script | Purpose |
|---|---|
| `build_temporal_sequence.py` | Build pose-aligned short-horizon BEV memory. |
| `evaluate_temporal_planning.py` | Compare single-frame and temporal routes. |
| `render_temporal_comparison.py` | Render a temporal-planning comparison animation. |
| `render_sequence_preview.py` | Select/lock a world goal and render continuous risk-aware planning. |
| `render_real_planning_demo.py` | Compare shortest and risk-aware routes on a real geometry grid. |

## Presentation helpers

| Script | Purpose |
|---|---|
| `build_presentation_assets.py` | Rebuild the final compact set of static result figures. |
| `render_oracle_risk.py` | Visualize ground-truth semantic risk for method debugging. |
| `render_predicted_risk.py` | Visualize prediction-derived semantic risk. |
| `render_multimodal_storyboard.py` | Legacy parallel-evidence storyboard retained for reproducibility. |

Scripts expect the repository package to be installed with `pip install -e .` or the project virtual environment to be active.
