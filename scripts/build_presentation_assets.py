"""Regenerate the self-contained presentation figures from local artifacts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(script: str, *arguments: str) -> None:
    command = [sys.executable, str(ROOT / "scripts" / script), *arguments]
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    run(
        "render_evaluation_scorecard.py",
        "artifacts/evaluation_val/summary.json",
        "artifacts/evaluation_val_finetuned_safety/summary.json",
        "artifacts/abstention_sweep_safety/recommended.json",
        "--output",
        "artifacts/evaluation_scorecard.png",
    )
    run(
        "render_real_planning_demo.py",
        "artifacts/goose3d_geometry_70.npz",
        "--evaluation-summary",
        "artifacts/geometry_planning_eval_component/summary.json",
        "--fixed-summary",
        "artifacts/geometry_planning_eval/summary.json",
        "--output",
        "artifacts/real_planning_demo.png",
    )
    run(
        "render_calibrated_fusion_demo.py",
        "artifacts/alice_seq02_fusion/fusion_layers.npz",
        "artifacts/alice_seq02_raw/camera.png",
        "artifacts/alice_seq02_fusion/projection_overlay.png",
        "--output",
        "artifacts/calibrated_fusion_demo.png",
    )
    print("presentation_assets=3 status=ok")


if __name__ == "__main__":
    main()
