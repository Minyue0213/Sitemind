"""Build calibrated fusion layers for every extracted sequence frame in parallel."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]


def build_one(command: list[str], frame_name: str) -> tuple[str, float, str]:
    started = perf_counter()
    process = subprocess.run(command, check=False, capture_output=True, text=True)
    elapsed = perf_counter() - started
    if process.returncode:
        detail = process.stderr.strip() or process.stdout.strip()
        raise RuntimeError(f"{frame_name} failed: {detail}")
    return frame_name, elapsed, process.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("frame_root", type=Path)
    parser.add_argument("prediction_root", type=Path)
    parser.add_argument("label_mapping", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/alice_sequence_fusion"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resolution", type=float, default=0.25)
    args = parser.parse_args()
    if args.workers <= 0 or args.limit < 0:
        raise SystemExit("workers must be positive and limit must be non-negative")

    frame_dirs = sorted(path.parent for path in args.frame_root.glob("frame_*/calibrated_frame.npz"))
    if args.limit:
        frame_dirs = frame_dirs[: args.limit]
    if not frame_dirs:
        raise SystemExit(f"no extracted frames found below {args.frame_root}")

    jobs: list[tuple[list[str], str]] = []
    for frame_dir in frame_dirs:
        name = frame_dir.name
        prediction = args.prediction_root / "label_ids" / name / "camera.png"
        uncertainty = args.prediction_root / "uncertainty" / name / "camera.png"
        if not prediction.is_file() or not uncertainty.is_file():
            raise SystemExit(f"missing prediction or uncertainty for {name}")
        output_dir = args.output / name
        command = [
            sys.executable,
            str(ROOT / "scripts" / "build_calibrated_fusion_layers.py"),
            str(frame_dir / "calibrated_frame.npz"),
            str(frame_dir / "camera.png"),
            str(prediction),
            str(uncertainty),
            str(args.label_mapping),
            "--prediction-source-image",
            str(frame_dir / "camera.png"),
            "--resolution",
            str(args.resolution),
            "--output-dir",
            str(output_dir),
        ]
        jobs.append((command, name))

    args.output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(build_one, command, name): name for command, name in jobs
        }
        for index, future in enumerate(as_completed(futures), start=1):
            name, elapsed, _ = future.result()
            results.append({"frame_name": name, "elapsed_seconds": elapsed})
            print(f"[{index}/{len(jobs)}] {name} {elapsed:.2f}s", flush=True)
    elapsed = perf_counter() - started
    summary = {
        "frame_count": len(results),
        "workers": args.workers,
        "wall_time_seconds": elapsed,
        "throughput_fps": len(results) / elapsed,
        "mean_frame_worker_seconds": sum(item["elapsed_seconds"] for item in results)
        / len(results),
        "frames": sorted(results, key=lambda item: item["frame_name"]),
    }
    (args.output / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "frames"}, indent=2))


if __name__ == "__main__":
    main()
