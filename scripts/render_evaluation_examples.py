"""Render understandable false-safe examples for baseline/training comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sitemind.data import discover_frame_pairs, load_label_ids, load_label_mapping, load_rgb
from sitemind.fusion import load_risk_values, semantic_ids_to_risk


ROOT = Path(__file__).resolve().parents[1]


def risk_colors(risk: np.ndarray) -> np.ndarray:
    risk = np.clip(risk, 0.0, 1.0)
    rgb = np.zeros((*risk.shape, 3), dtype=np.uint8)
    low = risk <= 0.5
    rgb[..., 0] = np.where(low, 2 * risk * 255, 255).astype(np.uint8)
    rgb[..., 1] = np.where(low, 200 + 2 * risk * 55, 2 * (1 - risk) * 255).astype(np.uint8)
    rgb[..., 2] = np.where(risk >= 0.99, 12, 30).astype(np.uint8)
    return rgb


def panel(title: str, image: Image.Image, *, width: int = 640) -> Image.Image:
    height = round(image.height * width / image.width)
    image = image.resize((width, height), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (width, height + 42), "#101820")
    canvas.paste(image, (0, 42))
    ImageDraw.Draw(canvas).text((12, 13), title, fill="white")
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("prediction_root", type=Path)
    parser.add_argument("--indices", default="0")
    parser.add_argument("--split", default="val")
    parser.add_argument("--risk-config", type=Path, default=ROOT / "configs" / "risk_mapping.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "evaluation_examples")
    args = parser.parse_args()

    pairs = discover_frame_pairs(args.dataset_root, args.split)
    mapping = load_label_mapping(args.dataset_root / "goose_label_mapping.csv")
    class_risk = load_risk_values(args.risk_config)
    image_root = args.dataset_root / "images" / args.split
    args.output.mkdir(parents=True, exist_ok=True)

    for index in [int(value) for value in args.indices.split(",") if value.strip()]:
        if not 0 <= index < len(pairs):
            raise SystemExit(f"frame index {index} outside 0..{len(pairs) - 1}")
        pair = pairs[index]
        relative = pair.image_path.relative_to(image_root).with_suffix(".png")
        prediction_path = args.prediction_root / "label_ids" / relative
        rgb = load_rgb(pair.image_path)
        target = load_label_ids(pair.label_path)
        prediction = load_label_ids(prediction_path)
        if prediction.shape != target.shape:
            prediction = np.asarray(
                Image.fromarray(prediction.astype(np.uint8)).resize(
                    (target.shape[1], target.shape[0]), Image.Resampling.NEAREST
                )
            )
        gt_risk = semantic_ids_to_risk(target, mapping, class_risk)
        predicted_risk = semantic_ids_to_risk(prediction, mapping, class_risk)
        hazardous = gt_risk >= 0.75
        false_safe = hazardous & (predicted_risk < gt_risk)
        false_safe_rate = false_safe.sum() / max(1, hazardous.sum())

        overlay = rgb.astype(np.float32)
        overlay[false_safe] = 0.35 * overlay[false_safe] + 0.65 * np.array(
            [255, 0, 30], dtype=np.float32
        )
        tiles = [
            panel("1  Camera image", Image.fromarray(rgb), width=640),
            panel("2  Ground-truth risk", Image.fromarray(risk_colors(gt_risk)), width=640),
            panel("3  Predicted risk", Image.fromarray(risk_colors(predicted_risk)), width=640),
            panel(
                f"4  Missed danger in red — false-safe {100 * false_safe_rate:.1f}%",
                Image.fromarray(overlay.astype(np.uint8)),
                width=640,
            ),
        ]
        gap = 10
        canvas = Image.new(
            "RGB",
            (tiles[0].width, sum(tile.height for tile in tiles) + gap * (len(tiles) - 1)),
            "#263238",
        )
        y = 0
        for tile in tiles:
            canvas.paste(tile, (0, y))
            y += tile.height + gap
        output_path = args.output / f"frame_{index:04d}.png"
        canvas.save(output_path)
        print(f"output={output_path} false_safe_rate={false_safe_rate:.6f}")


if __name__ == "__main__":
    main()
