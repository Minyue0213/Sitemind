"""Render RGB, ground-truth semantics and an Oracle semantic risk map."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sitemind.data import discover_frame_pairs, load_label_ids, load_label_mapping, load_rgb
from sitemind.fusion import load_risk_values, semantic_ids_to_risk


ROOT = Path(__file__).resolve().parents[1]


def label_colors(label_ids: np.ndarray) -> np.ndarray:
    ids = label_ids.astype(np.uint32)
    red = ((ids * 67 + 29) % 211 + 35).astype(np.uint8)
    green = ((ids * 97 + 53) % 211 + 35).astype(np.uint8)
    blue = ((ids * 131 + 71) % 211 + 35).astype(np.uint8)
    return np.stack((red, green, blue), axis=-1)


def risk_colors(risk: np.ndarray) -> np.ndarray:
    risk = np.clip(risk, 0.0, 1.0)
    rgb = np.zeros((*risk.shape, 3), dtype=np.uint8)
    low = risk <= 0.5
    rgb[..., 0] = np.where(low, 2 * risk * 255, 255).astype(np.uint8)
    rgb[..., 1] = np.where(low, 200 + 2 * risk * 55, 2 * (1 - risk) * 255).astype(np.uint8)
    rgb[..., 2] = np.where(risk >= 0.99, 12, 30).astype(np.uint8)
    return rgb


def make_panel(
    title: str,
    image: Image.Image,
    width: int = 640,
    *,
    resample: Image.Resampling = Image.Resampling.BILINEAR,
) -> Image.Image:
    height = round(image.height * width / image.width)
    resized = image.resize((width, height), resample)
    canvas = Image.new("RGB", (width, height + 38), "#101820")
    canvas.paste(resized, (0, 38))
    ImageDraw.Draw(canvas).text((12, 11), title, fill="white")
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--risk-config", type=Path, default=ROOT / "configs" / "risk_mapping.yaml"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "oracle_risk.png")
    args = parser.parse_args()

    pairs = discover_frame_pairs(args.dataset_root, args.split)
    if not pairs:
        raise SystemExit("No image/label pairs found")
    if not 0 <= args.index < len(pairs):
        raise SystemExit(f"index must be between 0 and {len(pairs) - 1}")
    pair = pairs[args.index]
    mapping = load_label_mapping(args.dataset_root / "goose_label_mapping.csv")
    class_risk = load_risk_values(args.risk_config)
    rgb = load_rgb(pair.image_path)
    labels = load_label_ids(pair.label_path)
    if rgb.shape[:2] != labels.shape:
        raise SystemExit(
            f"RGB/label shape mismatch: rgb={rgb.shape[:2]}, label={labels.shape}"
        )
    risk = semantic_ids_to_risk(labels, mapping, class_risk)
    color_label_path = pair.label_path.with_name(
        pair.label_path.name.replace("_labelids.png", "_color.png")
    )
    semantic_rgb = (
        load_rgb(color_label_path) if color_label_path.is_file() else label_colors(labels)
    )

    panels = [
        make_panel("1  Real ALICE RGB", Image.fromarray(rgb)),
        make_panel(
            "2  Ground-truth semantics",
            Image.fromarray(semantic_rgb),
            resample=Image.Resampling.NEAREST,
        ),
        make_panel(
            "3  Oracle semantic risk",
            Image.fromarray(risk_colors(risk)),
            resample=Image.Resampling.NEAREST,
        ),
    ]
    gap = 10
    canvas = Image.new(
        "RGB",
        (panels[0].width, sum(panel.height for panel in panels) + gap * 2),
        "#263238",
    )
    y = 0
    for panel in panels:
        canvas.paste(panel, (0, y))
        y += panel.height + gap
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"output={args.output}")
    print(f"sequence={pair.sequence}")
    print(f"timestamp={pair.timestamp}")
    print(f"risk_min={risk.min():.2f}")
    print(f"risk_max={risk.max():.2f}")


if __name__ == "__main__":
    main()
