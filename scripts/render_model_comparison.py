"""Render a presentation-ready baseline/fine-tuning/safety comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sitemind.data import discover_frame_pairs, load_label_ids, load_label_mapping, load_rgb
from sitemind.fusion import load_risk_values, semantic_ids_to_risk


ROOT = Path(__file__).resolve().parents[1]
BACKGROUND = "#0b1118"
PANEL_BACKGROUND = "#121c26"
ABSTAIN_COLOR = np.array([142, 68, 173], dtype=np.uint8)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:
        return ImageFont.load_default()


def risk_colors(risk: np.ndarray) -> np.ndarray:
    """Use discrete, presentation-friendly colors for risk levels."""
    rgb = np.empty((*risk.shape, 3), dtype=np.uint8)
    rgb[risk < 0.25] = (24, 183, 92)
    rgb[(risk >= 0.25) & (risk < 0.5)] = (170, 214, 48)
    rgb[(risk >= 0.5) & (risk < 0.75)] = (255, 193, 7)
    rgb[(risk >= 0.75) & (risk < 1.0)] = (255, 112, 40)
    rgb[risk >= 1.0] = (229, 57, 53)
    return rgb


def false_safe_rate(gt_risk: np.ndarray, predicted_risk: np.ndarray) -> float:
    hazardous = gt_risk >= 0.75
    misses = hazardous & (predicted_risk < gt_risk)
    return float(np.count_nonzero(misses) / max(1, hazardous.sum()))


def make_panel(
    title: str,
    subtitle: str,
    image: np.ndarray,
    *,
    width: int,
    height: int,
    resample: Image.Resampling = Image.Resampling.NEAREST,
) -> Image.Image:
    header = 76
    canvas = Image.new("RGB", (width, height + header), PANEL_BACKGROUND)
    source = Image.fromarray(image).resize((width, height), resample)
    canvas.paste(source, (0, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 10), title, fill="white", font=font(24, bold=True))
    draw.text((18, 43), subtitle, fill="#a9bdcc", font=font(17))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("baseline_root", type=Path)
    parser.add_argument("finetuned_root", type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--entropy-threshold", type=float, default=0.23529411764705882)
    parser.add_argument("--risk-config", type=Path, default=ROOT / "configs" / "risk_mapping.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "model_comparison.png")
    args = parser.parse_args()

    pairs = discover_frame_pairs(args.dataset_root, "val")
    pair = pairs[args.index]
    image_root = args.dataset_root / "images" / "val"
    relative = pair.image_path.relative_to(image_root).with_suffix(".png")
    mapping = load_label_mapping(args.dataset_root / "goose_label_mapping.csv")
    class_risk = load_risk_values(args.risk_config)
    rgb = load_rgb(pair.image_path)
    target = load_label_ids(pair.label_path)
    baseline = load_label_ids(args.baseline_root / "label_ids" / relative)
    finetuned = load_label_ids(args.finetuned_root / "label_ids" / relative)
    entropy = load_label_ids(args.finetuned_root / "uncertainty" / relative) / 255.0
    gt_risk = semantic_ids_to_risk(target, mapping, class_risk)
    baseline_risk = semantic_ids_to_risk(baseline, mapping, class_risk)
    finetuned_risk = semantic_ids_to_risk(finetuned, mapping, class_risk)
    abstained = entropy >= args.entropy_threshold
    final_risk = finetuned_risk.copy()
    final_risk[abstained] = 1.0

    baseline_fsr = false_safe_rate(gt_risk, baseline_risk)
    finetuned_fsr = false_safe_rate(gt_risk, finetuned_risk)
    final_fsr = false_safe_rate(gt_risk, final_risk)
    coverage = 1.0 - float(abstained.mean())

    final_map = risk_colors(final_risk)
    final_map[abstained] = ABSTAIN_COLOR
    uncertainty_map = np.zeros((*entropy.shape, 3), dtype=np.uint8)
    uncertainty_map[:] = (30, 39, 48)
    uncertainty_map[abstained] = ABSTAIN_COLOR

    width, height = 600, 338
    panels = [
        make_panel(
            "1. Camera input", "What the excavator sees", rgb,
            width=width, height=height, resample=Image.Resampling.BILINEAR,
        ),
        make_panel(
            "2. Ground-truth safety map", "Reference annotated by the dataset", risk_colors(gt_risk),
            width=width, height=height,
        ),
        make_panel(
            "3. Original model", f"Danger missed: {100 * baseline_fsr:.1f}%", risk_colors(baseline_risk),
            width=width, height=height,
        ),
        make_panel(
            "4. Fine-tuned SiteMind", f"Danger missed: {100 * finetuned_fsr:.1f}%", risk_colors(finetuned_risk),
            width=width, height=height,
        ),
        make_panel(
            "5. Uncertainty gate", f"Purple = ask LiDAR / slow down ({100 * (1 - coverage):.1f}% of image)", uncertainty_map,
            width=width, height=height,
        ),
        make_panel(
            "6. Final guarded output",
            f"Danger missed: {100 * final_fsr:.1f}%  |  camera coverage: {100 * coverage:.1f}%",
            final_map, width=width, height=height,
        ),
    ]

    gap = 16
    margin = 24
    title_height = 104
    footer_height = 90
    panel_height = panels[0].height
    canvas = Image.new(
        "RGB",
        (3 * width + 4 * gap, title_height + 2 * panel_height + gap + footer_height),
        BACKGROUND,
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 18), "SiteMind: from generic vision to safety-aware perception", fill="white", font=font(32, bold=True))
    draw.text(
        (margin, 62),
        "Red/orange = danger  |  Purple = model is uncertain, so the system refuses to call it safe",
        fill="#b9c8d3", font=font(20),
    )
    for index, tile in enumerate(panels):
        row, column = divmod(index, 3)
        x = gap + column * (width + gap)
        y = title_height + row * (panel_height + gap)
        canvas.paste(tile, (x, y))

    legend_y = title_height + 2 * panel_height + gap + 23
    legend = [
        ("Safe", (24, 183, 92)),
        ("Low risk", (170, 214, 48)),
        ("Caution", (255, 193, 7)),
        ("Danger", (229, 57, 53)),
        ("Blocked / uncertain", tuple(int(value) for value in ABSTAIN_COLOR)),
    ]
    x = margin
    for label, color in legend:
        draw.rounded_rectangle((x, legend_y, x + 38, legend_y + 38), radius=5, fill=color)
        draw.text((x + 50, legend_y + 7), label, fill="white", font=font(18))
        x += 260

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(
        f"output={args.output} frame={relative} baseline_fsr={baseline_fsr:.6f} "
        f"finetuned_fsr={finetuned_fsr:.6f} final_fsr={final_fsr:.6f} coverage={coverage:.6f}"
    )


if __name__ == "__main__":
    main()
