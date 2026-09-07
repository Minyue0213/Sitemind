"""Run the official GOOSE PP-LiteSeg checkpoint on AutoDL/Linux.

The model construction and image preprocessing follow the official GOOSE
``image_processing/inference.py`` example. Imports for the GPU stack stay in
``main`` so the lightweight local environment can still run the core tests.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def find_images(path: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png"}
    return sorted(candidate for candidate in path.rglob("*") if candidate.suffix.lower() in extensions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_root", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "models" / "ppliteseg_class_512.pth",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "predictions")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--classes", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="0 means all images")
    args = parser.parse_args()

    try:
        import torch
        import torch.nn.functional as functional
        import super_gradients as sg
        from super_gradients.common.object_names import Models
        from torchvision.transforms import ToTensor
    except ImportError as error:
        raise SystemExit(
            "PP-LiteSeg dependencies are unavailable. Run this script in the "
            "AutoDL Python 3.9 / PyTorch 1.13.1 environment."
        ) from error

    if not args.image_root.is_dir():
        raise SystemExit(f"missing image directory: {args.image_root}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"missing checkpoint: {args.checkpoint}")

    images = find_images(args.image_root)
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise SystemExit("no input images found")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = sg.training.models.get(
        model_name=Models.PP_LITE_B_SEG75,
        num_classes=args.classes,
        # The GOOSE checkpoint is a complete model state. Requesting the
        # Cityscapes preset as well triggers an unnecessary network download.
        pretrained_weights=None,
        checkpoint_path=str(args.checkpoint),
    ).to(device)
    model.eval()

    for index, image_path in enumerate(images, start=1):
        with Image.open(image_path) as source:
            rgb = source.convert("RGB")
            original_size = rgb.size
            resized = rgb.resize((args.width, args.height), Image.Resampling.BILINEAR)
            tensor = ToTensor()(resized).unsqueeze(0).to(device)

        with torch.inference_mode():
            logits = model(tensor)
            probabilities = functional.softmax(logits, dim=1)
            confidence, labels = probabilities.max(dim=1, keepdim=True)
            labels = labels.float()
            entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=1)
            entropy = entropy / np.log(args.classes)

        labels = functional.interpolate(labels, size=original_size[::-1], mode="nearest")
        entropy = functional.interpolate(
            entropy.unsqueeze(1),
            size=original_size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        confidence = functional.interpolate(
            confidence,
            size=original_size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        label_array = labels[0, 0].byte().cpu().numpy()
        entropy_array = (entropy[0, 0].clamp(0, 1) * 255).byte().cpu().numpy()
        confidence_array = (
            confidence[0, 0].clamp(0, 1) * 255
        ).byte().cpu().numpy()

        relative = image_path.relative_to(args.image_root).with_suffix(".png")
        label_path = args.output / "label_ids" / relative
        entropy_path = args.output / "uncertainty" / relative
        confidence_path = args.output / "confidence" / relative
        label_path.parent.mkdir(parents=True, exist_ok=True)
        entropy_path.parent.mkdir(parents=True, exist_ok=True)
        confidence_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(label_array).save(label_path)
        Image.fromarray(entropy_array).save(entropy_path)
        Image.fromarray(confidence_array).save(confidence_path)
        print(f"[{index}/{len(images)}] {relative}")

    print(f"device={device}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
