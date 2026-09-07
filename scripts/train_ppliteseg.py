"""Fine-tune the official 64-class GOOSE PP-LiteSeg model on GOOSE-Ex."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

from sitemind.data import discover_frame_pairs, load_label_mapping
from sitemind.evaluation import (
    risk_metrics_from_confusion,
    segmentation_metrics_from_confusion,
    update_confusion_matrix,
)
from sitemind.fusion import load_risk_values


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_root", type=Path)
    parser.add_argument("val_root", type=Path)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "ppliteseg_class_512.pth")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "training" / "ppliteseg_gooseex")
    parser.add_argument("--risk-config", type=Path, default=ROOT / "configs" / "risk_mapping.yaml")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--classes", type=int, default=64)
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--val-offset", type=int, default=0)
    parser.add_argument("--limit-train", type=int, default=0, help="0 uses the full split")
    parser.add_argument("--limit-val", type=int, default=0, help="0 uses the full split")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", choices=("ce", "dice", "ce_dice"), default="ce_dice")
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument(
        "--safety-loss-weight",
        type=float,
        default=0.0,
        help="Weight for differentiable predicted-risk underestimation loss",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Enable CUDA mixed precision")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def jitter_image(image: Image.Image) -> Image.Image:
    brightness = random.uniform(0.85, 1.15)
    contrast = random.uniform(0.85, 1.15)
    saturation = random.uniform(0.85, 1.15)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    return ImageEnhance.Color(image).enhance(saturation)


def emit(message: str) -> None:
    """Write progress even when SuperGradients redirects Python stdout."""

    os.write(1, (message + "\n").encode("utf-8"))


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 2:
        raise SystemExit("epochs must be positive and batch size must be at least 2")
    if args.train_offset < 0 or args.val_offset < 0:
        raise SystemExit("dataset offsets cannot be negative")
    if args.safety_loss_weight < 0 or not 0 <= args.dice_weight <= 1:
        raise SystemExit("invalid safety-loss or Dice weight")

    try:
        import torch
        import torch.nn.functional as functional
        import super_gradients as sg
        from super_gradients.common.object_names import Models
        from super_gradients.training.losses.dice_loss import GeneralizedDiceLoss
        from torch.utils.data import DataLoader, Dataset
        from torchvision.transforms import ToTensor
    except ImportError as error:
        raise SystemExit("Run training in the AutoDL SuperGradients environment") from error

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    train_pairs = discover_frame_pairs(args.train_root, args.train_split)
    val_pairs = discover_frame_pairs(args.val_root, args.val_split)
    train_pairs = train_pairs[args.train_offset :]
    val_pairs = val_pairs[args.val_offset :]
    if args.limit_train > 0:
        train_pairs = train_pairs[: args.limit_train]
    if args.limit_val > 0:
        val_pairs = val_pairs[: args.limit_val]
    if not train_pairs or not val_pairs:
        raise SystemExit("training and validation pairs are both required")
    train_identities = {(pair.sequence, pair.timestamp) for pair in train_pairs}
    val_identities = {(pair.sequence, pair.timestamp) for pair in val_pairs}
    overlap = train_identities & val_identities
    if overlap:
        raise SystemExit(f"training/validation leakage: {len(overlap)} frame identities overlap")

    class PairDataset(Dataset):
        def __init__(self, pairs, training: bool):
            self.pairs = pairs
            self.training = training
            self.to_tensor = ToTensor()

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, index):
            pair = self.pairs[index]
            with Image.open(pair.image_path) as source:
                image = source.convert("RGB")
            with Image.open(pair.label_path) as source:
                label = source.convert("L")
            if self.training and random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                label = label.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if self.training:
                image = jitter_image(image)
            image = image.resize((args.width, args.height), Image.Resampling.BILINEAR)
            label = label.resize((args.width, args.height), Image.Resampling.NEAREST)
            return self.to_tensor(image), torch.from_numpy(np.asarray(label, dtype=np.int64).copy())

    train_loader = DataLoader(
        PairDataset(train_pairs, True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        PairDataset(val_pairs, False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = sg.training.models.get(
        model_name=Models.PP_LITE_B_SEG75,
        num_classes=args.classes,
        pretrained_weights=None,
        checkpoint_path=str(args.checkpoint),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    train_mapping_path = args.train_root / "goose_label_mapping.csv"
    val_mapping_path = args.val_root / "goose_label_mapping.csv"
    if not train_mapping_path.is_file() or not val_mapping_path.is_file():
        raise SystemExit("both dataset roots must contain goose_label_mapping.csv")
    mapping = load_label_mapping(train_mapping_path)
    val_mapping = load_label_mapping(val_mapping_path)
    if mapping != val_mapping:
        raise SystemExit("training and validation label mappings differ")
    risk_values = load_risk_values(args.risk_config)
    unmapped = sorted(set(mapping.values()) - set(risk_values))
    if unmapped:
        raise SystemExit(f"risk mapping is missing semantic classes: {unmapped}")
    class_risk = np.array(
        [risk_values.get(mapping.get(index, ""), 1.0) for index in range(args.classes)],
        dtype=np.float32,
    )
    risk_tensor = torch.from_numpy(class_risk).to(device)
    ignore_ids = [index for index, name in mapping.items() if name == "undefined"]
    ignore_index = ignore_ids[0] if ignore_ids else -100
    dice_loss_function = GeneralizedDiceLoss(ignore_index=ignore_index)
    training_signature = {
        "train_root": str(args.train_root.resolve()),
        "val_root": str(args.val_root.resolve()),
        "train_split": args.train_split,
        "val_split": args.val_split,
        "width": args.width,
        "height": args.height,
        "classes": args.classes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_offset": args.train_offset,
        "val_offset": args.val_offset,
        "limit_train": args.limit_train,
        "limit_val": args.limit_val,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "loss": args.loss,
        "dice_weight": args.dice_weight,
        "safety_loss_weight": args.safety_loss_weight,
        "amp": args.amp,
        "seed": args.seed,
    }

    start_epoch = 0
    best_miou = -1.0
    best_safety_score = -float("inf")
    if args.resume is not None:
        saved = torch.load(args.resume, map_location=device)
        if saved.get("training_signature") != training_signature:
            raise SystemExit("resume checkpoint configuration does not match this run")
        model.load_state_dict(saved["net"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        if saved.get("scaler_state_dict"):
            scaler.load_state_dict(saved["scaler_state_dict"])
        if saved.get("scheduler_state_dict"):
            scheduler.load_state_dict(saved["scheduler_state_dict"])
        start_epoch = int(saved.get("epoch", -1)) + 1
        best_miou = float(saved.get("best_miou", -1.0))
        best_safety_score = float(saved.get("best_safety_score", -float("inf")))
        if saved.get("python_rng_state") is not None:
            random.setstate(saved["python_rng_state"])
            np.random.set_state(saved["numpy_rng_state"])
            torch.set_rng_state(saved["torch_rng_state"])
            if device.type == "cuda" and saved.get("cuda_rng_states") is not None:
                torch.cuda.set_rng_state_all(saved["cuda_rng_states"])

    args.output.mkdir(parents=True, exist_ok=True)
    history_path = args.output / "history.csv"
    if start_epoch == 0 and history_path.exists() and not args.overwrite:
        raise SystemExit(f"output already contains a run: {history_path}; use --overwrite")
    (args.output / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8"
    )
    if start_epoch == 0:
        with history_path.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "mean_iou",
                    "pixel_accuracy",
                    "hazard_recall",
                    "false_safe_rate",
                    "catastrophic_false_safe_rate",
                    "learning_rate",
                ]
            )

    emit(
        f"device={device} train={len(train_pairs)} val={len(val_pairs)} "
        f"size={args.width}x{args.height} batch={args.batch_size} amp={use_amp}"
    )
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for batch_index, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                ce_loss = functional.cross_entropy(logits, labels, ignore_index=ignore_index)
                dice_loss = dice_loss_function(logits, labels)
                if args.loss == "ce":
                    loss = ce_loss
                elif args.loss == "dice":
                    loss = dice_loss
                else:
                    loss = (1 - args.dice_weight) * ce_loss + args.dice_weight * dice_loss
                if args.safety_loss_weight:
                    probabilities = functional.softmax(logits, dim=1)
                    expected_risk = (probabilities * risk_tensor[None, :, None, None]).sum(dim=1)
                    target_risk = risk_tensor[labels.clamp(0, args.classes - 1)]
                    valid = labels != ignore_index
                    safety_loss = functional.relu(target_risk - expected_risk)[valid].mean()
                    loss = loss + args.safety_loss_weight * safety_loss
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            train_loss_sum += float(loss.detach())
            train_batches += 1
            if batch_index % 100 == 0:
                emit(
                    f"epoch={epoch + 1}/{args.epochs} batch={batch_index}/{len(train_loader)} "
                    f"loss={train_loss_sum / train_batches:.4f}"
                )

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        confusion = np.zeros((args.classes, args.classes), dtype=np.int64)
        with torch.inference_mode():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(images)
                    ce_loss = functional.cross_entropy(logits, labels, ignore_index=ignore_index)
                    dice_loss = dice_loss_function(logits, labels)
                    if args.loss == "ce":
                        val_loss = ce_loss
                    elif args.loss == "dice":
                        val_loss = dice_loss
                    else:
                        val_loss = (
                            (1 - args.dice_weight) * ce_loss
                            + args.dice_weight * dice_loss
                        )
                predictions = logits.argmax(dim=1)
                update_confusion_matrix(
                    confusion,
                    labels.cpu().numpy(),
                    predictions.cpu().numpy(),
                    ignore_ids=ignore_ids,
                )
                val_loss_sum += float(val_loss)
                val_batches += 1

        segmentation = segmentation_metrics_from_confusion(
            confusion, ignore_class_ids=ignore_ids
        )
        safety = risk_metrics_from_confusion(confusion, class_risk)
        mean_iou = float(segmentation["mean_iou"])
        safety_score = (
            mean_iou
            + 0.5
            * (float(safety["hazard_recall"]) - float(safety["false_safe_rate"]))
            - float(safety["catastrophic_false_safe_rate"])
        )
        row = [
            epoch + 1,
            train_loss_sum / train_batches,
            val_loss_sum / val_batches,
            mean_iou,
            segmentation["pixel_accuracy"],
            safety["hazard_recall"],
            safety["false_safe_rate"],
            safety["catastrophic_false_safe_rate"],
            optimizer.param_groups[0]["lr"],
        ]
        with history_path.open("a", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(row)

        improved_miou = mean_iou > best_miou
        improved_safety = safety_score > best_safety_score
        best_miou = max(best_miou, mean_iou)
        best_safety_score = max(best_safety_score, safety_score)
        scheduler.step()
        checkpoint = {
            "net": model.state_dict(),
            "acc": mean_iou,
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_miou": best_miou,
            "best_safety_score": best_safety_score,
            "training_signature": training_signature,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "metrics": {
                "mean_iou": mean_iou,
                "pixel_accuracy": float(segmentation["pixel_accuracy"]),
                "hazard_recall": float(safety["hazard_recall"]),
                "false_safe_rate": float(safety["false_safe_rate"]),
                "catastrophic_false_safe_rate": float(safety["catastrophic_false_safe_rate"]),
            },
        }
        latest_path = args.output / "latest.pth"
        torch.save(checkpoint, latest_path)
        if improved_miou:
            shutil.copy2(latest_path, args.output / "best_miou.pth")
        if improved_safety:
            shutil.copy2(latest_path, args.output / "best_safety.pth")
        emit(
            f"epoch={epoch + 1} train_loss={row[1]:.4f} val_loss={row[2]:.4f} "
            f"mIoU={mean_iou:.4f} hazard_recall={float(safety['hazard_recall']):.4f} "
            f"false_safe={float(safety['false_safe_rate']):.4f}"
        )


if __name__ == "__main__":
    main()
