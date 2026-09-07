"""Render a Chinese scorecard from baseline, fine-tuned, and guarded metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FONT_CANDIDATES = (
    Path("/System/Library/Fonts/STHeiti Light.ttc"),
    Path("/System/Library/Fonts/PingFang.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.truetype("DejaVuSans.ttf", size=size)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def rounded_card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color: str) -> None:
    draw.rounded_rectangle(box, radius=24, fill=color)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("finetuned", type=Path)
    parser.add_argument("guarded", type=Path)
    parser.add_argument("--platform", default="alice")
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation_scorecard.png"))
    args = parser.parse_args()

    baseline_summary = read_json(args.baseline)
    finetuned_summary = read_json(args.finetuned)
    guarded = read_json(args.guarded)
    baseline = baseline_summary["platforms"][args.platform]
    finetuned = finetuned_summary["platforms"][args.platform]

    canvas = Image.new("RGB", (1800, 1120), "#081019")
    draw = ImageDraw.Draw(canvas)
    white, muted, blue = "#f4f8fb", "#9bb0c1", "#38a3ff"
    green, yellow, red, purple = "#2ac27e", "#f6bd36", "#ef5350", "#9b59d0"

    draw.text((72, 48), "SiteMind 视觉安全模型：真实数据效果", fill=white, font=load_font(48))
    draw.text(
        (74, 116),
        f"最困难平台：ALICE 挖掘机相机 · {baseline['frames']} 帧 · 所有数字均来自未参与训练的验证集",
        fill=muted,
        font=load_font(26),
    )

    cards = [
        ("语义分割 mIoU", baseline["mean_iou"], finetuned["mean_iou"], blue),
        ("像素准确率", baseline["pixel_accuracy"], finetuned["pixel_accuracy"], green),
    ]
    for index, (label, before, after, color) in enumerate(cards):
        x = 72 + index * 566
        rounded_card(draw, (x, 182, x + 522, 374), "#111e2a")
        draw.text((x + 28, 207), label, fill=muted, font=load_font(24))
        draw.text((x + 28, 257), f"{100 * before:.1f}%", fill="#728695", font=load_font(42))
        draw.text((x + 190, 267), "→", fill=muted, font=load_font(32))
        draw.text((x + 256, 247), f"{100 * after:.1f}%", fill=color, font=load_font(58))
    x = 1204
    rounded_card(draw, (x, 182, x + 524, 374), "#1a1730")
    draw.text((x + 28, 207), "安全兜底", fill=muted, font=load_font(24))
    draw.text((x + 28, 251), f"{100 * guarded['coverage']:.1f}%", fill=purple, font=load_font(58))
    draw.text((x + 247, 272), "相机区域直接采用", fill=white, font=load_font(24))
    draw.text(
        (x + 30, 328),
        f"其余 {100 * guarded['abstention_rate']:.1f}% 交给 LiDAR / 减速停车",
        fill=muted,
        font=load_font(21),
    )

    draw.text((72, 428), "真正关系安全的三项指标", fill=white, font=load_font(34))
    draw.text((72, 477), "绿色越高越好；红色越低越好。紫色为加入“不确定就不冒险”的最终方案。", fill=muted, font=load_font(23))

    metrics = [
        ("危险区域召回率 ↑", "hazard_recall", green),
        ("危险漏判率 ↓", "false_safe_rate", red),
        ("严重漏判率 ↓", "catastrophic_false_safe_rate", red),
    ]
    stages = [("原模型", baseline, "#718493"), ("微调后", finetuned, blue), ("最终安全方案", guarded, purple)]
    left, chart_top, chart_width = 430, 566, 1240
    for row, (label, key, accent) in enumerate(metrics):
        y = chart_top + row * 150
        draw.text((76, y + 36), label, fill=white, font=load_font(28))
        draw.line((left, y + 104, left + chart_width, y + 104), fill="#263746", width=2)
        for stage_index, (stage_name, source, color) in enumerate(stages):
            value = float(source[key])
            bar_y = y + stage_index * 32
            bar_width = max(4, int(chart_width * value))
            draw.rounded_rectangle((left, bar_y, left + bar_width, bar_y + 20), radius=10, fill=color)
            draw.text((left + bar_width + 14, bar_y - 6), f"{100 * value:.1f}%", fill=accent, font=load_font(21))
            if row == 0:
                draw.text((left + stage_index * 260, chart_top - 42), stage_name, fill=color, font=load_font(22))

    rounded_card(draw, (72, 1020, 1728, 1082), "#102637")
    draw.text(
        (102, 1035),
        f"一句话结论：微调解决了“看不懂工地”，不确定性门控再把危险漏判从 {100 * baseline['false_safe_rate']:.1f}% 降到 {100 * guarded['false_safe_rate']:.1f}%。",
        fill=white,
        font=load_font(25),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
