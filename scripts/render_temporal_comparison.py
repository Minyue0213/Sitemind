"""Create a side-by-side GIF of single-frame and temporal route decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FONT_CANDIDATES = (
    Path("/System/Library/Fonts/STHeiti Light.ttc"),
    Path("/System/Library/Fonts/PingFang.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
)


def font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.truetype("DejaVuSans.ttf", size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("single_frame_rendered", type=Path)
    parser.add_argument("temporal_rendered", type=Path)
    parser.add_argument("sequence_metadata", type=Path)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/alice_seq02_temporal_comparison.gif")
    )
    args = parser.parse_args()
    metadata = json.loads(args.sequence_metadata.read_text(encoding="utf-8"))
    frames = metadata["frames"]
    first_timestamp = int(frames[0]["camera_timestamp_ns"])
    rendered: list[Image.Image] = []
    route_panel_box = (1260, 760, 1840, 1382)

    for index, item in enumerate(frames):
        frame_name = item["frame_name"]
        before_path = args.single_frame_rendered / f"{frame_name}.png"
        after_path = args.temporal_rendered / f"{frame_name}.png"
        before = Image.open(before_path).convert("RGB").crop(route_panel_box)
        after = Image.open(after_path).convert("RGB").crop(route_panel_box)
        canvas = Image.new("RGB", (1200, 720), "#081019")
        canvas.paste(before, (10, 62))
        canvas.paste(after, (610, 62))
        draw = ImageDraw.Draw(canvas)
        draw.text((18, 13), "优化前：只看当前帧", fill="#ffffff", font=font(28))
        draw.text((618, 13), "优化后：最近 7 帧记忆（约 1.4 秒）", fill="#ffffff", font=font(28))
        elapsed = (int(item["camera_timestamp_ns"]) - first_timestamp) / 1e9
        draw.text(
            (18, 690),
            f"同一起点终点 · 连续帧 {index + 1:02d}/{len(frames):02d} · t={elapsed:04.1f}s",
            fill="#d8e4ec",
            font=font(18),
        )
        rendered.append(canvas.quantize(colors=128, method=Image.Quantize.MEDIANCUT))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered[0].save(
        args.output,
        save_all=True,
        append_images=rendered[1:],
        duration=round(1000 / args.fps),
        loop=0,
        disposal=2,
        optimize=False,
    )
    print(f"output={args.output} frames={len(rendered)} fps={args.fps}")


if __name__ == "__main__":
    main()
