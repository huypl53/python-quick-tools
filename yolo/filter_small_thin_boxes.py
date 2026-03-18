"""
Remove small or thin bounding boxes from a YOLO dataset.

Dry-run (default) draws kept boxes and highlights dropped boxes for review.

Usage examples:
  python yolo_filter_small_thin_boxes.py /path/to/dataset --min-area 64 --min-width 4 --min-height 4
  python yolo_filter_small_thin_boxes.py /path/to/dataset --min-area 64 --max-aspect 10 --in-place
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
KEEP_COLOR = (0, 200, 0)
DROP_COLOR = (255, 60, 60)
KEEP_WIDTH = 1
DROP_WIDTH = 2
PAD=2


@dataclass
class ParsedLabel:
    raw: str
    class_id: int
    xc: float
    yc: float
    w: float
    h: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove small/thin boxes from a YOLO dataset (dry-run preview by default)."
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="YOLO dataset root containing labels/images folders.",
    )
    parser.add_argument(
        "--min-width",
        type=float,
        default=4.0,
        help="Drop boxes with pixel width below this value (disabled if 0).",
    )
    parser.add_argument(
        "--min-height",
        type=float,
        default=4.0,
        help="Drop boxes with pixel height below this value (disabled if 0).",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.0,
        help="Drop boxes with pixel area below this value (disabled if 0).",
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.0,
        help="Drop boxes with area ratio below this value (0-1, disabled if 0).",
    )
    parser.add_argument(
        "--max-aspect",
        type=float,
        default=0.0,
        help="Drop boxes whose max(width/height, height/width) exceeds this value (disabled if 0).",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--in-place",
        action="store_true",
        help="Modify label files in-place (no previews).",
    )
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview only (default).",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=None,
        help="Directory to write preview images in dry-run mode.",
    )
    return parser.parse_args()


def iter_label_files(root: Path) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(root):
        if "labels" not in Path(dirpath).parts:
            continue
        for filename in filenames:
            if filename.endswith(".txt"):
                yield Path(dirpath) / filename


def find_image_for_label(label_path: Path) -> Optional[Path]:
    parts = label_path.parts
    label_idx = None
    for idx, part in enumerate(parts):
        if part == "labels":
            label_idx = idx
    if label_idx is None:
        return None
    image_base = Path(*parts[:label_idx], "images", *parts[label_idx + 1 :])
    for ext in IMAGE_EXTS:
        candidate = image_base.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def parse_label_line(raw_line: str) -> Optional[ParsedLabel]:
    line = raw_line.strip()
    if not line:
        return None
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        class_id = int(parts[0])
        xc = float(parts[1])
        yc = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
    except ValueError:
        return None
    return ParsedLabel(raw=line, class_id=class_id, xc=xc, yc=yc, w=w, h=h)


def yolo_to_xyxy(
    xc: float, yc: float, w: float, h: float, img_w: int, img_h: int
) -> Tuple[int, int, int, int]:
    x0 = int(round((xc - w / 2.0) * img_w))
    y0 = int(round((yc - h / 2.0) * img_h))
    x1 = int(round((xc + w / 2.0) * img_w))
    y1 = int(round((yc + h / 2.0) * img_h))
    x0 = max(0, min(img_w - 1, x0))
    y0 = max(0, min(img_h - 1, y0))
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def should_drop(
    width_px: float,
    height_px: float,
    img_area: float,
    args: argparse.Namespace,
) -> bool:
    if width_px <= 0 or height_px <= 0:
        return True
    if args.min_width > 0 and width_px < args.min_width:
        return True
    if args.min_height > 0 and height_px < args.min_height:
        return True
    if args.min_area > 0 and (width_px * height_px) < args.min_area:
        return True
    if args.min_area_ratio > 0 and img_area > 0:
        if (width_px * height_px) / img_area < args.min_area_ratio:
            return True
    if args.max_aspect > 0:
        ratio = max(width_px / height_px, height_px / width_px)
        if ratio > args.max_aspect:
            return True
    return False


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def draw_preview(
    image_path: Path,
    output_path: Path,
    kept_boxes: List[Tuple[int, int, int, int]],
    dropped_boxes: List[Tuple[int, int, int, int]],
) -> None:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        kept_boxes = apply_pad(kept_boxes)
        dropped_boxes = apply_pad(dropped_boxes)
        for box in kept_boxes:
            draw.rectangle(box, outline=KEEP_COLOR, width=KEEP_WIDTH)
        for box in dropped_boxes:
            draw.rectangle(box, outline=DROP_COLOR, width=DROP_WIDTH)
        ensure_dir(output_path.parent)
        img.save(output_path)

def apply_pad(boxes: List[Tuple[int,int,int,int]], pad = PAD) -> List[Tuple[int,int,int,int]]:
    padded: List[Tuple[int,int,int,int]] = []
    for b in boxes:
        new_box = (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad)
        padded.append(new_box)
    return padded



def main() -> None:
    args = parse_args()
    dataset_root = args.dataset.resolve()

    if (
        args.min_width <= 0
        and args.min_height <= 0
        and args.min_area <= 0
        and args.min_area_ratio <= 0
        and args.max_aspect <= 0
    ):
        raise ValueError("At least one threshold must be set to drop boxes.")

    dry_run = not args.in_place
    if args.in_place and args.preview_dir is not None:
        raise ValueError("--preview-dir is only valid in dry-run mode.")

    preview_root = None
    if dry_run:
        preview_root = args.preview_dir
        if preview_root is None:
            preview_root = dataset_root / "_preview_small_thin"

    stats = {
        "label_files": 0,
        "missing_images": 0,
        "boxes_total": 0,
        "boxes_dropped": 0,
        "labels_modified": 0,
        "previews_written": 0,
        "invalid_lines": 0,
    }

    found_labels = False
    for label_path in iter_label_files(dataset_root):
        found_labels = True
        stats["label_files"] += 1
        image_path = find_image_for_label(label_path)
        if image_path is None or not image_path.exists():
            stats["missing_images"] += 1
            print(f"[WARN] Missing image for {label_path}")
            continue

        with Image.open(image_path) as img:
            img_w, img_h = img.size

        img_area = float(img_w * img_h)
        kept_lines: List[str] = []
        kept_boxes: List[Tuple[int, int, int, int]] = []
        dropped_boxes: List[Tuple[int, int, int, int]] = []
        dropped_count = 0

        with label_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                parsed = parse_label_line(raw_line)
                if parsed is None:
                    if raw_line.strip():
                        stats["invalid_lines"] += 1
                        kept_lines.append(raw_line.strip())
                    continue
                stats["boxes_total"] += 1
                width_px = parsed.w * img_w
                height_px = parsed.h * img_h
                if should_drop(width_px, height_px, img_area, args):
                    dropped_count += 1
                    dropped_boxes.append(
                        yolo_to_xyxy(parsed.xc, parsed.yc, parsed.w, parsed.h, img_w, img_h)
                    )
                else:
                    kept_lines.append(parsed.raw)
                    kept_boxes.append(
                        yolo_to_xyxy(parsed.xc, parsed.yc, parsed.w, parsed.h, img_w, img_h)
                    )

        if dropped_count == 0:
            continue

        stats["boxes_dropped"] += dropped_count

        if dry_run and preview_root is not None:
            try:
                rel_image = image_path.relative_to(dataset_root)
                output_path = preview_root / rel_image
            except ValueError:
                output_path = preview_root / image_path.name
            draw_preview(image_path, output_path, kept_boxes, dropped_boxes)
            stats["previews_written"] += 1
        elif args.in_place:
            with label_path.open("w", encoding="utf-8") as f:
                if kept_lines:
                    f.write("\n".join(kept_lines) + "\n")
            stats["labels_modified"] += 1

    if not found_labels:
        raise FileNotFoundError(f"No labels directory found under {dataset_root}")

    print(f"Label files processed: {stats['label_files']}")
    print(f"Missing images: {stats['missing_images']}")
    print(f"Total boxes: {stats['boxes_total']}")
    print(f"Boxes dropped: {stats['boxes_dropped']}")
    print(f"Invalid lines kept: {stats['invalid_lines']}")
    if dry_run:
        print(f"Previews written: {stats['previews_written']}")
        if preview_root is not None:
            print(f"Preview dir: {preview_root}")
    else:
        print(f"Label files modified: {stats['labels_modified']}")


if __name__ == "__main__":
    main()
