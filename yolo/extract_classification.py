"""
Extract YOLO bounding boxes into a classification-ready folder structure.

For each bbox in the dataset, crops the corresponding region and saves it under
output/<split>/<class_name>/, using the source image extension.

Example:
python yolo_extract_classification.py /path/to/yolo_dataset /path/to/output
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract YOLO bounding boxes into class-specific folders for classification."
    )
    parser.add_argument("source", type=Path, help="YOLO dataset root containing data.yaml.")
    parser.add_argument("dest", type=Path, help="Output root directory.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        help="Dataset splits to process (must contain images/labels). Default: train valid test.",
    )
    return parser.parse_args()


def load_class_names(data_yaml: Path) -> List[str]:
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if not isinstance(names, list):
        raise ValueError("data.yaml must include a list 'names'.")
    return names


def find_image(images_dir: Path, stem: str) -> Optional[Path]:
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def yolo_to_pixel_box(box: Sequence[float], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    x, y, w, h = box
    x1 = (x - w / 2) * width
    y1 = (y - h / 2) * height
    x2 = (x + w / 2) * width
    y2 = (y + h / 2) * height
    px1 = max(0, min(width, int(round(x1))))
    py1 = max(0, min(height, int(round(y1))))
    px2 = max(0, min(width, int(round(x2))))
    py2 = max(0, min(height, int(round(y2))))
    if px2 <= px1 or py2 <= py1:
        return None
    return px1, py1, px2, py2


def parse_label_line(line: str) -> Optional[Tuple[int, float, float, float, float]]:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        class_id = int(parts[0])
        x, y, w, h = map(float, parts[1:5])
    except ValueError:
        return None
    return class_id, x, y, w, h


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def process_split(
    split: str,
    source_root: Path,
    dest_root: Path,
    names: List[str],
) -> Tuple[int, int]:
    labels_dir = source_root / split / "labels"
    images_dir = source_root / split / "images"
    if not labels_dir.exists() or not images_dir.exists():
        print(f"[WARN] Skipping split '{split}' (missing labels or images).")
        return 0, 0

    total_boxes = 0
    saved = 0

    for filename in os.listdir(labels_dir):
        if not filename.endswith(".txt"):
            continue
        label_path = labels_dir / filename
        image_stem = Path(filename).stem
        image_path = find_image(images_dir, image_stem)
        if not image_path:
            print(f"[WARN] No image for {label_path}")
            continue

        with Image.open(image_path) as img:
            width, height = img.size
            with label_path.open("r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    parsed = parse_label_line(line)
                    if not parsed:
                        continue
                    class_id, x, y, w, h = parsed
                    class_name = names[class_id] if 0 <= class_id < len(names) else f"class_{class_id}"
                    pixel_box = yolo_to_pixel_box((x, y, w, h), width, height)
                    if not pixel_box:
                        continue

                    crop = img.crop(pixel_box)
                    out_dir = dest_root / split / class_name
                    ensure_dir(out_dir)
                    out_name = f"{image_stem}_{idx}{image_path.suffix}"
                    crop.save(out_dir / out_name)
                    saved += 1
                    total_boxes += 1

    return total_boxes, saved


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    dest = args.dest.resolve()
    ensure_dir(dest)

    names = load_class_names(source / "data.yaml")

    grand_total = 0
    grand_saved = 0
    for split in args.splits:
        total, saved = process_split(split, source, dest, names)
        grand_total += total
        grand_saved += saved
        print(f"[{split}] boxes: {total}, saved: {saved}")

    print(f"Done. Total boxes: {grand_total}, saved crops: {grand_saved}")
    print(f"Output root: {dest}")


if __name__ == "__main__":
    main()
