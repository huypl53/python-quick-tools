"""
Create classification crops from YOLO bounding boxes.

For selected classes, this script:
  - crops a top-left square (size = min(box_w, box_h)) as positive
  - crops a random square inside the same bbox that does not overlap the positive

Output layout:
  output_root/<class_name>/pos/
  output_root/<class_name>/neg/

Usage example:
  python yolo_classification_crops.py /path/to/yolo_root /path/to/output --classes button table
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from PIL import Image
import yaml


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate classification crops from YOLO bounding boxes."
    )
    parser.add_argument(
        "source",
        type=Path,
        help="YOLO dataset root containing data.yaml and images/labels folders.",
    )
    parser.add_argument(
        "dest",
        type=Path,
        help="Output directory for classification crops.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Class names to include (omit to use all classes in data.yaml).",
    )
    parser.add_argument(
        "--neg-ratio",
        type=float,
        default=1.0 / 3.0,
        help="Negative crops per positive crop (default: 1/3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for negative crops.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=2,
        help="Padding in pixels to surround each crop (default: 2).",
    )
    return parser.parse_args()


def load_class_map(data_yaml: Path) -> List[str]:
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if not isinstance(names, list):
        raise ValueError("data.yaml 'names' must be a list.")
    return names


def find_image_path(images_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def iter_label_files(root: Path) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(root):
        if Path(dirpath).name != "labels":
            continue
        for filename in filenames:
            if filename.endswith(".txt"):
                yield Path(dirpath) / filename


def clamp_box(
    x0: int, y0: int, size: int, img_w: int, img_h: int
) -> Tuple[int, int, int]:
    size = max(1, size)
    if x0 < 0:
        x0 = 0
    if y0 < 0:
        y0 = 0
    if x0 + size > img_w:
        size = img_w - x0
    if y0 + size > img_h:
        size = min(size, img_h - y0)
    return x0, y0, max(0, size)


def pad_crop(
    crop: Tuple[int, int, int], padding: int, img_w: int, img_h: int
) -> Tuple[int, int, int]:
    x0, y0, size = crop
    if padding <= 0:
        return crop
    x0 -= padding
    y0 -= padding
    size += padding * 2
    return clamp_box(x0, y0, size, img_w, img_h)


def overlaps(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> bool:
    ax, ay, asz = a
    bx, by, bsz = b
    if asz <= 0 or bsz <= 0:
        return False
    return not (
        ax + asz <= bx
        or bx + bsz <= ax
        or ay + asz <= by
        or by + bsz <= ay
    )


def random_square_inside(
    bbox: Tuple[int, int, int, int],
    min_size: int,
    max_size: int,
    avoid: Tuple[int, int, int],
    rng: random.Random,
    attempts: int = 15,
) -> Optional[Tuple[int, int, int]]:
    bx0, by0, bw, bh = bbox
    max_size = min(max_size, bw, bh)
    min_size = min(min_size, max_size)
    if max_size <= 0:
        return None
    for _ in range(attempts):
        size = rng.randint(min_size, max_size)
        x0 = rng.randint(bx0, bx0 + bw - size)
        y0 = rng.randint(by0, by0 + bh - size)
        candidate = (x0, y0, size)
        if not overlaps(candidate, avoid):
            return candidate
    return None


def save_crop(
    image: Image.Image,
    crop: Tuple[int, int, int],
    out_path: Path,
) -> None:
    x0, y0, size = crop
    x1 = x0 + size
    y1 = y0 + size
    if size <= 0:
        return
    crop_img = image.crop((x0, y0, x1, y1))
    crop_img.save(out_path)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    dest = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    class_names = load_class_map(source / "data.yaml")
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    if args.classes:
        missing = [name for name in args.classes if name not in name_to_idx]
        if missing:
            raise SystemExit(f"Classes not found in data.yaml: {', '.join(missing)}")
        selected = {name_to_idx[name]: name for name in args.classes}
    else:
        selected = {idx: name for idx, name in enumerate(class_names)}

    rng = random.Random(args.seed)
    pos_count = 0
    neg_count = 0

    for label_path in iter_label_files(source):
        images_dir = label_path.parent.parent / "images"
        image_path = find_image_path(images_dir, label_path.stem)
        if not image_path:
            continue

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img_w, img_h = img.size

            with label_path.open("r", encoding="utf-8") as f:
                for line_idx, raw_line in enumerate(f):
                    line = raw_line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    try:
                        class_id = int(parts[0])
                        xc = float(parts[1])
                        yc = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                    except ValueError:
                        continue
                    if class_id not in selected:
                        continue

                    box_w = int(round(w * img_w))
                    box_h = int(round(h * img_h))
                    if box_w <= 1 or box_h <= 1:
                        continue
                    box_x0 = int(round((xc - w / 2.0) * img_w))
                    box_y0 = int(round((yc - h / 2.0) * img_h))
                    box_x0 = max(0, box_x0)
                    box_y0 = max(0, box_y0)
                    if box_x0 + box_w > img_w:
                        box_w = img_w - box_x0
                    if box_y0 + box_h > img_h:
                        box_h = img_h - box_y0
                    if box_w <= 1 or box_h <= 1:
                        continue

                    square_size = min(box_w, box_h)
                    pos_x0, pos_y0, pos_size = clamp_box(
                        box_x0, box_y0, square_size, img_w, img_h
                    )
                    if pos_size <= 1:
                        continue
                    pos_crop = (pos_x0, pos_y0, pos_size)
                    pos_crop = pad_crop(pos_crop, args.padding, img_w, img_h)
                    if pos_crop[2] <= 1:
                        continue

                    class_name = selected[class_id]
                    pos_dir = dest / class_name / "pos"
                    neg_dir = dest / class_name / "neg"
                    pos_dir.mkdir(parents=True, exist_ok=True)
                    neg_dir.mkdir(parents=True, exist_ok=True)

                    base = f"{label_path.stem}_{line_idx}"
                    pos_path = pos_dir / f"{base}.jpg"
                    save_crop(img, pos_crop, pos_path)
                    pos_count += 1

                    target_neg = int(pos_count * args.neg_ratio)
                    if neg_count < target_neg:
                        min_size = max(1, int(round(pos_size * 0.3)))
                        bbox = (box_x0, box_y0, box_w, box_h)
                        avoid = pos_crop
                        candidate = random_square_inside(
                            bbox, min_size, square_size, avoid, rng
                        )
                        if candidate is not None:
                            candidate = pad_crop(candidate, args.padding, img_w, img_h)
                            if candidate[2] <= 1 or overlaps(candidate, avoid):
                                continue
                            neg_path = neg_dir / f"{base}_neg.jpg"
                            save_crop(img, candidate, neg_path)
                            neg_count += 1

    print(f"Positive crops: {pos_count}")
    print(f"Negative crops: {neg_count}")
    print(f"Output written to: {dest}")


if __name__ == "__main__":
    main()
