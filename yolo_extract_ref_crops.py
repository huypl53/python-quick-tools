"""
Create crops from reference boxes for images that contain target-class boxes.

For every image that contains a target class, cut out all reference boxes
(class id == ref-class) from that image. Each crop gets a label file with any
target boxes that fall inside the reference region (may be empty).

Example:
python yolo_extract_ref_crops.py \\
  /path/to/target_dataset \\
  /path/to/reference_labels \\
  /path/to/output_dataset \\
  --unpicked-dir /path/to/unpicked_out \\
  --target-classes 1 2 --ref-class 0 --threshold 0.9
"""

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For images that include target classes, crop every reference box. "
            "Crops include YOLO labels for target boxes inside each reference region."
        )
    )
    parser.add_argument(
        "target", type=Path, help="YOLO dataset root to read images/labels from."
    )
    parser.add_argument(
        "reference",
        type=Path,
        help="Directory containing reference label .txt files (stems must match target images).",
    )
    parser.add_argument(
        "dest", type=Path, help="Output directory for the cropped dataset."
    )
    parser.add_argument(
        "--target-classes",
        type=int,
        nargs="+",
        required=True,
        help="One or more target class ids to look for inside reference boxes.",
    )
    parser.add_argument(
        "--ref-class",
        type=int,
        default=0,
        help="Reference class id that defines the crop regions. Default: 0.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Minimum fraction of target box area that must lie inside a reference box. Default: 0.9.",
    )
    parser.add_argument(
        "--unpicked-dir",
        type=Path,
        default=None,
        help="Optional directory to copy images that do not produce any crops.",
    )
    return parser.parse_args()


def yolo_to_corners(
    x: float, y: float, w: float, h: float
) -> Tuple[float, float, float, float]:
    half_w = w / 2.0
    half_h = h / 2.0
    return x - half_w, y - half_h, x + half_w, y + half_h


def to_pixel_box(
    box: Sequence[float], width: int, height: int
) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = box
    px1 = max(0, min(width, int(round(x1 * width))))
    py1 = max(0, min(height, int(round(y1 * height))))
    px2 = max(0, min(width, int(round(x2 * width))))
    py2 = max(0, min(height, int(round(y2 * height))))
    if px2 <= px1 or py2 <= py1:
        return None
    return px1, py1, px2, py2


def box_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    x_left = max(ax1, bx1)
    y_top = max(ay1, by1)
    x_right = min(ax2, bx2)
    y_bottom = min(ay2, by2)
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    return (x_right - x_left) * (y_bottom - y_top)


def contains_with_threshold(
    target_box: Sequence[float], ref_box: Sequence[float], threshold: float
) -> bool:
    area = box_area(target_box)
    if area <= 0:
        return False
    return intersection_area(target_box, ref_box) >= threshold * area


def normalize_to_ref(
    target_box: Sequence[float], ref_box: Sequence[float]
) -> Optional[Tuple[float, float, float, float]]:
    """Convert a global normalized box to normalized coords within ref_box."""
    tx1, ty1, tx2, ty2 = target_box
    rx1, ry1, rx2, ry2 = ref_box
    ref_w = rx2 - rx1
    ref_h = ry2 - ry1
    if ref_w <= 0 or ref_h <= 0:
        return None
    cx1 = max(tx1, rx1)
    cy1 = max(ty1, ry1)
    cx2 = min(tx2, rx2)
    cy2 = min(ty2, ry2)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    new_x = ((cx1 + cx2) / 2.0 - rx1) / ref_w
    new_y = ((cy1 + cy2) / 2.0 - ry1) / ref_h
    new_w = (cx2 - cx1) / ref_w
    new_h = (cy2 - cy1) / ref_h
    if new_w <= 0 or new_h <= 0:
        return None
    return new_x, new_y, new_w, new_h


def parse_label_line(
    line: str,
) -> Optional[Tuple[int, float, float, float, float, List[str]]]:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        class_id = int(parts[0])
        x, y, w, h = map(float, parts[1:5])
    except ValueError:
        return None
    extras = parts[5:]
    return class_id, x, y, w, h, extras


def load_reference_boxes(
    path: Path, ref_class: int
) -> List[Tuple[float, float, float, float]]:
    boxes: List[Tuple[float, float, float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_label_line(line)
            if not parsed:
                continue
            class_id, x, y, w, h, _ = parsed
            if class_id == ref_class:
                boxes.append(yolo_to_corners(x, y, w, h))
    return boxes


def canonical_stem(stem: str) -> str:
    """
    Normalize target label stems to match reference label naming.
    """
    if ".rf." in stem:
        stem = stem.split(".rf.", 1)[0]
    stem = re.sub(r"(_png)?_jpg$", "", stem)
    stem = re.sub(r"_png$", "", stem)
    return stem


def find_image(images_dir: Path, stem: str) -> Optional[Path]:
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_metadata(src_root: Path, dest_root: Path) -> None:
    for name in ["data.yaml", "README.dataset.txt", "README.roboflow.txt"]:
        candidate = src_root / name
        if candidate.exists():
            ensure_dir(dest_root)
            dest = dest_root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(candidate.read_bytes())


def copy_unpicked_image(
    image_path: Path,
    images_dir: Path,
    target_root: Path,
    unpicked_root: Path,
) -> None:
    rel_images_dir = images_dir.relative_to(target_root)
    dest_image_dir = unpicked_root / rel_images_dir
    ensure_dir(dest_image_dir)
    shutil.copy2(image_path, dest_image_dir / image_path.name)


def crop_reference_boxes(
    target_root: Path,
    ref_root: Path,
    dest_root: Path,
    target_classes: List[int],
    ref_class: int,
    threshold: float,
    unpicked_root: Optional[Path],
) -> Dict[str, int]:
    stats: Dict[str, int] = {
        "processed_labels": 0,
        "missing_ref": 0,
        "missing_image": 0,
        "skipped_no_ref_boxes": 0,
        "skipped_no_target_boxes": 0,
        "crops_written": 0,
        "unpicked_copied": 0,
    }

    found_labels_dir = False

    for dirpath, _, filenames in os.walk(target_root):
        if Path(dirpath).name != "labels":
            continue
        found_labels_dir = True
        labels_dir = Path(dirpath)
        images_dir = labels_dir.parent / "images"

        for filename in filenames:
            if not filename.endswith(".txt"):
                continue
            stats["processed_labels"] += 1
            label_path = labels_dir / filename
            raw_stem = Path(filename).stem
            stem = canonical_stem(raw_stem)
            ref_label = ref_root / f"{stem}.txt"
            if not ref_label.exists():
                stats["missing_ref"] += 1
                continue

            ref_boxes = load_reference_boxes(ref_label, ref_class)
            if not ref_boxes:
                stats["skipped_no_ref_boxes"] += 1
                continue

            target_boxes: List[
                Tuple[int, Tuple[float, float, float, float], List[str]]
            ] = []
            with label_path.open("r", encoding="utf-8") as f:
                for raw_line in f:
                    parsed = parse_label_line(raw_line)
                    if not parsed:
                        continue
                    class_id, x, y, w, h, extras = parsed
                    if class_id not in target_classes:
                        continue
                    target_boxes.append((class_id, yolo_to_corners(x, y, w, h), extras))

            if not target_boxes:
                stats["skipped_no_target_boxes"] += 1
                image_path = find_image(images_dir, raw_stem) or find_image(
                    images_dir, stem
                )
                if not image_path:
                    stats["missing_image"] += 1
                    continue
                if unpicked_root:
                    copy_unpicked_image(
                        image_path, images_dir, target_root, unpicked_root
                    )
                    stats["unpicked_copied"] += 1
                continue

            image_path = find_image(images_dir, raw_stem) or find_image(images_dir, stem)
            if not image_path:
                stats["missing_image"] += 1
                continue

            with Image.open(image_path) as img:
                width, height = img.size
                crop_index = 0
                for ref_idx, ref_box in enumerate(ref_boxes):
                    px_box = to_pixel_box(ref_box, width, height)
                    if not px_box:
                        continue
                    ref_norm_box = (
                        px_box[0] / width,
                        px_box[1] / height,
                        px_box[2] / width,
                        px_box[3] / height,
                    )

                    # Collect target boxes contained in this reference box.
                    contained: List[
                        Tuple[int, Tuple[float, float, float, float], List[str]]
                    ] = []
                    for class_id, target_box, extras in target_boxes:
                        if contains_with_threshold(target_box, ref_norm_box, threshold):
                            contained.append((class_id, target_box, extras))

                    crop = img.crop(px_box)

                    dest_label_dir = dest_root / labels_dir.relative_to(target_root)
                    ensure_dir(dest_label_dir)
                    dest_image_dir = dest_root / images_dir.relative_to(target_root)
                    ensure_dir(dest_image_dir)

                    crop_stem = f"{raw_stem}_ref{ref_idx}_crop{crop_index}"
                    dest_image_path = dest_image_dir / f"{crop_stem}{image_path.suffix}"
                    dest_label_path = dest_label_dir / f"{crop_stem}.txt"

                    new_lines: List[str] = []
                    for class_id, target_box, extras in contained:
                        normalized = normalize_to_ref(target_box, ref_norm_box)
                        if not normalized:
                            continue
                        cx, cy, w, h = normalized
                        tokens = [
                            str(class_id),
                            f"{cx}",
                            f"{cy}",
                            f"{w}",
                            f"{h}",
                        ] + extras
                        new_lines.append(" ".join(tokens))

                    crop.save(dest_image_path)
                    # Write an empty label file when no target boxes fall inside the crop.
                    with dest_label_path.open("w", encoding="utf-8") as out:
                        if new_lines:
                            out.write("\n".join(new_lines) + "\n")

                    stats["crops_written"] += 1
                    crop_index += 1

    if not found_labels_dir:
        raise FileNotFoundError(f"No labels directory found under {target_root}")

    return stats


def main() -> None:
    args = parse_args()
    target_root = args.target.resolve()
    ref_root = args.reference.resolve()
    dest_root = args.dest.resolve()
    ensure_dir(dest_root)

    copy_metadata(target_root, dest_root)
    stats = crop_reference_boxes(
        target_root=target_root,
        ref_root=ref_root,
        dest_root=dest_root,
        target_classes=args.target_classes,
        ref_class=args.ref_class,
        threshold=args.threshold,
        unpicked_root=args.unpicked_dir.resolve() if args.unpicked_dir else None,
    )

    print("--- Summary ---")
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').capitalize()}: {value}")
    print(f"Output: {dest_root}")


if __name__ == "__main__":
    main()
