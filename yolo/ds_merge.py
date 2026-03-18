"""
Merge/trim a YOLO dataset by keeping only target boxes that mostly lie inside
reference boxes (class id == ref-class) from another annotation set.

Example:
python yolo_ds_merge.py \\
  /path/to/target_dataset \\
  /path/to/reference_labels \\
  /path/to/output_dataset \\
  --ref-class 0 --threshold 0.9
"""

import argparse
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep target YOLO boxes whose area is mostly inside reference boxes."
    )
    parser.add_argument("target", type=Path, help="YOLO dataset root to be filtered.")
    parser.add_argument(
        "reference",
        type=Path,
        help="Directory containing reference label .txt files (stems must match target images).",
    )
    parser.add_argument(
        "dest", type=Path, help="Output directory for the merged dataset."
    )
    parser.add_argument(
        "--ref-class",
        type=int,
        default=0,
        help="Reference class id that defines the keep regions. Default: 0.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Minimum fraction of target box area that must lie inside a reference box. Default: 0.9",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Copy images even if all boxes were removed (writes empty label files).",
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


def within_reference(
    target_box: Sequence[float], refs: Iterable[Sequence[float]], threshold: float
) -> bool:
    target_area = box_area(target_box)
    if target_area <= 0:
        return False
    required = threshold * target_area
    for ref_box in refs:
        if intersection_area(target_box, ref_box) >= required:
            return True
    return False


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
) -> Tuple[List[Tuple[float, float, float, float]], List[Tuple[float, float, float, float]]]:
    keep_boxes: List[Tuple[float, float, float, float]] = []
    other_boxes: List[Tuple[float, float, float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_label_line(line)
            if not parsed:
                continue
            class_id, x, y, w, h, _ = parsed
            if class_id == ref_class:
                keep_boxes.append(yolo_to_corners(x, y, w, h))
            else:
                other_boxes.append(yolo_to_corners(x, y, w, h))
    return keep_boxes, other_boxes


def canonical_stem(stem: str) -> str:
    """
    Normalize target label stems to match reference label naming.

    Examples:
    - abc_png_jpg.rf.123 -> abc
    - abc_jpg.rf.123 -> abc
    - abc.rf.123 -> abc
    """
    if ".rf." in stem:
        stem = stem.split(".rf.", 1)[0]
    # Strip known conversion suffixes
    stem = re.sub(r"(_png)?_jpg$", "", stem)
    stem = re.sub(r"_png$", "", stem)
    return stem


def find_image(images_dir: Path, stem: str) -> Optional[Path]:
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def mask_image_keep_regions(src: Path, dest: Path, ref_boxes: List[Tuple[float, float, float, float]]) -> None:
    """Copy image keeping only regions inside ref_boxes; others are zeroed."""
    with Image.open(src) as img:
        width, height = img.size
        base = Image.new(img.mode, img.size)
        for box in ref_boxes:
            px_box = to_pixel_box(box, width, height)
            if not px_box:
                continue
            crop = img.crop(px_box)
            base.paste(crop, px_box)
        base.save(dest)


def sample_background_color(img: Image.Image, sample: int = 10) -> Tuple[int, ...]:
    """Estimate background by sampling image borders and taking most common color."""
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")
    width, height = img.size
    s = max(1, min(sample, width // 2, height // 2))
    pixels = []
    pixels.extend(list(img.crop((0, 0, width, s)).getdata()))
    pixels.extend(list(img.crop((0, height - s, width, height)).getdata()))
    pixels.extend(list(img.crop((0, 0, s, height)).getdata()))
    pixels.extend(list(img.crop((width - s, 0, width, height)).getdata()))
    counts = Counter(pixels)
    return counts.most_common(1)[0][0]


def fill_removed_regions(
    src: Path, dest: Path, removed_boxes: List[Tuple[float, float, float, float]]
) -> None:
    """Fill removed regions with background color and save."""
    if not removed_boxes:
        shutil.copy2(src, dest)
        return
    with Image.open(src) as img:
        width, height = img.size
        bg = sample_background_color(img)
        draw = ImageDraw.Draw(img)
        for box in removed_boxes:
            px_box = to_pixel_box(box, width, height)
            if not px_box:
                continue
            draw.rectangle(px_box, fill=bg)
        img.save(dest)


def copy_metadata(src_root: Path, dest_root: Path) -> None:
    for name in ["data.yaml", "README.dataset.txt", "README.roboflow.txt"]:
        candidate = src_root / name
        if candidate.exists():
            shutil.copy2(candidate, dest_root / name)


def process_labels(
    target_root: Path,
    ref_root: Path,
    dest_root: Path,
    ref_class: int,
    threshold: float,
    keep_empty: bool,
) -> None:
    processed = 0
    kept_files = 0
    dropped_no_ref = 0
    dropped_empty = 0
    dropped_no_refs = 0
    filtered_boxes = 0

    for dirpath, _, filenames in os.walk(target_root):
        if Path(dirpath).name != "labels":
            continue
        labels_dir = Path(dirpath)
        images_dir = labels_dir.parent / "images"

        for filename in filenames:
            if not filename.endswith(".txt"):
                continue
            processed += 1
            label_path = labels_dir / filename
            raw_stem = Path(filename).stem
            stem = canonical_stem(raw_stem)
            ref_label = ref_root / f"{stem}.txt"

            if not ref_label.exists():
                dropped_no_ref += 1
                print(f"[WARN] Missing reference for {label_path} (looking for {ref_label.name}), dropping.")
                continue

            ref_boxes, ref_other_boxes = load_reference_boxes(ref_label, ref_class)
            if not ref_boxes:
                dropped_no_refs += 1
                print(
                    f"[WARN] No keep boxes (class {ref_class}) in {ref_label}, dropping {label_path}."
                )
                continue

            filtered_lines: List[str] = []
            removed_here = 0
            removed_boxes: List[Tuple[float, float, float, float]] = []
            with label_path.open("r", encoding="utf-8") as f:
                for raw_line in f:
                    parsed = parse_label_line(raw_line)
                    if not parsed:
                        continue
                    class_id, x, y, w, h, extras = parsed
                    corners = yolo_to_corners(x, y, w, h)
                    if within_reference(corners, ref_boxes, threshold):
                        tokens = [
                            str(class_id),
                            str(x),
                            str(y),
                            str(w),
                            str(h),
                        ] + extras
                        filtered_lines.append(" ".join(tokens))
                    else:
                        removed_here += 1
                        filtered_boxes += 1
                        removed_boxes.append(corners)

            if not filtered_lines and not keep_empty:
                dropped_empty += 1
                continue

            # Prepare dest paths
            rel_labels_dir = labels_dir.relative_to(target_root)
            dest_label_dir = dest_root / rel_labels_dir
            dest_label_dir.mkdir(parents=True, exist_ok=True)

            dest_label_path = dest_label_dir / filename
            with dest_label_path.open("w", encoding="utf-8") as out:
                if filtered_lines:
                    out.write("\n".join(filtered_lines) + "\n")

            image_path = find_image(images_dir, raw_stem) or find_image(images_dir, stem)
            if image_path:
                dest_images_dir = dest_root / images_dir.relative_to(target_root)
                dest_images_dir.mkdir(parents=True, exist_ok=True)
                dest_image_path = dest_images_dir / image_path.name
                try:
                    # Erase both removed target boxes and non-reference-class boxes from the ref file.
                    erase_list = removed_boxes + ref_other_boxes
                    fill_removed_regions(image_path, dest_image_path, erase_list)
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[WARN] Failed to fill removed regions for {image_path} ({exc}); copying original."
                    )
                    shutil.copy2(image_path, dest_image_path)
            else:
                print(
                    f"[WARN] Image not found for {label_path}; label kept without image copy."
                )

            kept_files += 1
            if removed_here:
                print(f"[INFO] Filtered {removed_here} boxes from {label_path}")

    print("--- Summary ---")
    print(f"Processed label files: {processed}")
    print(f"Kept files: {kept_files}")
    print(f"Dropped (missing ref file): {dropped_no_ref}")
    print(f"Dropped (no keep boxes in ref): {dropped_no_refs}")
    print(f"Dropped (emptied by filtering): {dropped_empty}")
    print(f"Total boxes removed: {filtered_boxes}")
    print(f"Output: {dest_root}")


def main() -> None:
    args = parse_args()
    target_root = args.target.resolve()
    ref_root = args.reference.resolve()
    dest_root = args.dest.resolve()
    dest_root.mkdir(parents=True, exist_ok=True)

    copy_metadata(target_root, dest_root)
    process_labels(
        target_root,
        ref_root,
        dest_root,
        ref_class=args.ref_class,
        threshold=args.threshold,
        keep_empty=args.keep_empty,
    )


if __name__ == "__main__":
    main()
