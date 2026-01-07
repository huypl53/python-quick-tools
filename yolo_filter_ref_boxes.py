"""
Filter reference boxes by target classes and keep matching target annotations.

For each image/label pair in the target dataset, find reference boxes
(class id == ref-class) that contain at least one box from the provided target
classes. Keep only those target-class boxes that lie inside matched reference
boxes and drop the rest. Images are copied alongside the filtered labels.

Example:
python yolo_filter_ref_boxes.py \\
  /path/to/target_dataset \\
  /path/to/reference_labels \\
  /path/to/output_dataset \\
  --target-classes 1 2 --ref-class 0 --threshold 0.9
"""

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep target-class boxes that lie inside reference boxes which themselves contain target classes. "
            "Drops unmatched references and writes a filtered YOLO dataset."
        )
    )
    parser.add_argument("target", type=Path, help="YOLO dataset root to read images/labels from.")
    parser.add_argument(
        "reference",
        type=Path,
        help="Directory containing reference label .txt files (stems must match target images).",
    )
    parser.add_argument("dest", type=Path, help="Output directory for the filtered dataset.")
    parser.add_argument(
        "--target-classes",
        type=int,
        nargs="+",
        required=True,
        help="One or more target class ids to keep if they lie inside a matched reference box.",
    )
    parser.add_argument(
        "--ref-class",
        type=int,
        default=0,
        help="Reference class id that defines candidate regions. Default: 0.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Minimum fraction of a target box that must overlap a reference box to be considered inside. Default: 0.9.",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Copy images even if no boxes remain; writes empty label files.",
    )
    return parser.parse_args()


def yolo_to_corners(
    x: float, y: float, w: float, h: float
) -> Tuple[float, float, float, float]:
    half_w = w / 2.0
    half_h = h / 2.0
    return x - half_w, y - half_h, x + half_w, y + half_h


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


def load_reference_boxes(path: Path, ref_class: int) -> List[Tuple[float, float, float, float]]:
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


def filter_reference_boxes(
    target_root: Path,
    ref_root: Path,
    dest_root: Path,
    target_classes: List[int],
    ref_class: int,
    threshold: float,
    keep_empty: bool,
) -> Dict[str, int]:
    stats: Dict[str, int] = {
        "processed_labels": 0,
        "missing_ref": 0,
        "dropped_empty": 0,
        "missing_image": 0,
        "kept_files": 0,
        "boxes_kept": 0,
    }

    for dirpath, _, filenames in os.walk(target_root):
        if Path(dirpath).name != "labels":
            continue
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
                stats["dropped_empty"] += 1
                continue

            target_boxes: List[Tuple[int, Tuple[float, float, float, float], List[str]]] = []
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
                stats["dropped_empty"] += 1
                continue

            matched_refs = set()
            for idx, ref_box in enumerate(ref_boxes):
                for _, target_box, _ in target_boxes:
                    if contains_with_threshold(target_box, ref_box, threshold):
                        matched_refs.add(idx)
                        break

            if not matched_refs:
                stats["dropped_empty"] += 1
                continue

            new_lines: List[str] = []
            for class_id, target_box, extras in target_boxes:
                if any(
                    contains_with_threshold(target_box, ref_boxes[idx], threshold)
                    for idx in matched_refs
                ):
                    cx1, cy1, cx2, cy2 = target_box
                    cx = (cx1 + cx2) / 2.0
                    cy = (cy1 + cy2) / 2.0
                    w = cx2 - cx1
                    h = cy2 - cy1
                    tokens = [str(class_id), f"{cx}", f"{cy}", f"{w}", f"{h}"] + extras
                    new_lines.append(" ".join(tokens))

            if not new_lines and not keep_empty:
                stats["dropped_empty"] += 1
                continue

            rel_labels_dir = labels_dir.relative_to(target_root)
            dest_label_dir = dest_root / rel_labels_dir
            ensure_dir(dest_label_dir)
            dest_label_path = dest_label_dir / filename
            with dest_label_path.open("w", encoding="utf-8") as out:
                if new_lines:
                    out.write("\n".join(new_lines) + "\n")

            image_path = find_image(images_dir, raw_stem) or find_image(images_dir, stem)
            if image_path:
                rel_images_dir = images_dir.relative_to(target_root)
                dest_image_dir = dest_root / rel_images_dir
                ensure_dir(dest_image_dir)
                shutil.copy2(image_path, dest_image_dir / image_path.name)
            else:
                stats["missing_image"] += 1

            stats["kept_files"] += 1
            stats["boxes_kept"] += len(new_lines)

    return stats


def main() -> None:
    args = parse_args()
    target_root = args.target.resolve()
    ref_root = args.reference.resolve()
    dest_root = args.dest.resolve()
    ensure_dir(dest_root)

    copy_metadata(target_root, dest_root)
    stats = filter_reference_boxes(
        target_root=target_root,
        ref_root=ref_root,
        dest_root=dest_root,
        target_classes=args.target_classes,
        ref_class=args.ref_class,
        threshold=args.threshold,
        keep_empty=args.keep_empty,
    )

    print("--- Summary ---")
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').capitalize()}: {value}")
    print(f"Output: {dest_root}")


if __name__ == "__main__":
    main()
