"""
Convert YOLO segmentation labels (polygon/keypoints) to bounding boxes.

Usage examples:
  python yolo_seg_to_bbox.py /path/to/labels_root /path/to/output_labels
  python yolo_seg_to_bbox.py /path/to/labels_root --in-place
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert YOLO segmentation keypoints to YOLO bounding boxes."
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Root directory containing YOLO label .txt files (nested folders allowed).",
    )
    parser.add_argument(
        "dest",
        type=Path,
        nargs="?",
        default=None,
        help="Output directory for converted labels (omit when using --in-place).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite label files in the source directory.",
    )
    return parser.parse_args()


def iter_label_files(root: Path) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(".txt"):
                yield Path(dirpath) / filename


def parse_polygon(coords: List[float]) -> Tuple[float, float, float, float]:
    xs = coords[0::2]
    ys = coords[1::2]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    x_center = (min_x + max_x) / 2.0
    y_center = (min_y + max_y) / 2.0
    width = max_x - min_x
    height = max_y - min_y
    return x_center, y_center, width, height


def convert_line(line: str) -> str | None:
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        class_id = int(parts[0])
        coords = [float(value) for value in parts[1:]]
    except ValueError:
        return None
    if len(coords) == 4:
        x_center, y_center, width, height = coords
        return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    if len(coords) > 4 and len(coords) % 2 == 0:
        x_center, y_center, width, height = parse_polygon(coords)
        return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    if len(coords) > 4 and (len(coords) - 4) % 3 == 0:
        x_center, y_center, width, height = coords[:4]
        return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    return None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    if args.in_place and args.dest is not None:
        raise SystemExit("Provide either --in-place or a destination directory, not both.")
    if not args.in_place and args.dest is None:
        raise SystemExit("Destination directory is required unless using --in-place.")

    dest = source if args.in_place else args.dest.resolve()
    if not args.in_place:
        ensure_dir(dest)

    total_files = 0
    total_lines = 0
    converted_lines = 0
    skipped_lines = 0

    for label_path in iter_label_files(source):
        total_files += 1
        rel_path = label_path.relative_to(source)
        out_path = label_path if args.in_place else dest / rel_path
        ensure_dir(out_path.parent)

        output_lines: List[str] = []
        with label_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                total_lines += 1
                converted = convert_line(line)
                if converted is None:
                    skipped_lines += 1
                    continue
                output_lines.append(converted)
                converted_lines += 1

        with out_path.open("w", encoding="utf-8") as f:
            if output_lines:
                f.write("\n".join(output_lines) + "\n")

    print(f"Processed label files: {total_files}")
    print(f"Total non-empty lines: {total_lines}")
    print(f"Converted lines: {converted_lines}")
    print(f"Skipped lines: {skipped_lines}")
    print(f"Output written to: {dest}")


if __name__ == "__main__":
    main()
