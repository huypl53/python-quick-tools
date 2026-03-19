"""
Merge overlapping class boxes in a YOLO dataset by absorbing smaller boxes
that are mostly contained within larger boxes.

When a smaller box's area is mostly inside a larger box (above threshold),
the smaller box is dropped and the larger box is kept.

Usage examples:
  # Drop any box that is >=90% contained inside a larger box
  python merge_overlapping.py /path/to/src /path/to/dest --threshold 0.9

  # Only absorb specific classes into others (e.g., drop "text" inside "label")
  python merge_overlapping.py /path/to/src /path/to/dest --absorb text --into label
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Absorb smaller boxes that are mostly inside larger boxes."
    )
    parser.add_argument("source", type=Path, help="YOLO dataset root (with data.yaml).")
    parser.add_argument("dest", type=Path, help="Output dataset root.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Min fraction of smaller box area inside the larger box to absorb. Default: 0.9",
    )
    parser.add_argument(
        "--absorb",
        nargs="+",
        help="Only absorb boxes of these class names. If omitted, any class can be absorbed.",
    )
    parser.add_argument(
        "--into",
        nargs="+",
        help="Only absorb into boxes of these class names. If omitted, any larger box absorbs.",
    )
    parser.add_argument(
        "--drop-empty",
        action="store_true",
        help="Skip images where all annotations were removed (default: keep as negative examples).",
    )
    return parser.parse_args()


def load_names(data_yaml: Path) -> List[str]:
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("names", [])


def name_to_ids(names: List[str], selection: Optional[List[str]]) -> Optional[Set[int]]:
    if selection is None:
        return None
    name_map = {n: i for i, n in enumerate(names)}
    missing = [n for n in selection if n not in name_map]
    if missing:
        raise ValueError(f"Classes not found in data.yaml: {', '.join(missing)}")
    return {name_map[n] for n in selection}


Box = Tuple[float, float, float, float]


def yolo_to_corners(x: float, y: float, w: float, h: float) -> Box:
    hw, hh = w / 2.0, h / 2.0
    return (x - hw, y - hh, x + hw, y + hh)


def box_area(b: Sequence[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def parse_label_line(line: str) -> Optional[Tuple[int, float, float, float, float, List[str]]]:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        cid = int(parts[0])
        x, y, w, h = map(float, parts[1:5])
    except ValueError:
        return None
    return cid, x, y, w, h, parts[5:]


def find_image(images_dir: Path, stem: str) -> Optional[Path]:
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def merge_boxes(
    entries: List[Tuple[int, float, float, float, float, List[str]]],
    threshold: float,
    absorb_ids: Optional[Set[int]],
    into_ids: Optional[Set[int]],
) -> Tuple[List[Tuple[int, float, float, float, float, List[str]]], int]:
    """Return (kept_entries, absorbed_count)."""
    n = len(entries)
    corners = [yolo_to_corners(e[1], e[2], e[3], e[4]) for e in entries]
    areas = [box_area(c) for c in corners]
    absorbed: Set[int] = set()

    for i in range(n):
        if i in absorbed:
            continue
        for j in range(n):
            if i == j or j in absorbed:
                continue
            # i is the potential absorber (larger), j is the candidate to absorb (smaller)
            if areas[i] <= areas[j]:
                continue
            # Check class filters
            if into_ids is not None and entries[i][0] not in into_ids:
                continue
            if absorb_ids is not None and entries[j][0] not in absorb_ids:
                continue
            # Check containment
            inter = intersection_area(corners[i], corners[j])
            if areas[j] > 0 and inter / areas[j] >= threshold:
                absorbed.add(j)

    kept = [e for idx, e in enumerate(entries) if idx not in absorbed]
    return kept, len(absorbed)


def format_line(entry: Tuple[int, float, float, float, float, List[str]]) -> str:
    cid, x, y, w, h, extras = entry
    tokens = [str(cid), str(x), str(y), str(w), str(h)] + extras
    return " ".join(tokens)


def copy_metadata(src: Path, dest: Path) -> None:
    for name in ["data.yaml", "README.dataset.txt", "README.roboflow.txt"]:
        candidate = src / name
        if candidate.exists():
            shutil.copy2(candidate, dest / name)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    dest = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    names = load_names(source / "data.yaml")
    absorb_ids = name_to_ids(names, args.absorb)
    into_ids = name_to_ids(names, args.into)

    total_files = 0
    copied_files = 0
    total_absorbed = 0

    for dirpath, _, filenames in os.walk(source):
        if Path(dirpath).name != "labels":
            continue
        labels_dir = Path(dirpath)
        images_dir = labels_dir.parent / "images"

        for filename in filenames:
            if not filename.endswith(".txt"):
                continue
            total_files += 1
            label_path = labels_dir / filename

            entries: List[Tuple[int, float, float, float, float, List[str]]] = []
            with label_path.open("r", encoding="utf-8") as f:
                for raw in f:
                    parsed = parse_label_line(raw)
                    if parsed:
                        entries.append(parsed)

            kept, absorbed_count = merge_boxes(
                entries, args.threshold, absorb_ids, into_ids
            )
            total_absorbed += absorbed_count

            if not kept and args.drop_empty:
                continue

            stem = Path(filename).stem
            image_path = find_image(images_dir, stem)
            if not image_path:
                print(f"Warning: image not found for {label_path}")
                continue

            rel_label_dir = labels_dir.relative_to(source)
            dest_label_dir = dest / rel_label_dir
            dest_label_dir.mkdir(parents=True, exist_ok=True)
            dest_label_path = dest_label_dir / filename
            with dest_label_path.open("w", encoding="utf-8") as out:
                if kept:
                    out.write("\n".join(format_line(e) for e in kept) + "\n")

            dest_images_dir = dest / images_dir.relative_to(source)
            dest_images_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, dest_images_dir / image_path.name)
            copied_files += 1

    copy_metadata(source, dest)

    print(f"Processed label files: {total_files}")
    print(f"Copied files: {copied_files}")
    print(f"Boxes absorbed: {total_absorbed}")
    print(f"Output: {dest}")


if __name__ == "__main__":
    main()
