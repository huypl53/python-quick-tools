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
    reclassify_to: Optional[int],
) -> Tuple[List[Tuple[int, float, float, float, float, List[str]]], int, int]:
    """Return (kept_entries, absorbed_count, reclassified_count).

    Boxes of absorb classes that are contained in into boxes are dropped.
    Remaining absorb boxes that were NOT absorbed are reclassified to the
    ``reclassify_to`` class (when set).
    """
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

    # Reclassify leftover absorb boxes to the into class
    reclassified = 0
    kept: List[Tuple[int, float, float, float, float, List[str]]] = []
    for idx, e in enumerate(entries):
        if idx in absorbed:
            continue
        if reclassify_to is not None and absorb_ids is not None and e[0] in absorb_ids:
            kept.append((reclassify_to, e[1], e[2], e[3], e[4], e[5]))
            reclassified += 1
        else:
            kept.append(e)
    return kept, len(absorbed), reclassified


def format_line(entry: Tuple[int, float, float, float, float, List[str]]) -> str:
    cid, x, y, w, h, extras = entry
    tokens = [str(cid), str(x), str(y), str(w), str(h)] + extras
    return " ".join(tokens)


def build_remap(
    names: List[str], absorb_ids: Optional[Set[int]]
) -> Tuple[Dict[int, int], List[str]]:
    """Build old_to_new ID map and new names list, removing absorbed classes."""
    if absorb_ids is None:
        return {i: i for i in range(len(names))}, list(names)
    kept_names: List[str] = []
    old_to_new: Dict[int, int] = {}
    new_id = 0
    for old_id, name in enumerate(names):
        if old_id in absorb_ids:
            continue
        old_to_new[old_id] = new_id
        kept_names.append(name)
        new_id += 1
    return old_to_new, kept_names


def remap_entry(
    entry: Tuple[int, float, float, float, float, List[str]],
    old_to_new: Dict[int, int],
) -> Tuple[int, float, float, float, float, List[str]]:
    cid, x, y, w, h, extras = entry
    return (old_to_new[cid], x, y, w, h, extras)


def write_data_yaml(source: Path, dest: Path, new_names: List[str]) -> None:
    data_yaml = source / "data.yaml"
    if data_yaml.exists():
        with data_yaml.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        data["names"] = new_names
        data["nc"] = len(new_names)
        with (dest / "data.yaml").open("w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def copy_metadata(src: Path, dest: Path) -> None:
    for name in ["README.dataset.txt", "README.roboflow.txt"]:
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

    # Determine reclassify target: first --into class
    reclassify_to: Optional[int] = None
    if args.into and into_ids:
        name_map = {n: i for i, n in enumerate(names)}
        reclassify_to = name_map[args.into[0]]

    # Build ID remap: absorbed classes are removed from the output
    old_to_new, new_names = build_remap(names, absorb_ids)

    total_files = 0
    copied_files = 0
    total_absorbed = 0
    total_reclassified = 0

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

            merged, absorbed_count, reclassified_count = merge_boxes(
                entries, args.threshold, absorb_ids, into_ids, reclassify_to
            )
            total_absorbed += absorbed_count
            total_reclassified += reclassified_count
            kept = [remap_entry(e, old_to_new) for e in merged]

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

    write_data_yaml(source, dest, new_names)
    copy_metadata(source, dest)

    print(f"Processed label files: {total_files}")
    print(f"Copied files: {copied_files}")
    print(f"Boxes absorbed (dropped): {total_absorbed}")
    print(f"Boxes reclassified to --into class: {total_reclassified}")
    print(f"Output: {dest}")


if __name__ == "__main__":
    main()
