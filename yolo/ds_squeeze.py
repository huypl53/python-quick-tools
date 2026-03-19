"""
Squeeze/merge YOLO classes.

Features:
- Rename all classes into a single class.
- Group sub-classes by their underscore prefix (e.g., label_side_bar -> label).

Usage examples:
  # Collapse everything to one class named "object"
  python yolo_ds_squeeze.py /path/to/src /path/to/dest --mode single --single-name object

  # Group by prefix (text before first underscore)
  python yolo_ds_squeeze.py /path/to/src /path/to/dest --mode group_prefix
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge/rename YOLO classes.")
    parser.add_argument("source", type=Path, help="YOLO dataset root (with data.yaml).")
    parser.add_argument("dest", type=Path, help="Output dataset root.")
    parser.add_argument(
        "--mode",
        choices=["single", "group_prefix"],
        required=True,
        help="single: map all classes to one. group_prefix: map by prefix before underscore.",
    )
    parser.add_argument(
        "--single-name",
        default="object",
        help="Class name to use in single mode. Default: object.",
    )
    parser.add_argument(
        "--drop-empty",
        action="store_true",
        help="Skip images where all annotations were removed (default: keep them as negative examples).",
    )
    return parser.parse_args()


def load_names(data_yaml: Path) -> List[str]:
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if not isinstance(names, list):
        raise ValueError("data.yaml: 'names' must be a list.")
    return names


def build_mapping_single(names: List[str], single_name: str) -> Tuple[Dict[int, int], List[str]]:
    mapping = {i: 0 for i in range(len(names))}
    return mapping, [single_name]


def build_mapping_group_prefix(names: List[str]) -> Tuple[Dict[int, int], List[str]]:
    prefix_to_idx: Dict[str, int] = {}
    mapping: Dict[int, int] = {}
    ordered_prefixes: List[str] = []
    for idx, name in enumerate(names):
        prefix = name.split("_", 1)[0]
        if prefix not in prefix_to_idx:
            prefix_to_idx[prefix] = len(ordered_prefixes)
            ordered_prefixes.append(prefix)
        mapping[idx] = prefix_to_idx[prefix]
    return mapping, ordered_prefixes


def parse_label_line(line: str) -> Optional[Tuple[int, List[str]]]:
    parts = line.strip().split()
    if len(parts) < 1:
        return None
    try:
        class_id = int(parts[0])
    except ValueError:
        return None
    extras = parts[1:]
    return class_id, extras


def copy_metadata(src: Path, dest: Path, new_names: List[str]) -> None:
    src_yaml = src / "data.yaml"
    if src_yaml.exists():
        with src_yaml.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        data["names"] = new_names
        data["nc"] = len(new_names)
        dest_yaml = dest / "data.yaml"
        with dest_yaml.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    for name in ["README.dataset.txt", "README.roboflow.txt"]:
        candidate = src / name
        if candidate.exists():
            shutil.copy2(candidate, dest / name)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def process_dataset(
    source: Path,
    dest: Path,
    mapping: Dict[int, int],
    drop_empty: bool,
) -> Tuple[int, int]:
    total_labels = 0
    copied = 0

    for dirpath, _, filenames in os.walk(source):
        if Path(dirpath).name != "labels":
            continue
        labels_dir = Path(dirpath)
        images_dir = labels_dir.parent / "images"
        for filename in filenames:
            if not filename.endswith(".txt"):
                continue
            total_labels += 1
            label_path = labels_dir / filename
            rel_label_dir = labels_dir.relative_to(source)
            dest_label_dir = dest / rel_label_dir
            ensure_dir(dest_label_dir)
            dest_label_path = dest_label_dir / filename

            new_lines: List[str] = []
            with label_path.open("r", encoding="utf-8") as f:
                for raw in f:
                    parsed = parse_label_line(raw)
                    if not parsed:
                        continue
                    class_id, extras = parsed
                    if class_id not in mapping:
                        continue
                    new_class = mapping[class_id]
                    if extras:
                        new_lines.append(" ".join([str(new_class)] + extras))
                    else:
                        new_lines.append(str(new_class))

            if not new_lines and drop_empty:
                continue

            with dest_label_path.open("w", encoding="utf-8") as out:
                if new_lines:
                    out.write("\n".join(new_lines) + "\n")

            # Copy matching image
            stem = Path(filename).stem
            src_img = find_image(images_dir, stem)
            if src_img:
                dest_images_dir = dest / images_dir.relative_to(source)
                ensure_dir(dest_images_dir)
                shutil.copy2(src_img, dest_images_dir / src_img.name)
            copied += 1

    return total_labels, copied


def find_image(images_dir: Path, stem: str) -> Optional[Path]:
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    dest = args.dest.resolve()
    ensure_dir(dest)

    names = load_names(source / "data.yaml")
    if args.mode == "single":
        mapping, new_names = build_mapping_single(names, args.single_name)
    else:
        mapping, new_names = build_mapping_group_prefix(names)

    copy_metadata(source, dest, new_names)
    total, copied = process_dataset(source, dest, mapping, args.drop_empty)

    print(f"Processed label files: {total}")
    print(f"Copied files: {copied}")
    print(f"New classes ({len(new_names)}): {new_names}")


if __name__ == "__main__":
    main()
