"""
Filter YOLO datasets by dropping specific label names.

Usage example:
python yolo_filter_label.py /path/to/input_dataset /path/to/output_dataset --drop label_a label_b
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iterate over a YOLO dataset and remove annotations for selected labels."
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Input dataset root containing data.yaml and images/labels folders.",
    )
    parser.add_argument(
        "dest",
        type=Path,
        help="Output directory for the filtered dataset.",
    )
    parser.add_argument(
        "--drop",
        nargs="+",
        required=True,
        help="List of label names to remove (must exist in data.yaml).",
    )
    parser.add_argument(
        "--drop-empty",
        action="store_true",
        help="Skip images where all annotations were removed (default: keep them as negative examples).",
    )
    return parser.parse_args()


def load_label_indices(
    data_yaml: Path, drop_names: Iterable[str]
) -> Tuple[Set[int], Dict[int, int], List[str]]:
    """Return (drop_indices, old_to_new_id_map, new_names_list)."""
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names: List[str] = data.get("names", [])
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    missing = [name for name in drop_names if name not in name_to_idx]
    if missing:
        raise ValueError(f"Labels not found in data.yaml: {', '.join(missing)}")
    drop_indices = {name_to_idx[name] for name in drop_names}
    kept_names = [n for i, n in enumerate(names) if i not in drop_indices]
    old_to_new: Dict[int, int] = {}
    new_id = 0
    for old_id, name in enumerate(names):
        if old_id not in drop_indices:
            old_to_new[old_id] = new_id
            new_id += 1
    return drop_indices, old_to_new, kept_names


def find_image_path(images_dir: Path, stem: str) -> Optional[Path]:
    """Return the image path matching the given stem in images_dir."""
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def filter_label_file(
    label_path: Path, drop_indices: Set[int], old_to_new: Dict[int, int]
) -> Tuple[List[str], int]:
    filtered: List[str] = []
    drop_count = 0
    with label_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                class_id = int(parts[0])
            except (ValueError, IndexError):
                continue
            if class_id not in drop_indices:
                parts[0] = str(old_to_new[class_id])
                filtered.append(" ".join(parts))
            else:
                drop_count += 1
    return filtered, drop_count


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_updated_data_yaml(source: Path, dest: Path, new_names: List[str]) -> None:
    data_yaml = source / "data.yaml"
    if data_yaml.exists():
        with data_yaml.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        data["names"] = new_names
        data["nc"] = len(new_names)
        with (dest / "data.yaml").open("w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def copy_metadata(source: Path, dest: Path) -> None:
    for filename in ["README.dataset.txt", "README.roboflow.txt"]:
        candidate = source / filename
        if candidate.exists():
            shutil.copy2(candidate, dest / filename)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    dest = args.dest.resolve()
    ensure_dir(dest)

    drop_indices, old_to_new, new_names = load_label_indices(
        source / "data.yaml", args.drop
    )

    total_files = 0
    copied_files = 0
    removed_labels = 0

    for dirpath, dirnames, filenames in os.walk(source):
        if Path(dirpath).name != "labels":
            continue
        labels_dir = Path(dirpath)
        images_dir = labels_dir.parent / "images"
        for filename in filenames:
            if not filename.endswith(".txt"):
                continue
            total_files += 1
            label_path = labels_dir / filename
            filtered_lines, removed_count = filter_label_file(
                label_path, drop_indices, old_to_new
            )
            removed_labels += removed_count
            image_stem = Path(filename).stem
            image_path = find_image_path(images_dir, image_stem)
            if not filtered_lines and args.drop_empty:
                continue

            if not image_path:
                print(f"Warning: could not find image for {label_path}")
                continue

            rel_dir = labels_dir.relative_to(source)
            dest_label_dir = dest / rel_dir
            ensure_dir(dest_label_dir)

            dest_label_path = dest_label_dir / filename
            with dest_label_path.open("w", encoding="utf-8") as f:
                if filtered_lines:
                    f.write("\n".join(filtered_lines) + "\n")

            dest_image_dir = dest / images_dir.relative_to(source)
            ensure_dir(dest_image_dir)
            shutil.copy2(image_path, dest_image_dir / image_path.name)
            copied_files += 1

    write_updated_data_yaml(source, dest, new_names)
    copy_metadata(source, dest)

    print(f"Processed label files: {total_files}")
    print(f"Copied files (with labels kept/empty): {copied_files}")
    print(f"Annotations removed: {removed_labels}")
    print(f"Output written to: {dest}")


if __name__ == "__main__":
    main()
