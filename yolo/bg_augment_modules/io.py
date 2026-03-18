"""Dataset IO helpers for YOLO background augmentation."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import yaml

from yolo.bg_augment_modules.constants import IMAGE_EXTS
from yolo.bg_augment_modules.geometry import pixel_to_yolo_box


def load_class_names(data_yaml: Path) -> List[str]:
    """Load class names from a YOLO data.yaml file."""

    try:
        with data_yaml.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        raise OSError(f"Failed to load {data_yaml}: {e}") from e
    if data is None:
        raise ValueError(f"data.yaml is empty: {data_yaml}")
    names = data.get("names", [])
    if not isinstance(names, list):
        raise ValueError("data.yaml must include a list 'names'.")
    return names


def copy_metadata(src_root: Path, dest_root: Path) -> None:
    """Copy metadata files from source to destination."""

    try:
        dest_root.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"[WARN] Failed to create directory {dest_root}: {e}")
        return
    for name in ["data.yaml", "README.dataset.txt", "README.roboflow.txt"]:
        candidate = src_root / name
        if candidate.exists():
            try:
                shutil.copy2(candidate, dest_root / name)
            except OSError as e:
                print(f"[WARN] Failed to copy {candidate}: {e}")


def ensure_dir(path: Path) -> bool:
    """Ensure a directory exists, creating it if necessary."""

    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except OSError as e:
        print(f"[WARN] Failed to create directory {path}: {e}")
        return False


def sanitize_name(name: str) -> str:
    """Sanitize a string to be used as a filename."""

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return cleaned or "class"


def list_image_files(folder: Path) -> List[Path]:
    """List all image files in a folder."""

    try:
        return sorted(
            [
                path
                for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTS
            ]
        )
    except OSError as e:
        print(f"[WARN] Failed to list files in {folder}: {e}")
        return []


def find_image(images_dir: Path, stem: str) -> Optional[Path]:
    """Find an image file by stem name, trying all supported extensions."""

    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def iter_label_images(
    source_root: Path, splits: Sequence[str]
) -> Iterable[Tuple[str, Path, Path]]:
    """Iterate over all label/image pairs in the specified dataset splits."""

    for split in splits:
        labels_dir = source_root / split / "labels"
        images_dir = source_root / split / "images"
        if not labels_dir.exists() or not images_dir.exists():
            print(f"[WARN] Skipping split '{split}' (missing labels or images).")
            continue
        for filename in os.listdir(labels_dir):
            if not filename.endswith(".txt"):
                continue
            label_path = labels_dir / filename
            stem = Path(filename).stem
            image_path = find_image(images_dir, stem)
            if not image_path:
                print(f"[WARN] No image for {label_path}")
                continue
            yield split, image_path, label_path


def parse_label_line(line: str) -> Optional[Tuple[int, float, float, float, float]]:
    """Parse a single line from a YOLO label file."""

    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        class_id = int(parts[0])
        x, y, w, h = map(float, parts[1:5])
    except ValueError:
        return None
    return class_id, x, y, w, h


def write_label_file(
    path: Path, labels: List[Tuple[int, int, int, int, int]], width: int, height: int
) -> bool:
    """Write YOLO format label file."""

    try:
        with path.open("w", encoding="utf-8") as f:
            for class_id, x1, y1, x2, y2 in labels:
                cx, cy, w, h = pixel_to_yolo_box((x1, y1, x2, y2), width, height)
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        return True
    except OSError as e:
        print(f"[WARN] Failed to write label file {path}: {e}")
        return False


def copy_original_dataset(
    source_root: Path, dest_root: Path, splits: Sequence[str]
) -> None:
    """Copy the original dataset to the destination directory."""

    copy_metadata(source_root, dest_root)
    for split in splits:
        for folder in ["images", "labels"]:
            src_dir = source_root / split / folder
            if not src_dir.exists():
                continue
            dest_dir = dest_root / split / folder
            ensure_dir(dest_dir)
            for dirpath, _, filenames in os.walk(src_dir):
                rel = Path(dirpath).relative_to(src_dir)
                target_dir = dest_dir / rel
                ensure_dir(target_dir)
                for filename in filenames:
                    shutil.copy2(Path(dirpath) / filename, target_dir / filename)
