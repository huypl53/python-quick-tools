"""Dataset statistics and heatmap utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from yolo.bg_augment_modules.geometry import yolo_to_pixel_box
from yolo.bg_augment_modules.image import validate_image
from yolo.bg_augment_modules.io import parse_label_line


def collect_counts(
    entries: Sequence[Tuple[str, Path, Path]],
    selected_ids: Iterable[int],
) -> Dict[int, int]:
    """Count occurrences of selected classes across label files."""

    selected = set(selected_ids)
    counts = {class_id: 0 for class_id in selected}
    for _, _, label_path in tqdm(entries, desc="Counting labels", unit="image"):
        try:
            with label_path.open("r", encoding="utf-8") as f:
                for line in f:
                    parsed = parse_label_line(line)
                    if not parsed:
                        continue
                    class_id, _, _, _, _ = parsed
                    if class_id in selected:
                        counts[class_id] += 1
        except OSError as e:
            print(f"[WARN] Failed to read label file {label_path}: {e}")
    return counts


def collect_heatmaps(
    entries: Sequence[Tuple[str, Path, Path]],
    selected_ids: Iterable[int],
    grid_w: int,
    grid_h: int,
) -> Dict[int, np.ndarray]:
    """Collect spatial heatmaps showing class distribution across images."""

    selected = set(selected_ids)
    heatmaps: Dict[int, np.ndarray] = {
        class_id: np.zeros((grid_h, grid_w), dtype=np.int32) for class_id in selected
    }
    for _, image_path, label_path in tqdm(entries, desc="Building heatmaps", unit="image"):
        image = cv2.imread(str(image_path))
        image = validate_image(image, str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        if width == 0 or height == 0 or grid_w == 0 or grid_h == 0:
            continue
        try:
            with label_path.open("r", encoding="utf-8") as f:
                for line in f:
                    parsed = parse_label_line(line)
                    if not parsed:
                        continue
                    class_id, x, y, w, h = parsed
                    if class_id not in selected:
                        continue
                    pixel_box = yolo_to_pixel_box((x, y, w, h), width, height)
                    if not pixel_box:
                        continue
                    x1, y1, x2, y2 = pixel_box
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    grid_x = min(grid_w - 1, max(0, int(cx / width * grid_w)))
                    grid_y = min(grid_h - 1, max(0, int(cy / height * grid_h)))
                    heatmaps[class_id][grid_y, grid_x] += 1
        except OSError as e:
            print(f"[WARN] Failed to read label file {label_path}: {e}")
    return heatmaps


def print_heatmaps(
    heatmaps: Dict[int, np.ndarray], classes: Sequence[str], name_to_idx: Dict[str, int]
) -> None:
    """Print location heatmaps for each class to stdout."""

    print("Location heatmaps (counts by grid cell):")
    for name in classes:
        class_id = name_to_idx[name]
        grid = heatmaps.get(class_id)
        if grid is None:
            continue
        print(f"  {name}:")
        for row in grid:
            print("    " + " ".join(f"{val:4d}" for val in row))


def build_stats(
    selected_ids: Sequence[int],
    class_names: Sequence[str],
    counts: Optional[Dict[int, int]],
    heatmaps: Optional[Dict[int, np.ndarray]],
    grid_w: int,
    grid_h: int,
    bucket_counts: Optional[Dict[int, int]],
    seg_count: int,
    bucket_dir: Optional[Path],
) -> Dict[str, object]:
    """Build a statistics dictionary for JSON serialization."""

    classes = [
        {"id": class_id, "name": class_names[class_id]}
        for class_id in selected_ids
        if class_id < len(class_names)
    ]
    stats: Dict[str, object] = {
        "version": 1,
        "classes": classes,
        "heatmap_grid": [grid_w, grid_h],
        "seg_bucket_count": seg_count,
    }
    if bucket_dir is not None:
        stats["bucket_dir"] = str(bucket_dir)
    if counts is not None:
        stats["dataset_counts"] = {str(cid): int(counts.get(cid, 0)) for cid in selected_ids}
    if heatmaps is not None:
        stats["heatmaps"] = {str(cid): heatmaps[cid].tolist() for cid in selected_ids}
    if bucket_counts is not None:
        stats["bucket_counts"] = {str(cid): int(bucket_counts.get(cid, 0)) for cid in selected_ids}
    return stats


def save_stats(stats: Dict[str, object], output_path: Path) -> None:
    """Save statistics to a JSON file."""

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, sort_keys=True)
    except (OSError, TypeError) as e:
        print(f"[WARN] Failed to save stats to {output_path}: {e}")
        raise


def load_stats(stats_path: Path) -> Dict[str, object]:
    """Load statistics from a JSON file."""

    try:
        with stats_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[WARN] Failed to load stats from {stats_path}: {e}")
        raise


def parse_stats_counts(
    stats: Dict[str, object], selected_ids: Sequence[int]
) -> Optional[Dict[int, int]]:
    """Parse dataset counts from a stats dictionary."""

    raw = stats.get("dataset_counts")
    if not isinstance(raw, dict):
        return None
    counts: Dict[int, int] = {}
    for class_id in selected_ids:
        value = raw.get(str(class_id), raw.get(class_id))
        if value is None:
            return None
        counts[class_id] = int(value)
    return counts


def parse_stats_heatmaps(
    stats: Dict[str, object], selected_ids: Sequence[int]
) -> Optional[Dict[int, np.ndarray]]:
    """Parse heatmaps from a stats dictionary."""

    raw = stats.get("heatmaps")
    if not isinstance(raw, dict):
        return None
    heatmaps: Dict[int, np.ndarray] = {}
    for class_id in selected_ids:
        value = raw.get(str(class_id), raw.get(class_id))
        if value is None:
            return None
        heatmaps[class_id] = np.array(value, dtype=np.int32)
    return heatmaps
