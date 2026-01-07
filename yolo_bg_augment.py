"""
Augment a YOLO dataset by extracting non-background blobs and rebalancing
selected class boxes via cut-and-paste rearrangement.

Example:
python yolo_bg_augment.py /path/to/yolo_dataset /path/to/output \\
  --classes cat dog --splits train --max-aug-total 500
"""

# from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]

_WORKER_CONFIG: Optional[Dict[str, object]] = None


@dataclass
class Blob:
    image: np.ndarray
    class_id: Optional[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebalance selected YOLO classes by extracting non-background blobs, "
            "then rearranging/pasting boxes without collisions."
        )
    )
    parser.add_argument("source", type=Path, help="YOLO dataset root containing data.yaml.")
    parser.add_argument("dest", type=Path, help="Output dataset root.")
    parser.add_argument(
        "--classes",
        nargs="+",
        required=True,
        help="Selected class names to rebalance (must exist in data.yaml).",
    )
    parser.add_argument(
        "--class-weights",
        nargs="+",
        type=float,
        default=None,
        help="Optional per-class augmentation weights aligned with --classes.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        help="Splits to process (default: train).",
    )
    parser.add_argument(
        "--border-pad",
        type=int,
        default=2,
        help="Border padding in pixels used to sample background color. Default: 2.",
    )
    parser.add_argument(
        "--bg-threshold",
        type=float,
        default=30.0,
        help="Per-channel tolerance for background masking. Default: 30.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=200,
        help="Minimum area for color-segment boxes. Default: 200.",
    )
    parser.add_argument(
        "--merge-iou",
        type=float,
        default=0.2,
        help="IoU threshold to merge color-segment boxes with selected YOLO boxes. Default: 0.2.",
    )
    parser.add_argument(
        "--drop-rate",
        type=float,
        default=0.4,
        help="Chance to drop non-target blobs when composing augmented images. Default: 0.4.",
    )
    parser.add_argument(
        "--placement",
        choices=["dense", "random"],
        default="dense",
        help="Placement strategy for blobs (default: dense).",
    )
    parser.add_argument(
        "--dense-step",
        type=int,
        default=2,
        help="Pixel step for dense placement scans. Default: 2.",
    )
    parser.add_argument(
        "--fill-empty",
        action="store_true",
        default=True,
        help="Fill empty space with extra segmentation blobs.",
    )
    parser.add_argument(
        "--fill-ratio",
        type=float,
        default=0.85,
        help="Target fill ratio before stopping extra fills. Default: 0.75.",
    )
    parser.add_argument(
        "--fill-max-blobs",
        type=int,
        default=40,
        help="Max number of extra blobs to fill per image. Default: 40.",
    )
    parser.add_argument(
        "--fill-max-tries",
        type=int,
        default=100,
        help="Max placement attempts when filling empty space. Default: 200.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1)),
        help="Number of worker processes for augmentation. Default: CPU count.",
    )
    parser.add_argument(
        "--mp-start-method",
        choices=["fork", "spawn", "forkserver"],
        default="spawn",
        help="Multiprocessing start method. Default: spawn (safer with OpenCV).",
    )
    parser.add_argument(
        "--collision-pad",
        type=int,
        default=2,
        help="Padding in pixels used when checking box collisions. Default: 2.",
    )
    parser.add_argument(
        "--keep-background",
        action="store_true",
        help="Keep original background outside removed boxes (default: clear to background color).",
    )
    parser.add_argument(
        "--max-aug-per-class",
        type=int,
        default=500,
        help="Max augmented boxes to add per class. Default: 500.",
    )
    parser.add_argument(
        "--max-aug-total",
        type=int,
        default=2000,
        help="Max total augmented images to add. Default: 2000.",
    )
    parser.add_argument(
        "--max-bucket-per-class",
        type=int,
        default=500,
        help="Max stored blobs per class in the bucket. Default: 500.",
    )
    parser.add_argument(
        "--max-seg-bucket",
        type=int,
        default=500,
        help="Max stored segmentation blobs in the bucket. Default: 500.",
    )
    parser.add_argument(
        "--bucket-max-images",
        type=int,
        default=0,
        help="Limit number of images used to build the bucket (0 = no limit).",
    )
    parser.add_argument(
        "--bucket-sample-rate",
        type=float,
        default=1.0,
        help="Sample rate for bucket images (0-1]. Default: 1.0.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save debug masks with detected boxes while building the bucket.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Directory to write debug masks (default: <dest>/debug).",
    )
    parser.add_argument(
        "--debug-max",
        type=int,
        default=200,
        help="Max number of debug masks to save. Default: 200.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed. Default: 13.",
    )
    parser.add_argument(
        "--heatmap-grid",
        type=int,
        nargs=2,
        default=[5, 5],
        metavar=("GRID_W", "GRID_H"),
        help="Grid size for class heatmaps (default: 5 5).",
    )
    parser.add_argument(
        "--copy-original",
        action="store_true",
        help="Copy original dataset into the destination (default: only augmented data).",
    )
    return parser.parse_args()


def load_class_names(data_yaml: Path) -> List[str]:
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if not isinstance(names, list):
        raise ValueError("data.yaml must include a list 'names'.")
    return names


def copy_metadata(src_root: Path, dest_root: Path) -> None:
    dest_root.mkdir(parents=True, exist_ok=True)
    for name in ["data.yaml", "README.dataset.txt", "README.roboflow.txt"]:
        candidate = src_root / name
        if candidate.exists():
            shutil.copy2(candidate, dest_root / name)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_image(images_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def iter_label_images(
    source_root: Path, splits: Sequence[str]
) -> Iterable[Tuple[str, Path, Path]]:
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
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        class_id = int(parts[0])
        x, y, w, h = map(float, parts[1:5])
    except ValueError:
        return None
    return class_id, x, y, w, h


def yolo_to_pixel_box(
    box: Sequence[float], width: int, height: int
) -> Optional[Tuple[int, int, int, int]]:
    x, y, w, h = box
    x1 = (x - w / 2) * width
    y1 = (y - h / 2) * height
    x2 = (x + w / 2) * width
    y2 = (y + h / 2) * height
    px1 = max(0, min(width, int(round(x1))))
    py1 = max(0, min(height, int(round(y1))))
    px2 = max(0, min(width, int(round(x2))))
    py2 = max(0, min(height, int(round(y2))))
    if px2 <= px1 or py2 <= py1:
        return None
    return px1, py1, px2, py2


def pixel_to_yolo_box(
    box: Sequence[int], width: int, height: int
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0 / width
    cy = (y1 + y2) / 2.0 / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return cx, cy, w, h


def estimate_background_color(image: np.ndarray, pad: int) -> Tuple[int, int, int]:
    height, width = image.shape[:2]
    pad = max(1, min(pad, min(height, width) // 2))
    mask = np.zeros((height, width), dtype=bool)
    mask[:pad, :] = True
    mask[-pad:, :] = True
    mask[:, :pad] = True
    mask[:, -pad:] = True
    samples = image[mask]
    if samples.size == 0:
        return int(image[0, 0, 0]), int(image[0, 0, 1]), int(image[0, 0, 2])
    quant = (samples // 16).astype(np.int32)
    keys = (quant[:, 0] * 16 + quant[:, 1]) * 16 + quant[:, 2]
    counts = np.bincount(keys, minlength=4096)
    dominant = int(np.argmax(counts))
    target = np.array(
        [dominant // 256, (dominant // 16) % 16, dominant % 16], dtype=np.int32
    )
    match = np.all(quant == target, axis=1)
    if not np.any(match):
        mean_color = samples.mean(axis=0)
    else:
        mean_color = samples[match].mean(axis=0)
    return int(mean_color[0]), int(mean_color[1]), int(mean_color[2])


def segment_foreground_boxes(
    image: np.ndarray, bg_color: Tuple[int, int, int], threshold: float, min_area: int
) -> List[Tuple[int, int, int, int]]:
    mask = build_foreground_mask(image, bg_color, threshold)
    h, w = mask.shape
    mask = cv2.rectangle(mask, (0, 0), (w, h), 0, thickness=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(mask)
    aspect_thresh = 5.0
    border_tol = 3
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw == 0 or ch == 0:
            continue
        aspect = max(cw / float(ch), ch / float(cw))
        touches_border = (
            x <= border_tol
            or y <= border_tol
            or (x + cw) >= (w - border_tol)
            or (y + ch) >= (h - border_tol)
        )
        if touches_border and aspect >= aspect_thresh:
            continue
        cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)
    mask = filtered_mask

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Tuple[int, int, int, int]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w_cnt, h_cnt = cv2.boundingRect(contour)
        boxes.append((x, y, x + w_cnt, y + h_cnt))
    return boxes


def build_foreground_mask(
    image: np.ndarray, bg_color: Tuple[int, int, int], threshold: float
) -> np.ndarray:
    tol = int(round(threshold))
    lower = np.clip(np.array(bg_color, dtype=np.int16) - tol, 0, 255).astype(np.uint8)
    upper = np.clip(np.array(bg_color, dtype=np.int16) + tol, 0, 255).astype(np.uint8)
    bg_mask = cv2.inRange(image, lower, upper)
    return cv2.bitwise_not(bg_mask)


def iou(box_a: Sequence[int], box_b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom


def union_box(box_a: Sequence[int], box_b: Sequence[int]) -> Tuple[int, int, int, int]:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)


def merge_boxes(
    yolo_boxes: List[Tuple[int, int, int, int]],
    seg_boxes: List[Tuple[int, int, int, int]],
    merge_iou: float,
) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
    remaining_seg = seg_boxes[:]
    merged_yolo: List[Tuple[int, int, int, int]] = []
    for yolo_box in yolo_boxes:
        best_idx = None
        best_iou = 0.0
        for idx, seg_box in enumerate(remaining_seg):
            score = iou(yolo_box, seg_box)
            if score > best_iou:
                best_iou = score
                best_idx = idx
        if best_idx is not None and best_iou >= merge_iou:
            merged = union_box(yolo_box, remaining_seg[best_idx])
            merged_yolo.append(merged)
            remaining_seg.pop(best_idx)
        else:
            merged_yolo.append(yolo_box)
    return merged_yolo, remaining_seg


def rects_collide(box_a: Sequence[int], box_b: Sequence[int], pad: int = 0) -> bool:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ax1 -= pad
    ay1 -= pad
    ax2 += pad
    ay2 += pad
    bx1 -= pad
    by1 -= pad
    bx2 += pad
    by2 += pad
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def place_blob(
    canvas: np.ndarray,
    blob: Blob,
    placed: List[Tuple[int, int, int, int]],
    collision_pad: int = 0,
    max_tries: int = 50,
) -> Optional[Tuple[int, int, int, int]]:
    height, width = canvas.shape[:2]
    blob_h, blob_w = blob.image.shape[:2]
    if blob_h >= height or blob_w >= width:
        return None
    for _ in range(max_tries):
        x1 = random.randint(0, width - blob_w)
        y1 = random.randint(0, height - blob_h)
        x2 = x1 + blob_w
        y2 = y1 + blob_h
        candidate = (x1, y1, x2, y2)
        if any(rects_collide(candidate, other, collision_pad) for other in placed):
            continue
        canvas[y1:y2, x1:x2] = blob.image
        placed.append(candidate)
        return candidate
    return None


def scan_positions(limit: int, step: int) -> List[int]:
    if limit < 0:
        return []
    step = max(1, step)
    positions = list(range(0, limit + 1, step))
    if positions and positions[-1] != limit:
        positions.append(limit)
    elif not positions:
        positions = [0]
    return positions


def place_blob_dense(
    canvas: np.ndarray,
    blob: Blob,
    placed: List[Tuple[int, int, int, int]],
    collision_pad: int = 0,
    step: int = 2,
    heatmap: Optional[np.ndarray] = None,
) -> Optional[Tuple[int, int, int, int]]:
    height, width = canvas.shape[:2]
    blob_h, blob_w = blob.image.shape[:2]
    if blob_h >= height or blob_w >= width:
        return None
    xs = scan_positions(width - blob_w, step)
    ys = scan_positions(height - blob_h, step)
    best: Optional[Tuple[int, int, int, int]] = None
    best_score: Optional[int] = None
    grid_h = grid_w = 0
    if heatmap is not None:
        grid_h, grid_w = heatmap.shape
    for y in ys:
        for x in xs:
            x2 = x + blob_w
            y2 = y + blob_h
            candidate = (x, y, x2, y2)
            if any(rects_collide(candidate, other, collision_pad) for other in placed):
                continue
            if heatmap is None:
                canvas[y:y2, x:x2] = blob.image
                placed.append(candidate)
                return candidate
            cx = (x + x2) / 2.0
            cy = (y + y2) / 2.0
            grid_x = min(grid_w - 1, max(0, int(cx / width * grid_w)))
            grid_y = min(grid_h - 1, max(0, int(cy / height * grid_h)))
            score = int(heatmap[grid_y, grid_x])
            if best_score is None or score < best_score:
                best_score = score
                best = candidate
                if best_score == 0:
                    canvas[y:y2, x:x2] = blob.image
                    placed.append(candidate)
                    heatmap[grid_y, grid_x] += 1
                    return candidate
    if best is not None and heatmap is not None:
        x1, y1, x2, y2 = best
        canvas[y1:y2, x1:x2] = blob.image
        placed.append(best)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        grid_x = min(grid_w - 1, max(0, int(cx / width * grid_w)))
        grid_y = min(grid_h - 1, max(0, int(cy / height * grid_h)))
        heatmap[grid_y, grid_x] += 1
        return best
    return None


def total_area(boxes: Sequence[Tuple[int, int, int, int]]) -> int:
    area = 0
    for x1, y1, x2, y2 in boxes:
        area += max(0, x2 - x1) * max(0, y2 - y1)
    return area


def fill_empty_space(
    canvas: np.ndarray,
    placed: List[Tuple[int, int, int, int]],
    seg_bucket: List[Blob],
    collision_pad: int,
    placement: str,
    dense_step: int,
    fill_ratio: float,
    fill_max_blobs: int,
    fill_max_tries: int,
) -> None:
    if not seg_bucket:
        return
    height, width = canvas.shape[:2]
    target_ratio = max(0.0, min(1.0, fill_ratio))
    added = 0
    tries = 0
    while added < fill_max_blobs and tries < fill_max_tries:
        current_ratio = total_area(placed) / float(width * height)
        if current_ratio >= target_ratio:
            break
        blob = random.choice(seg_bucket)
        if placement == "dense":
            placed_box = place_blob_dense(
                canvas,
                blob,
                placed,
                collision_pad=collision_pad,
                step=dense_step,
                heatmap=None,
            )
        else:
            placed_box = place_blob(canvas, blob, placed, collision_pad=collision_pad)
        tries += 1
        if placed_box:
            added += 1


def place_blob_biased(
    canvas: np.ndarray,
    blob: Blob,
    placed: List[Tuple[int, int, int, int]],
    heatmap: np.ndarray,
    collision_pad: int = 0,
    max_tries: int = 50,
) -> Optional[Tuple[int, int, int, int]]:
    height, width = canvas.shape[:2]
    blob_h, blob_w = blob.image.shape[:2]
    if blob_h >= height or blob_w >= width:
        return None
    grid_h, grid_w = heatmap.shape
    candidates: List[Tuple[int, int, int, int, int]] = []
    tries = max(10, max_tries)
    for _ in range(tries):
        x1 = random.randint(0, width - blob_w)
        y1 = random.randint(0, height - blob_h)
        x2 = x1 + blob_w
        y2 = y1 + blob_h
        candidate = (x1, y1, x2, y2)
        if any(rects_collide(candidate, other, collision_pad) for other in placed):
            continue
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        grid_x = min(grid_w - 1, max(0, int(cx / width * grid_w)))
        grid_y = min(grid_h - 1, max(0, int(cy / height * grid_h)))
        score = heatmap[grid_y, grid_x]
        candidates.append((score, x1, y1, x2, y2))
    if not candidates:
        return place_blob(canvas, blob, placed, collision_pad=collision_pad, max_tries=max_tries)
    candidates.sort(key=lambda item: item[0])
    _, x1, y1, x2, y2 = candidates[0]
    canvas[y1:y2, x1:x2] = blob.image
    placed.append((x1, y1, x2, y2))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    grid_x = min(grid_w - 1, max(0, int(cx / width * grid_w)))
    grid_y = min(grid_h - 1, max(0, int(cy / height * grid_h)))
    heatmap[grid_y, grid_x] += 1
    return x1, y1, x2, y2


def collect_counts(
    entries: Sequence[Tuple[str, Path, Path]],
    selected_ids: Iterable[int],
) -> Dict[int, int]:
    selected = set(selected_ids)
    counts = {class_id: 0 for class_id in selected}
    for _, _, label_path in tqdm(entries, desc="Counting labels", unit="image"):
        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                parsed = parse_label_line(line)
                if not parsed:
                    continue
                class_id, x, y, w, h = parsed
                if class_id in selected:
                    counts[class_id] += 1
    return counts


def collect_heatmaps(
    entries: Sequence[Tuple[str, Path, Path]],
    selected_ids: Iterable[int],
    grid_w: int,
    grid_h: int,
) -> Dict[int, np.ndarray]:
    selected = set(selected_ids)
    heatmaps: Dict[int, np.ndarray] = {
        class_id: np.zeros((grid_h, grid_w), dtype=np.int32)
        for class_id in selected
    }
    for _, image_path, label_path in tqdm(entries, desc="Building heatmaps", unit="image"):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
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
    return heatmaps


def print_heatmaps(
    heatmaps: Dict[int, np.ndarray], classes: Sequence[str], name_to_idx: Dict[str, int]
) -> None:
    print("Location heatmaps (counts by grid cell):")
    for name in classes:
        class_id = name_to_idx[name]
        grid = heatmaps.get(class_id)
        if grid is None:
            continue
        print(f"  {name}:")
        for row in grid:
            print("    " + " ".join(f"{val:4d}" for val in row))


def build_bucket(
    entries: Sequence[Tuple[str, Path, Path]],
    selected_ids: Iterable[int],
    border_pad: int,
    bg_threshold: float,
    min_area: int,
    merge_iou: float,
    max_per_class: int,
    max_seg_bucket: int,
    bucket_max_images: int,
    bucket_sample_rate: float,
    debug: bool,
    debug_dir: Path,
    debug_max: int,
) -> Tuple[Dict[int, List[Blob]], List[Blob]]:
    bucket: Dict[int, List[Blob]] = {class_id: [] for class_id in selected_ids}
    seg_bucket: List[Blob] = []
    selected = set(selected_ids)
    bucket_entries = list(entries)
    if bucket_sample_rate < 1.0:
        bucket_entries = [
            entry for entry in bucket_entries if random.random() <= bucket_sample_rate
        ]
    if bucket_max_images and bucket_max_images > 0:
        random.shuffle(bucket_entries)
        bucket_entries = bucket_entries[:bucket_max_images]

    debug_saved = 0
    for split, image_path, label_path in tqdm(
        bucket_entries, desc="Building bucket", unit="image"
    ):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARN] Failed to read image {image_path}")
            continue
        height, width = image.shape[:2]
        bg_color = estimate_background_color(image, border_pad)
        seg_boxes = segment_foreground_boxes(image, bg_color, bg_threshold, min_area)
        yolo_boxes: List[Tuple[int, int, int, int]] = []
        yolo_box_ids: List[int] = []
        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                parsed = parse_label_line(line)
                if not parsed:
                    continue
                class_id, x, y, w, h = parsed
                if class_id not in selected:
                    continue
                pixel_box = yolo_to_pixel_box((x, y, w, h), width, height)
                if pixel_box:
                    yolo_boxes.append(pixel_box)
                    yolo_box_ids.append(class_id)
        merged_yolo, remaining_seg = merge_boxes(yolo_boxes, seg_boxes, merge_iou)
        if debug and debug_saved < debug_max:
            mask = build_foreground_mask(image, bg_color, bg_threshold)
            debug_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            for box in seg_boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            for box in yolo_boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            debug_name = f"{image_path.stem}_mask.png"
            ensure_dir(debug_dir)
            cv2.imwrite(str(debug_dir / debug_name), debug_img)
            debug_saved += 1
        for box, class_id in zip(merged_yolo, yolo_box_ids):
            if len(bucket[class_id]) >= max_per_class:
                continue
            x1, y1, x2, y2 = box
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            bucket[class_id].append(Blob(crop, class_id))
        for box in remaining_seg:
            if len(seg_bucket) >= max_seg_bucket:
                break
            x1, y1, x2, y2 = box
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            seg_bucket.append(Blob(crop, None))
        buckets_full = all(len(blobs) >= max_per_class for blobs in bucket.values())
        if buckets_full and len(seg_bucket) >= max_seg_bucket:
            break
    return bucket, seg_bucket


def copy_original_dataset(
    source_root: Path, dest_root: Path, splits: Sequence[str]
) -> None:
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


def build_canvas(
    image: np.ndarray,
    yolo_boxes: List[Tuple[int, int, int, int]],
    seg_boxes: List[Tuple[int, int, int, int]],
    bg_color: Tuple[int, int, int],
    keep_background: bool,
) -> np.ndarray:
    if keep_background:
        canvas = image.copy()
        for box in yolo_boxes + seg_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), bg_color, thickness=-1)
        return canvas
    canvas = np.zeros_like(image)
    canvas[:] = np.array(bg_color, dtype=np.uint8).reshape((1, 1, 3))
    return canvas


def build_blobs_from_boxes(
    image: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    class_id: Optional[int],
) -> List[Blob]:
    blobs: List[Blob] = []
    for box in boxes:
        x1, y1, x2, y2 = box
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        blobs.append(Blob(crop, class_id))
    return blobs


def pick_random_blob(bucket: Dict[int, List[Blob]], class_id: int) -> Optional[Blob]:
    options = bucket.get(class_id, [])
    if not options:
        return None
    return random.choice(options)


def build_augmented_image(
    image_path: Path,
    label_path: Path,
    selected_ids: Iterable[int],
    bucket: Dict[int, List[Blob]],
    seg_bucket: List[Blob],
    target_class_id: int,
    heatmaps: Dict[int, np.ndarray],
    border_pad: int,
    bg_threshold: float,
    min_area: int,
    merge_iou: float,
    drop_rate: float,
    collision_pad: int,
    keep_background: bool,
    placement: str,
    dense_step: int,
    fill_empty: bool,
    fill_ratio: float,
    fill_max_blobs: int,
    fill_max_tries: int,
) -> Optional[Tuple[np.ndarray, List[Tuple[int, int, int, int, int]]]]:
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    height, width = image.shape[:2]
    bg_color = estimate_background_color(image, border_pad)
    seg_boxes = segment_foreground_boxes(image, bg_color, bg_threshold, min_area)
    selected = set(selected_ids)
    yolo_boxes: List[Tuple[int, int, int, int]] = []
    yolo_class_ids: List[int] = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_label_line(line)
            if not parsed:
                continue
            class_id, x, y, w, h = parsed
            if class_id not in selected:
                continue
            pixel_box = yolo_to_pixel_box((x, y, w, h), width, height)
            if pixel_box:
                yolo_boxes.append(pixel_box)
                yolo_class_ids.append(class_id)

    merged_yolo, remaining_seg = merge_boxes(yolo_boxes, seg_boxes, merge_iou)
    canvas = build_canvas(image, merged_yolo, remaining_seg, bg_color, keep_background)

    blobs: List[Blob] = []
    for box, class_id in zip(merged_yolo, yolo_class_ids):
        if class_id != target_class_id and random.random() < drop_rate:
            continue
        blobs.extend(build_blobs_from_boxes(image, [box], class_id))
    for box in remaining_seg:
        if random.random() < drop_rate:
            continue
        blobs.extend(build_blobs_from_boxes(image, [box], None))

    target_blob = pick_random_blob(bucket, target_class_id)
    if target_blob is None:
        return None
    blobs.append(target_blob)
    if seg_bucket and random.random() < 0.5:
        blobs.append(random.choice(seg_bucket))

    placed_boxes: List[Tuple[int, int, int, int]] = []
    labels: List[Tuple[int, int, int, int, int]] = []
    random.shuffle(blobs)
    blobs = [target_blob] + [blob for blob in blobs if blob is not target_blob]
    target_heat = heatmaps.get(target_class_id)
    if placement == "dense":
        first = place_blob_dense(
            canvas,
            target_blob,
            placed_boxes,
            collision_pad=collision_pad,
            step=dense_step,
            heatmap=target_heat,
        )
    else:
        if target_heat is None:
            first = place_blob(canvas, target_blob, placed_boxes, collision_pad=collision_pad)
        else:
            first = place_blob_biased(
                canvas, target_blob, placed_boxes, target_heat, collision_pad=collision_pad
            )
    if not first:
        return None
    labels.append((target_class_id, first[0], first[1], first[2], first[3]))
    for blob in blobs[1:]:
        if placement == "dense":
            heat = heatmaps.get(blob.class_id) if blob.class_id is not None else None
            placed = place_blob_dense(
                canvas,
                blob,
                placed_boxes,
                collision_pad=collision_pad,
                step=dense_step,
                heatmap=heat,
            )
        else:
            if blob.class_id is not None and blob.class_id in heatmaps:
                placed = place_blob_biased(
                    canvas,
                    blob,
                    placed_boxes,
                    heatmaps[blob.class_id],
                    collision_pad=collision_pad,
                )
            else:
                placed = place_blob(canvas, blob, placed_boxes, collision_pad=collision_pad)
        if not placed:
            continue
        if blob.class_id is not None:
            x1, y1, x2, y2 = placed
            labels.append((blob.class_id, x1, y1, x2, y2))
    if fill_empty:
        fill_empty_space(
            canvas,
            placed_boxes,
            seg_bucket,
            collision_pad,
            placement,
            dense_step,
            fill_ratio,
            fill_max_blobs,
            fill_max_tries,
        )
    return canvas, labels


def write_label_file(
    path: Path, labels: List[Tuple[int, int, int, int, int]], width: int, height: int
) -> None:
    with path.open("w", encoding="utf-8") as f:
        for class_id, x1, y1, x2, y2 in labels:
            cx, cy, w, h = pixel_to_yolo_box((x1, y1, x2, y2), width, height)
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def init_worker(config: Dict[str, object]) -> None:
    global _WORKER_CONFIG
    _WORKER_CONFIG = config
    seed = int(config.get("seed", 13))
    seed = seed + os.getpid()
    random.seed(seed)
    np.random.seed(seed % (2**32))


def augment_worker(entry: Tuple[str, Path, Path], class_id: int, index: int) -> bool:
    cfg = _WORKER_CONFIG
    if cfg is None:
        raise RuntimeError("Worker config is not initialized.")
    split, image_path, label_path = entry
    result = build_augmented_image(
        image_path,
        label_path,
        cfg["selected_ids"],
        cfg["bucket"],
        cfg["seg_bucket"],
        class_id,
        cfg["heatmaps"],
        cfg["border_pad"],
        cfg["bg_threshold"],
        cfg["min_area"],
        cfg["merge_iou"],
        cfg["drop_rate"],
        cfg["collision_pad"],
        cfg["keep_background"],
        cfg["placement"],
        cfg["dense_step"],
        cfg["fill_empty"],
        cfg["fill_ratio"],
        cfg["fill_max_blobs"],
        cfg["fill_max_tries"],
    )
    if result is None:
        return False
    canvas, labels = result
    height, width = canvas.shape[:2]
    dest_root: Path = cfg["dest_root"]
    image_name = f"{image_path.stem}_aug_{index}{image_path.suffix}"
    label_name = f"{image_path.stem}_aug_{index}.txt"
    out_image = dest_root / split / "images" / image_name
    out_label = dest_root / split / "labels" / label_name
    ensure_dir(out_image.parent)
    ensure_dir(out_label.parent)
    if not cv2.imwrite(str(out_image), canvas):
        print(f"[WARN] Failed to write image {out_image}")
        return False
    write_label_file(out_label, labels, width, height)
    return True


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.bucket_sample_rate <= 0 or args.bucket_sample_rate > 1.0:
        raise SystemExit("--bucket-sample-rate must be in (0, 1].")
    if args.dense_step < 1:
        raise SystemExit("--dense-step must be >= 1.")
    if args.fill_ratio < 0 or args.fill_ratio > 1.0:
        raise SystemExit("--fill-ratio must be in [0, 1].")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1.")
    if args.mp_start_method not in mp.get_all_start_methods():
        raise SystemExit(f"Start method '{args.mp_start_method}' is not available.")

    class_names = load_class_names(args.source / "data.yaml")
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    missing = [name for name in args.classes if name not in name_to_idx]
    if missing:
        raise SystemExit(f"Classes not found in data.yaml: {', '.join(missing)}")
    selected_ids = [name_to_idx[name] for name in args.classes]
    if args.class_weights is not None:
        if len(args.class_weights) != len(args.classes):
            raise SystemExit("--class-weights must match the number of --classes.")
        weights = {
            name_to_idx[name]: max(0.0, float(weight))
            for name, weight in zip(args.classes, args.class_weights)
        }
    else:
        weights = {class_id: 1.0 for class_id in selected_ids}

    entries = list(
        tqdm(iter_label_images(args.source, args.splits), desc="Indexing images", unit="image")
    )
    if not entries:
        print("[WARN] No images found for the requested splits.")
        return

    counts = collect_counts(entries, selected_ids)
    heatmaps = collect_heatmaps(
        entries,
        selected_ids,
        args.heatmap_grid[0],
        args.heatmap_grid[1],
    )
    print("Selected class counts:")
    for name in args.classes:
        class_id = name_to_idx[name]
        weight = weights.get(class_id, 1.0)
        print(f"  {name}: {counts.get(class_id, 0)} (weight={weight})")
    print_heatmaps(heatmaps, args.classes, name_to_idx)

    if args.copy_original:
        copy_original_dataset(args.source, args.dest, args.splits)
    else:
        copy_metadata(args.source, args.dest)

    target = max(counts.values()) if counts else 0
    if target == 0:
        print("[WARN] No selected class boxes found; skipping augmentation.")
        return

    needed = {}
    for class_id, count in counts.items():
        shortfall = max(0, target - count)
        scaled = int(np.ceil(shortfall * weights.get(class_id, 1.0)))
        if scaled <= 0:
            continue
        needed[class_id] = min(scaled, args.max_aug_per_class)
    if not needed:
        print("Selected classes already balanced; no augmentation needed.")
        return

    debug_dir = args.debug_dir or (args.dest / "debug")
    bucket, seg_bucket = build_bucket(
        entries,
        selected_ids,
        args.border_pad,
        args.bg_threshold,
        args.min_area,
        args.merge_iou,
        args.max_bucket_per_class,
        args.max_seg_bucket,
        args.bucket_max_images,
        args.bucket_sample_rate,
        args.debug,
        debug_dir,
        args.debug_max,
    )
    if not any(bucket.values()):
        print("[WARN] No class blobs collected for augmentation.")
        return

    dest_root = args.dest
    aug_index = itertools.count()
    total_aug = 0
    fail_counts: Dict[int, int] = {cid: 0 for cid in needed}
    max_failures = 25

    pending_counts: Dict[int, int] = {cid: 0 for cid in needed}

    def pick_weighted_class() -> Optional[int]:
        candidates = [cid for cid, need in needed.items() if need - pending_counts[cid] > 0]
        if not candidates:
            return None
        weights_list = [
            max(0.0, weights.get(cid, 1.0)) * float(needed[cid] - pending_counts[cid])
            for cid in candidates
        ]
        total = sum(weights_list)
        if total <= 0:
            return random.choice(candidates)
        return random.choices(candidates, weights=weights_list, k=1)[0]

    target_total = min(args.max_aug_total, sum(needed.values()))
    worker_config: Dict[str, object] = {
        "selected_ids": selected_ids,
        "bucket": bucket,
        "seg_bucket": seg_bucket,
        "heatmaps": heatmaps,
        "border_pad": args.border_pad,
        "bg_threshold": args.bg_threshold,
        "min_area": args.min_area,
        "merge_iou": args.merge_iou,
        "drop_rate": args.drop_rate,
        "collision_pad": args.collision_pad,
        "keep_background": args.keep_background,
        "placement": args.placement,
        "dense_step": args.dense_step,
        "fill_empty": args.fill_empty,
        "fill_ratio": args.fill_ratio,
        "fill_max_blobs": args.fill_max_blobs,
        "fill_max_tries": args.fill_max_tries,
        "dest_root": dest_root,
        "seed": args.seed,
    }
    init_worker(worker_config)

    with tqdm(total=target_total, desc="Augmenting", unit="image") as pbar:
        if args.workers == 1:
            while total_aug < target_total:
                class_id = pick_weighted_class()
                if class_id is None:
                    break
                entry = random.choice(entries)
                idx = next(aug_index)
                success = augment_worker(entry, class_id, idx)
                if success:
                    needed[class_id] -= 1
                    total_aug += 1
                    pbar.update(1)
                else:
                    fail_counts[class_id] += 1
                    if fail_counts[class_id] >= max_failures:
                        print(
                            f"[WARN] Failed to augment class {class_id} after {max_failures} attempts."
                        )
                        needed[class_id] = 0
        else:
            pending: Dict[object, int] = {}
            mp_context = mp.get_context(args.mp_start_method)
            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=init_worker,
                initargs=(worker_config,),
                mp_context=mp_context,
            ) as executor:
                while total_aug < target_total:
                    while (
                        len(pending) < args.workers
                        and total_aug + len(pending) < target_total
                    ):
                        class_id = pick_weighted_class()
                        if class_id is None:
                            break
                        entry = random.choice(entries)
                        idx = next(aug_index)
                        future = executor.submit(augment_worker, entry, class_id, idx)
                        pending[future] = class_id
                        pending_counts[class_id] += 1
                    if not pending:
                        break
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        class_id = pending.pop(future)
                        pending_counts[class_id] = max(0, pending_counts[class_id] - 1)
                        try:
                            success = future.result()
                        except Exception as exc:
                            print(f"[WARN] Augmentation task failed: {exc}")
                            success = False
                        if success:
                            needed[class_id] -= 1
                            total_aug += 1
                            pbar.update(1)
                        else:
                            fail_counts[class_id] += 1
                            if fail_counts[class_id] >= max_failures:
                                print(
                                    f"[WARN] Failed to augment class {class_id} after {max_failures} attempts."
                                )
                                needed[class_id] = 0

    print(f"Augmentation complete. Added {total_aug} images.")


if __name__ == "__main__":
    main()
