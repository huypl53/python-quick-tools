"""Bucket build/load utilities for background augmentation."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
from tqdm import tqdm

from yolo.bg_augment_modules.geometry import merge_boxes, yolo_to_pixel_box
from yolo.bg_augment_modules.image import (
    build_foreground_mask,
    estimate_background_color,
    segment_foreground_boxes,
    validate_image,
)
from yolo.bg_augment_modules.io import ensure_dir, list_image_files, parse_label_line, sanitize_name
from yolo.bg_augment_modules.types import Blob


def build_bucket(
    entries: Sequence[Tuple[str, Path, Path]],
    selected_ids: Sequence[int],
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
    no_seg: bool = False,
) -> Tuple[Dict[int, List[Blob]], List[Blob]]:
    """Build blob buckets by extracting crops from dataset images."""

    bucket: Dict[int, List[Blob]] = {class_id: [] for class_id in selected_ids}
    seg_bucket: List[Blob] = []
    selected = set(selected_ids)
    bucket_entries = list(entries)
    if bucket_sample_rate < 1.0:
        bucket_entries = [entry for entry in bucket_entries if random.random() <= bucket_sample_rate]
    if bucket_max_images and bucket_max_images > 0:
        random.shuffle(bucket_entries)
        bucket_entries = bucket_entries[:bucket_max_images]

    debug_saved = 0
    for _, image_path, label_path in tqdm(
        bucket_entries, desc="Building bucket", unit="image"
    ):
        image = cv2.imread(str(image_path))
        image = validate_image(image, str(image_path))
        if image is None:
            print(f"[WARN] Failed to read or validate image {image_path}")
            continue
        height, width = image.shape[:2]
        bg_color = estimate_background_color(image, border_pad)
        seg_boxes = segment_foreground_boxes(image, bg_color, bg_threshold, min_area)
        yolo_boxes: List[Tuple[int, int, int, int]] = []
        yolo_box_ids: List[int] = []
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
                    if pixel_box:
                        yolo_boxes.append(pixel_box)
                        yolo_box_ids.append(class_id)
        except OSError as e:
            print(f"[WARN] Failed to read label file {label_path}: {e}")
            continue
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
            if ensure_dir(debug_dir):
                if not cv2.imwrite(str(debug_dir / debug_name), debug_img):
                    print(f"[WARN] Failed to write debug image {debug_dir / debug_name}")
                else:
                    debug_saved += 1
        for box, class_id in zip(merged_yolo, yolo_box_ids):
            if len(bucket[class_id]) >= max_per_class:
                continue
            x1, y1, x2, y2 = box
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            bucket[class_id].append(Blob(crop, class_id))
        if not no_seg:
            for box in remaining_seg:
                if len(seg_bucket) >= max_seg_bucket:
                    break
                x1, y1, x2, y2 = box
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                seg_bucket.append(Blob(crop, None))
        buckets_full = all(len(blobs) >= max_per_class for blobs in bucket.values())
        if buckets_full and (no_seg or len(seg_bucket) >= max_seg_bucket):
            break
    return bucket, seg_bucket


def dump_bucket(
    bucket: Dict[int, List[Blob]],
    seg_bucket: List[Blob],
    class_names: Sequence[str],
    output_dir: Path,
    limit: int,
) -> None:
    """Dump bucket blobs to disk for debugging inspection."""

    save_bucket_dir(bucket, seg_bucket, class_names, output_dir, limit=limit)


def save_bucket_dir(
    bucket: Dict[int, List[Blob]],
    seg_bucket: List[Blob],
    class_names: Sequence[str],
    output_dir: Path,
    limit: int = 0,
) -> Tuple[Dict[int, int], int]:
    """Save bucket blobs to disk as images."""

    try:
        if output_dir.exists() and any(output_dir.iterdir()):
            print(f"[WARN] Bucket dir not empty: {output_dir}")
    except OSError as e:
        print(f"[WARN] Error checking output dir {output_dir}: {e}")
    if not ensure_dir(output_dir):
        return {}, 0
    classes_dir = output_dir / "classes"
    seg_dir = output_dir / "seg"
    if not ensure_dir(classes_dir) or not ensure_dir(seg_dir):
        return {}, 0

    saved_counts: Dict[int, int] = {}
    for class_id, blobs in bucket.items():
        name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        class_dir = classes_dir / f"{class_id:03d}_{sanitize_name(name)}"
        if not ensure_dir(class_dir):
            saved_counts[class_id] = 0
            continue
        indices = list(range(len(blobs)))
        if limit and limit > 0 and len(indices) > limit:
            indices = random.sample(indices, limit)
        saved = 0
        for out_idx, idx in enumerate(indices):
            blob = blobs[idx]
            out_path = class_dir / f"{out_idx:06d}.png"
            if cv2.imwrite(str(out_path), blob.image):
                saved += 1
            else:
                print(f"[WARN] Failed to write {out_path}")
        saved_counts[class_id] = saved

    seg_saved = 0
    indices = list(range(len(seg_bucket)))
    if limit and limit > 0 and len(indices) > limit:
        indices = random.sample(indices, limit)
    for out_idx, idx in enumerate(indices):
        blob = seg_bucket[idx]
        out_path = seg_dir / f"{out_idx:06d}.png"
        if cv2.imwrite(str(out_path), blob.image):
            seg_saved += 1
        else:
            print(f"[WARN] Failed to write {out_path}")

    return saved_counts, seg_saved


def load_bucket_from_dir(
    bucket_dir: Path,
    selected_ids: Sequence[int],
    class_names: Sequence[str],
    max_per_class: int,
    max_seg_bucket: int,
) -> Tuple[Dict[int, List[Blob]], List[Blob]]:
    """Load blob buckets from a previously saved bucket directory."""

    bucket: Dict[int, List[Blob]] = {class_id: [] for class_id in selected_ids}
    seg_bucket: List[Blob] = []
    classes_dir = bucket_dir / "classes"
    seg_dir = bucket_dir / "seg"
    if classes_dir.exists():
        class_dirs = [path for path in classes_dir.iterdir() if path.is_dir()]
    else:
        class_dirs = [
            path for path in bucket_dir.iterdir() if path.is_dir() and path.name != "seg"
        ]
        if class_dirs:
            print("[WARN] Bucket dir missing 'classes' folder; using root folders.")
    if not class_dirs:
        print(f"[WARN] No class folders found in {bucket_dir}")
    name_lookup = {sanitize_name(name): idx for idx, name in enumerate(class_names)}

    class_files: List[Tuple[int, Path]] = []
    for class_dir in class_dirs:
        match = re.match(r"^(\d+)", class_dir.name)
        class_id = int(match.group(1)) if match else None
        if class_id is None:
            lookup_id = name_lookup.get(sanitize_name(class_dir.name))
            class_id = lookup_id if lookup_id is not None else None
        if class_id is None:
            print(f"[WARN] Skipping bucket folder without class id: {class_dir}")
            continue
        if class_id not in selected_ids:
            continue
        files = list_image_files(class_dir)
        if max_per_class > 0 and len(files) > max_per_class:
            files = random.sample(files, max_per_class)
        class_files.extend([(class_id, path) for path in files])

    seg_files: List[Path] = []
    if seg_dir.exists():
        seg_files = list_image_files(seg_dir)
        if max_seg_bucket > 0 and len(seg_files) > max_seg_bucket:
            seg_files = random.sample(seg_files, max_seg_bucket)

    total_files = len(class_files) + len(seg_files)
    if total_files == 0:
        return bucket, seg_bucket

    with tqdm(total=total_files, desc="Loading bucket", unit="image") as pbar:
        for class_id, path in class_files:
            image = cv2.imread(str(path))
            image = validate_image(image, str(path))
            if image is None:
                print(f"[WARN] Failed to read or validate bucket image {path}")
                pbar.update(1)
                continue
            bucket[class_id].append(Blob(image, class_id))
            pbar.update(1)
        for path in seg_files:
            image = cv2.imread(str(path))
            image = validate_image(image, str(path))
            if image is None:
                print(f"[WARN] Failed to read or validate bucket image {path}")
                pbar.update(1)
                continue
            seg_bucket.append(Blob(image, None))
            pbar.update(1)

    return bucket, seg_bucket


def print_bucket_stats(
    bucket_counts: Dict[int, int],
    seg_count: int,
    class_names: Sequence[str],
    selected_ids: Sequence[int],
) -> None:
    """Print bucket statistics to stdout."""

    print("Bucket stats (saved items):")
    for class_id in selected_ids:
        name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        print(f"  {name}: {bucket_counts.get(class_id, 0)}")
    print(f"  seg: {seg_count}")
