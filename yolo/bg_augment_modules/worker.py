"""Worker helpers for multiprocessing augmentation."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from yolo.bg_augment_modules.augment import build_augmented_image
from yolo.bg_augment_modules.io import ensure_dir, write_label_file
from yolo.bg_augment_modules.types import WorkerConfigDict


_WORKER_CONFIG: Optional[WorkerConfigDict] = None


def init_worker(config: WorkerConfigDict) -> None:
    """Initialize a worker process with the given configuration."""

    global _WORKER_CONFIG
    _WORKER_CONFIG = config
    seed = int(config.get("seed", 13))
    seed = seed + os.getpid()
    random.seed(seed)
    np.random.seed(seed % (2**32))


def augment_worker(entry: Tuple[str, Path, Path], class_id: int, index: int) -> bool:
    """Worker function to build and save a single augmented image."""

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
        cfg["no_seg"],
        cfg["feather_radius"],
        cfg["scale_range"],
        cfg["rotate_max"],
        cfg["flip_h_prob"],
        cfg["flip_v_prob"],
        cfg["color_jitter"],
        cfg["layout_perturb_prob"],
        cfg["layout_jitter"],
        cfg["layout_max_tries"],
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
