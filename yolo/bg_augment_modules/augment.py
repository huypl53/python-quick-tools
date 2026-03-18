"""Augmentation pipeline for background rebalancing."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from yolo.bg_augment_modules.constants import (
    DEFAULT_MAX_PLACEMENT_TRIES,
    MIN_BIASED_PLACEMENT_TRIES,
    SEG_FILL_PROBABILITY,
    SPATIAL_GRID_CELL_SIZE,
)
from yolo.bg_augment_modules.geometry import (
    SpatialGrid,
    merge_boxes,
    rects_collide,
    scan_positions,
    total_area,
    yolo_to_pixel_box,
)
from yolo.bg_augment_modules.image import (
    estimate_background_color,
    segment_foreground_boxes,
    validate_image,
)
from yolo.bg_augment_modules.io import parse_label_line
from yolo.bg_augment_modules.transforms import paste_blob_with_blending, transform_blob
from yolo.bg_augment_modules.types import AugmentConfig, Blob


def build_canvas(
    image: np.ndarray,
    yolo_boxes: List[Tuple[int, int, int, int]],
    seg_boxes: List[Tuple[int, int, int, int]],
    bg_color: Tuple[int, int, int],
    keep_background: bool,
) -> np.ndarray:
    """Create a canvas for blob placement."""

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
    """Extract blobs from an image given a list of bounding boxes."""

    blobs: List[Blob] = []
    for box in boxes:
        x1, y1, x2, y2 = box
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        blobs.append(Blob(crop, class_id))
    return blobs


def pick_random_blob(bucket: Dict[int, List[Blob]], class_id: int) -> Optional[Blob]:
    """Pick a random blob of the specified class from the bucket."""

    options = bucket.get(class_id, [])
    if not options:
        return None
    return random.choice(options)


def place_blob(
    canvas: np.ndarray,
    blob: Blob,
    placed: Union[List[Tuple[int, int, int, int]], SpatialGrid],
    collision_pad: int = 0,
    max_tries: int = DEFAULT_MAX_PLACEMENT_TRIES,
    feather_radius: int = 0,
) -> Optional[Tuple[int, int, int, int]]:
    """Place a blob at a random non-colliding position on the canvas."""

    height, width = canvas.shape[:2]
    blob_h, blob_w = blob.image.shape[:2]
    if blob_h >= height or blob_w >= width:
        return None

    use_grid = isinstance(placed, SpatialGrid)

    for _ in range(max_tries):
        x1 = random.randint(0, width - blob_w)
        y1 = random.randint(0, height - blob_h)
        x2 = x1 + blob_w
        y2 = y1 + blob_h
        candidate = (x1, y1, x2, y2)

        if use_grid:
            if placed.collides(candidate, collision_pad):
                continue
        else:
            if any(rects_collide(candidate, other, collision_pad) for other in placed):
                continue

        if feather_radius > 0:
            paste_blob_with_blending(canvas, blob.image, x1, y1, feather_radius)
        else:
            canvas[y1:y2, x1:x2] = blob.image
        if use_grid:
            placed.insert(candidate)
        else:
            placed.append(candidate)
        return candidate
    return None


def place_blob_near_anchor(
    canvas: np.ndarray,
    blob: Blob,
    anchor: Tuple[int, int, int, int],
    placed: Union[List[Tuple[int, int, int, int]], SpatialGrid],
    collision_pad: int = 0,
    jitter: int = 0,
    max_tries: int = 20,
    feather_radius: int = 0,
) -> Optional[Tuple[int, int, int, int]]:
    """Place a blob near an anchor box with bounded jitter."""

    height, width = canvas.shape[:2]
    blob_h, blob_w = blob.image.shape[:2]
    if blob_h >= height or blob_w >= width:
        return None

    ax1, ay1, ax2, ay2 = anchor
    anchor_cx = (ax1 + ax2) / 2.0
    anchor_cy = (ay1 + ay2) / 2.0
    jitter = max(0, int(jitter))
    use_grid = isinstance(placed, SpatialGrid)

    for _ in range(max_tries):
        dx = random.randint(-jitter, jitter) if jitter > 0 else 0
        dy = random.randint(-jitter, jitter) if jitter > 0 else 0
        cx = int(round(anchor_cx + dx))
        cy = int(round(anchor_cy + dy))
        x1 = cx - blob_w // 2
        y1 = cy - blob_h // 2
        x2 = x1 + blob_w
        y2 = y1 + blob_h
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            continue
        candidate = (x1, y1, x2, y2)

        if use_grid:
            if placed.collides(candidate, collision_pad):
                continue
        else:
            if any(rects_collide(candidate, other, collision_pad) for other in placed):
                continue

        if feather_radius > 0:
            paste_blob_with_blending(canvas, blob.image, x1, y1, feather_radius)
        else:
            canvas[y1:y2, x1:x2] = blob.image
        if use_grid:
            placed.insert(candidate)
        else:
            placed.append(candidate)
        return candidate
    return None


def place_blob_dense(
    canvas: np.ndarray,
    blob: Blob,
    placed: Union[List[Tuple[int, int, int, int]], SpatialGrid],
    collision_pad: int = 0,
    step: int = 2,
    heatmap: Optional[np.ndarray] = None,
    feather_radius: int = 0,
) -> Optional[Tuple[int, int, int, int]]:
    """Place a blob using dense scanning to find optimal position."""

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

    use_grid = isinstance(placed, SpatialGrid)

    for y in ys:
        for x in xs:
            x2 = x + blob_w
            y2 = y + blob_h
            candidate = (x, y, x2, y2)

            if use_grid:
                if placed.collides(candidate, collision_pad):
                    continue
            else:
                if any(rects_collide(candidate, other, collision_pad) for other in placed):
                    continue

            if heatmap is None:
                if feather_radius > 0:
                    paste_blob_with_blending(canvas, blob.image, x, y, feather_radius)
                else:
                    canvas[y:y2, x:x2] = blob.image
                if use_grid:
                    placed.insert(candidate)
                else:
                    placed.append(candidate)
                return candidate
            cx = (x + x2) / 2.0
            cy = (y + y2) / 2.0
            if width > 0 and height > 0 and grid_w > 0 and grid_h > 0:
                grid_x = min(grid_w - 1, max(0, int(cx / width * grid_w)))
                grid_y = min(grid_h - 1, max(0, int(cy / height * grid_h)))
            else:
                grid_x = grid_y = 0
            score = int(heatmap[grid_y, grid_x])
            if best_score is None or score < best_score:
                best_score = score
                best = candidate
                if best_score == 0:
                    if feather_radius > 0:
                        paste_blob_with_blending(canvas, blob.image, x, y, feather_radius)
                    else:
                        canvas[y:y2, x:x2] = blob.image
                    if use_grid:
                        placed.insert(candidate)
                    else:
                        placed.append(candidate)
                    heatmap[grid_y, grid_x] += 1
                    return candidate
    if best is not None and heatmap is not None:
        x1, y1, x2, y2 = best
        if feather_radius > 0:
            paste_blob_with_blending(canvas, blob.image, x1, y1, feather_radius)
        else:
            canvas[y1:y2, x1:x2] = blob.image
        if use_grid:
            placed.insert(best)
        else:
            placed.append(best)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        if width > 0 and height > 0 and grid_w > 0 and grid_h > 0:
            grid_x = min(grid_w - 1, max(0, int(cx / width * grid_w)))
            grid_y = min(grid_h - 1, max(0, int(cy / height * grid_h)))
        else:
            grid_x = grid_y = 0
        heatmap[grid_y, grid_x] += 1
        return best
    return None


def fill_empty_space(
    canvas: np.ndarray,
    placed: Union[List[Tuple[int, int, int, int]], SpatialGrid],
    seg_bucket: List[Blob],
    collision_pad: int,
    placement: str,
    dense_step: int,
    fill_ratio: float,
    fill_max_blobs: int,
    fill_max_tries: int,
    feather_radius: int = 0,
) -> None:
    """Fill empty canvas space with extra segmentation blobs."""

    if not seg_bucket:
        return
    height, width = canvas.shape[:2]
    if width == 0 or height == 0:
        return
    canvas_area = float(width * height)
    target_ratio = max(0.0, min(1.0, fill_ratio))
    added = 0
    tries = 0

    if isinstance(placed, SpatialGrid):
        boxes_for_area = placed.get_all_boxes()
    else:
        boxes_for_area = placed

    while added < fill_max_blobs and tries < fill_max_tries:
        if isinstance(placed, SpatialGrid):
            boxes_for_area = placed.get_all_boxes()
        current_ratio = total_area(boxes_for_area) / canvas_area
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
                feather_radius=feather_radius,
            )
        else:
            placed_box = place_blob(
                canvas, blob, placed, collision_pad=collision_pad, feather_radius=feather_radius
            )
        tries += 1
        if placed_box:
            added += 1


def place_blob_biased(
    canvas: np.ndarray,
    blob: Blob,
    placed: Union[List[Tuple[int, int, int, int]], SpatialGrid],
    heatmap: np.ndarray,
    collision_pad: int = 0,
    max_tries: int = DEFAULT_MAX_PLACEMENT_TRIES,
    feather_radius: int = 0,
) -> Optional[Tuple[int, int, int, int]]:
    """Place a blob with bias toward low-density heatmap regions."""

    height, width = canvas.shape[:2]
    blob_h, blob_w = blob.image.shape[:2]
    if blob_h >= height or blob_w >= width:
        return None
    grid_h, grid_w = heatmap.shape
    if width == 0 or height == 0 or grid_w == 0 or grid_h == 0:
        return None
    candidates: List[Tuple[int, int, int, int, int]] = []
    tries = max(MIN_BIASED_PLACEMENT_TRIES, max_tries)

    use_grid = isinstance(placed, SpatialGrid)

    for _ in range(tries):
        x1 = random.randint(0, width - blob_w)
        y1 = random.randint(0, height - blob_h)
        x2 = x1 + blob_w
        y2 = y1 + blob_h
        candidate = (x1, y1, x2, y2)

        if use_grid:
            if placed.collides(candidate, collision_pad):
                continue
        else:
            if any(rects_collide(candidate, other, collision_pad) for other in placed):
                continue

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        grid_x = min(grid_w - 1, max(0, int(cx / width * grid_w)))
        grid_y = min(grid_h - 1, max(0, int(cy / height * grid_h)))
        score = heatmap[grid_y, grid_x]
        candidates.append((score, x1, y1, x2, y2))
    if not candidates:
        return place_blob(
            canvas,
            blob,
            placed,
            collision_pad=collision_pad,
            max_tries=max_tries,
            feather_radius=feather_radius,
        )
    candidates.sort(key=lambda item: item[0])
    _, x1, y1, x2, y2 = candidates[0]
    if feather_radius > 0:
        paste_blob_with_blending(canvas, blob.image, x1, y1, feather_radius)
    else:
        canvas[y1:y2, x1:x2] = blob.image
    if use_grid:
        placed.insert((x1, y1, x2, y2))
    else:
        placed.append((x1, y1, x2, y2))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    grid_x = min(grid_w - 1, max(0, int(cx / width * grid_w)))
    grid_y = min(grid_h - 1, max(0, int(cy / height * grid_h)))
    heatmap[grid_y, grid_x] += 1
    return x1, y1, x2, y2


def _extract_scene(
    image_path: Path,
    label_path: Path,
    selected_ids: Iterable[int],
    border_pad: int,
    bg_threshold: float,
    min_area: int,
    merge_iou: float,
    no_seg: bool,
) -> Optional[
    Tuple[
        np.ndarray,
        Tuple[int, int, int],
        List[Tuple[int, int, int, int]],
        List[Tuple[int, int, int, int]],
        List[int],
    ]
]:
    image = cv2.imread(str(image_path))
    image = validate_image(image, str(image_path))
    if image is None:
        return None
    height, width = image.shape[:2]
    bg_color = estimate_background_color(image, border_pad)
    if no_seg:
        seg_boxes: List[Tuple[int, int, int, int]] = []
    else:
        seg_boxes = segment_foreground_boxes(image, bg_color, bg_threshold, min_area)
    selected = set(selected_ids)
    yolo_boxes: List[Tuple[int, int, int, int]] = []
    yolo_class_ids: List[int] = []
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
                    yolo_class_ids.append(class_id)
    except OSError as e:
        print(f"[WARN] Failed to read label file {label_path}: {e}")
        return None

    merged_yolo, remaining_seg = merge_boxes(yolo_boxes, seg_boxes, merge_iou)
    return image, bg_color, merged_yolo, remaining_seg, yolo_class_ids


def _build_recompose_augmented_image(
    image: np.ndarray,
    bg_color: Tuple[int, int, int],
    merged_yolo: List[Tuple[int, int, int, int]],
    remaining_seg: List[Tuple[int, int, int, int]],
    yolo_class_ids: List[int],
    bucket: Dict[int, List[Blob]],
    seg_bucket: List[Blob],
    target_class_id: int,
    heatmaps: Dict[int, np.ndarray],
    drop_rate: float,
    collision_pad: int,
    keep_background: bool,
    placement: str,
    dense_step: int,
    fill_empty: bool,
    fill_ratio: float,
    fill_max_blobs: int,
    fill_max_tries: int,
    no_seg: bool,
    feather_radius: int,
    scale_range: Optional[Tuple[float, float]],
    rotate_max: float,
    flip_h_prob: float,
    flip_v_prob: float,
    color_jitter: float,
) -> Optional[Tuple[np.ndarray, List[Tuple[int, int, int, int, int]]]]:
    canvas = build_canvas(image, merged_yolo, remaining_seg, bg_color, keep_background)

    blobs: List[Blob] = []
    for box, class_id in zip(merged_yolo, yolo_class_ids):
        if class_id != target_class_id and random.random() < drop_rate:
            continue
        blobs.extend(build_blobs_from_boxes(image, [box], class_id))
    if not no_seg:
        for box in remaining_seg:
            if random.random() < drop_rate:
                continue
            blobs.extend(build_blobs_from_boxes(image, [box], None))

    target_blob = pick_random_blob(bucket, target_class_id)
    if target_blob is None:
        return None

    transforms_enabled = (
        scale_range is not None
        or rotate_max > 0
        or flip_h_prob > 0
        or flip_v_prob > 0
        or color_jitter > 0
    )

    if transforms_enabled:
        transform_config = AugmentConfig(
            scale_range=scale_range,
            rotate_max=rotate_max,
            flip_h_prob=flip_h_prob,
            flip_v_prob=flip_v_prob,
            color_jitter=color_jitter,
        )
        transformed_target = transform_blob(target_blob, transform_config, bg_color)
        if transformed_target is None:
            return None
        target_blob = transformed_target

        transformed_blobs: List[Blob] = []
        for blob in blobs:
            if blob.class_id is not None and random.random() < 0.5:
                transformed = transform_blob(blob, transform_config, bg_color)
                if transformed is not None:
                    transformed_blobs.append(transformed)
                else:
                    transformed_blobs.append(blob)
            else:
                transformed_blobs.append(blob)
        blobs = transformed_blobs

    blobs.append(target_blob)
    if not no_seg and seg_bucket and random.random() < SEG_FILL_PROBABILITY:
        blobs.append(random.choice(seg_bucket))

    height, width = image.shape[:2]
    spatial_grid = SpatialGrid(width, height, cell_size=SPATIAL_GRID_CELL_SIZE)
    labels: List[Tuple[int, int, int, int, int]] = []
    random.shuffle(blobs)
    blobs = [target_blob] + [blob for blob in blobs if blob is not target_blob]
    target_heat = heatmaps.get(target_class_id)
    if placement == "dense":
        first = place_blob_dense(
            canvas,
            target_blob,
            spatial_grid,
            collision_pad=collision_pad,
            step=dense_step,
            heatmap=target_heat,
            feather_radius=feather_radius,
        )
    else:
        if target_heat is None:
            first = place_blob(
                canvas,
                target_blob,
                spatial_grid,
                collision_pad=collision_pad,
                feather_radius=feather_radius,
            )
        else:
            first = place_blob_biased(
                canvas,
                target_blob,
                spatial_grid,
                target_heat,
                collision_pad=collision_pad,
                feather_radius=feather_radius,
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
                spatial_grid,
                collision_pad=collision_pad,
                step=dense_step,
                heatmap=heat,
                feather_radius=feather_radius,
            )
        else:
            if blob.class_id is not None and blob.class_id in heatmaps:
                placed = place_blob_biased(
                    canvas,
                    blob,
                    spatial_grid,
                    heatmaps[blob.class_id],
                    collision_pad=collision_pad,
                    feather_radius=feather_radius,
                )
            else:
                placed = place_blob(
                    canvas,
                    blob,
                    spatial_grid,
                    collision_pad=collision_pad,
                    feather_radius=feather_radius,
                )
        if not placed:
            continue
        if blob.class_id is not None:
            x1, y1, x2, y2 = placed
            labels.append((blob.class_id, x1, y1, x2, y2))
    if fill_empty and not no_seg:
        fill_empty_space(
            canvas,
            spatial_grid,
            seg_bucket,
            collision_pad,
            placement,
            dense_step,
            fill_ratio,
            fill_max_blobs,
            fill_max_tries,
            feather_radius=feather_radius,
        )
    return canvas, labels


def _build_layout_augmented_image(
    image: np.ndarray,
    bg_color: Tuple[int, int, int],
    merged_yolo: List[Tuple[int, int, int, int]],
    remaining_seg: List[Tuple[int, int, int, int]],
    yolo_class_ids: List[int],
    bucket: Dict[int, List[Blob]],
    seg_bucket: List[Blob],
    target_class_id: int,
    heatmaps: Dict[int, np.ndarray],
    drop_rate: float,
    collision_pad: int,
    keep_background: bool,
    placement: str,
    dense_step: int,
    fill_empty: bool,
    fill_ratio: float,
    fill_max_blobs: int,
    fill_max_tries: int,
    no_seg: bool,
    feather_radius: int,
    scale_range: Optional[Tuple[float, float]],
    rotate_max: float,
    flip_h_prob: float,
    flip_v_prob: float,
    color_jitter: float,
    layout_jitter: int,
    layout_max_tries: int,
) -> Optional[Tuple[np.ndarray, List[Tuple[int, int, int, int, int]]]]:
    canvas = build_canvas(image, merged_yolo, remaining_seg, bg_color, keep_background)

    anchored: List[Tuple[Blob, Tuple[int, int, int, int]]] = []
    for box, class_id in zip(merged_yolo, yolo_class_ids):
        if class_id != target_class_id and random.random() < drop_rate:
            continue
        x1, y1, x2, y2 = box
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        anchored.append((Blob(crop, class_id), box))
    if not no_seg:
        for box in remaining_seg:
            if random.random() < drop_rate:
                continue
            x1, y1, x2, y2 = box
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            anchored.append((Blob(crop, None), box))

    target_blob = pick_random_blob(bucket, target_class_id)
    if target_blob is None:
        return None

    transforms_enabled = (
        scale_range is not None
        or rotate_max > 0
        or flip_h_prob > 0
        or flip_v_prob > 0
        or color_jitter > 0
    )
    if transforms_enabled:
        transform_config = AugmentConfig(
            scale_range=scale_range,
            rotate_max=rotate_max,
            flip_h_prob=flip_h_prob,
            flip_v_prob=flip_v_prob,
            color_jitter=color_jitter,
        )
        transformed_anchored: List[Tuple[Blob, Tuple[int, int, int, int]]] = []
        for blob, anchor in anchored:
            transformed = transform_blob(blob, transform_config, bg_color)
            transformed_anchored.append((transformed or blob, anchor))
        anchored = transformed_anchored
        transformed_target = transform_blob(target_blob, transform_config, bg_color)
        if transformed_target is None:
            return None
        target_blob = transformed_target

    anchored.sort(
        key=lambda item: (item[1][2] - item[1][0]) * (item[1][3] - item[1][1]),
        reverse=True,
    )

    height, width = image.shape[:2]
    spatial_grid = SpatialGrid(width, height, cell_size=SPATIAL_GRID_CELL_SIZE)
    labels: List[Tuple[int, int, int, int, int]] = []

    for blob, anchor in anchored:
        placed = place_blob_near_anchor(
            canvas,
            blob,
            anchor,
            spatial_grid,
            collision_pad=collision_pad,
            jitter=layout_jitter,
            max_tries=layout_max_tries,
            feather_radius=feather_radius,
        )
        if not placed:
            if placement == "dense":
                heat = heatmaps.get(blob.class_id) if blob.class_id is not None else None
                placed = place_blob_dense(
                    canvas,
                    blob,
                    spatial_grid,
                    collision_pad=collision_pad,
                    step=dense_step,
                    heatmap=heat,
                    feather_radius=feather_radius,
                )
            else:
                if blob.class_id is not None and blob.class_id in heatmaps:
                    placed = place_blob_biased(
                        canvas,
                        blob,
                        spatial_grid,
                        heatmaps[blob.class_id],
                        collision_pad=collision_pad,
                        feather_radius=feather_radius,
                    )
                else:
                    placed = place_blob(
                        canvas,
                        blob,
                        spatial_grid,
                        collision_pad=collision_pad,
                        feather_radius=feather_radius,
                    )
        if not placed:
            continue
        if blob.class_id is not None:
            labels.append((blob.class_id, placed[0], placed[1], placed[2], placed[3]))

    target_heat = heatmaps.get(target_class_id)
    if placement == "dense":
        first = place_blob_dense(
            canvas,
            target_blob,
            spatial_grid,
            collision_pad=collision_pad,
            step=dense_step,
            heatmap=target_heat,
            feather_radius=feather_radius,
        )
    else:
        if target_heat is None:
            first = place_blob(
                canvas,
                target_blob,
                spatial_grid,
                collision_pad=collision_pad,
                feather_radius=feather_radius,
            )
        else:
            first = place_blob_biased(
                canvas,
                target_blob,
                spatial_grid,
                target_heat,
                collision_pad=collision_pad,
                feather_radius=feather_radius,
            )
    if not first:
        return None
    labels.append((target_class_id, first[0], first[1], first[2], first[3]))

    if fill_empty and not no_seg:
        fill_empty_space(
            canvas,
            spatial_grid,
            seg_bucket,
            collision_pad,
            placement,
            dense_step,
            fill_ratio,
            fill_max_blobs,
            fill_max_tries,
            feather_radius=feather_radius,
        )

    return canvas, labels


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
    no_seg: bool = False,
    feather_radius: int = 0,
    scale_range: Optional[Tuple[float, float]] = None,
    rotate_max: float = 0.0,
    flip_h_prob: float = 0.0,
    flip_v_prob: float = 0.0,
    color_jitter: float = 0.0,
    layout_perturb_prob: float = 0.0,
    layout_jitter: int = 12,
    layout_max_tries: int = 20,
) -> Optional[Tuple[np.ndarray, List[Tuple[int, int, int, int, int]]]]:
    """Build a single augmented image by rearranging blobs."""

    scene = _extract_scene(
        image_path,
        label_path,
        selected_ids,
        border_pad,
        bg_threshold,
        min_area,
        merge_iou,
        no_seg,
    )
    if scene is None:
        return None
    image, bg_color, merged_yolo, remaining_seg, yolo_class_ids = scene

    if layout_perturb_prob > 0 and random.random() < layout_perturb_prob:
        return _build_layout_augmented_image(
            image,
            bg_color,
            merged_yolo,
            remaining_seg,
            yolo_class_ids,
            bucket,
            seg_bucket,
            target_class_id,
            heatmaps,
            drop_rate,
            collision_pad,
            keep_background,
            placement,
            dense_step,
            fill_empty,
            fill_ratio,
            fill_max_blobs,
            fill_max_tries,
            no_seg,
            feather_radius,
            scale_range,
            rotate_max,
            flip_h_prob,
            flip_v_prob,
            color_jitter,
            layout_jitter,
            layout_max_tries,
        )

    return _build_recompose_augmented_image(
        image,
        bg_color,
        merged_yolo,
        remaining_seg,
        yolo_class_ids,
        bucket,
        seg_bucket,
        target_class_id,
        heatmaps,
        drop_rate,
        collision_pad,
        keep_background,
        placement,
        dense_step,
        fill_empty,
        fill_ratio,
        fill_max_blobs,
        fill_max_tries,
        no_seg,
        feather_radius,
        scale_range,
        rotate_max,
        flip_h_prob,
        flip_v_prob,
        color_jitter,
    )
