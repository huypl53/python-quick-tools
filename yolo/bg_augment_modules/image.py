"""Image preprocessing utilities."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from yolo.bg_augment_modules.constants import (
    ASPECT_RATIO_THRESHOLD,
    BORDER_TOLERANCE,
    COLOR_QUANT_BIN_SIZE,
    COLOR_QUANT_BINS,
)


def validate_image(image: Optional[np.ndarray], source: str = "image") -> Optional[np.ndarray]:
    """Validate that an image was loaded correctly and has valid dimensions."""

    if image is None:
        return None
    if image.ndim < 2:
        print(f"[WARN] Invalid image dimensions for {source}: ndim={image.ndim}")
        return None
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] not in (3, 4):
        print(f"[WARN] Unexpected channel count for {source}: {image.shape[2]}")
        return None
    if image.shape[0] == 0 or image.shape[1] == 0:
        print(f"[WARN] Zero-dimension image for {source}: {image.shape}")
        return None
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def estimate_background_color(image: np.ndarray, pad: int) -> Tuple[int, int, int]:
    """Estimate the background color by sampling border pixels."""

    height, width = image.shape[:2]
    if height == 0 or width == 0:
        return (0, 0, 0)
    pad = max(1, min(pad, min(height, width) // 2))

    top = image[:pad, :].reshape(-1, image.shape[2])
    bottom = image[-pad:, :].reshape(-1, image.shape[2])
    if height > 2 * pad:
        left = image[pad:-pad, :pad].reshape(-1, image.shape[2])
        right = image[pad:-pad, -pad:].reshape(-1, image.shape[2])
    else:
        left = np.empty((0, image.shape[2]))
        right = np.empty((0, image.shape[2]))

    if left.size > 0 or right.size > 0:
        samples = np.vstack([top, bottom, left, right])
    else:
        samples = np.vstack([top, bottom])

    if samples.size == 0:
        return int(image[0, 0, 0]), int(image[0, 0, 1]), int(image[0, 0, 2])

    quant = (samples // COLOR_QUANT_BIN_SIZE).astype(np.int32)
    keys = (quant[:, 0] * COLOR_QUANT_BIN_SIZE + quant[:, 1]) * COLOR_QUANT_BIN_SIZE + quant[:, 2]
    counts = np.bincount(keys, minlength=COLOR_QUANT_BINS)
    dominant = int(np.argmax(counts))

    target = np.array(
        [
            dominant // 256,
            (dominant // COLOR_QUANT_BIN_SIZE) % COLOR_QUANT_BIN_SIZE,
            dominant % COLOR_QUANT_BIN_SIZE,
        ],
        dtype=np.int32,
    )
    match = np.all(quant == target, axis=1)
    if not np.any(match):
        mean_color = samples.mean(axis=0)
    else:
        mean_color = samples[match].mean(axis=0)
    return int(mean_color[0]), int(mean_color[1]), int(mean_color[2])


def build_foreground_mask(
    image: np.ndarray, bg_color: Tuple[int, int, int], threshold: float
) -> np.ndarray:
    """Create a binary mask of foreground (non-background) pixels."""

    tol = int(round(threshold))
    lower = np.clip(np.array(bg_color, dtype=np.int16) - tol, 0, 255).astype(np.uint8)
    upper = np.clip(np.array(bg_color, dtype=np.int16) + tol, 0, 255).astype(np.uint8)
    bg_mask = cv2.inRange(image, lower, upper)
    return cv2.bitwise_not(bg_mask)


def segment_foreground_boxes(
    image: np.ndarray, bg_color: Tuple[int, int, int], threshold: float, min_area: int
) -> List[Tuple[int, int, int, int]]:
    """Segment foreground objects and return their bounding boxes."""

    mask = build_foreground_mask(image, bg_color, threshold)
    h, w = mask.shape
    if h == 0 or w == 0:
        return []
    mask = cv2.rectangle(mask, (0, 0), (w, h), 0, thickness=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(mask)
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw == 0 or ch == 0:
            continue
        aspect = max(cw / float(ch), ch / float(cw))
        touches_border = (
            x <= BORDER_TOLERANCE
            or y <= BORDER_TOLERANCE
            or (x + cw) >= (w - BORDER_TOLERANCE)
            or (y + ch) >= (h - BORDER_TOLERANCE)
        )
        if touches_border and aspect >= ASPECT_RATIO_THRESHOLD:
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
