"""Blob transform utilities (scale, rotate, flip, color jitter, blending)."""

from __future__ import annotations

import random
from typing import Optional, Tuple

import cv2
import numpy as np

from yolo.bg_augment_modules.types import AugmentConfig, Blob


def create_feathered_mask(blob_shape: Tuple[int, int], feather_radius: int = 5) -> np.ndarray:
    """Create a feathered alpha mask for smooth blending."""

    h, w = blob_shape
    mask = np.ones((h, w), dtype=np.float32)

    for i in range(min(feather_radius, h // 2, w // 2)):
        alpha = (i + 1) / feather_radius
        mask[i, :] = np.minimum(mask[i, :], alpha)
        mask[h - 1 - i, :] = np.minimum(mask[h - 1 - i, :], alpha)
        mask[:, i] = np.minimum(mask[:, i], alpha)
        mask[:, w - 1 - i] = np.minimum(mask[:, w - 1 - i], alpha)

    ksize = feather_radius * 2 + 1
    return cv2.GaussianBlur(mask, (ksize, ksize), 0)


def paste_blob_with_blending(
    canvas: np.ndarray, blob: np.ndarray, x: int, y: int, feather_radius: int = 5
) -> None:
    """Paste blob onto canvas with feathered edges (modifies canvas in place)."""

    h, w = blob.shape[:2]
    mask = create_feathered_mask((h, w), feather_radius)
    mask = mask[:, :, np.newaxis]

    roi = canvas[y : y + h, x : x + w].astype(np.float32)
    blob_f = blob.astype(np.float32)
    blended = roi * (1 - mask) + blob_f * mask
    canvas[y : y + h, x : x + w] = blended.astype(np.uint8)


def apply_random_scale(
    blob: np.ndarray, scale_range: Tuple[float, float] = (0.7, 1.3), min_size: int = 16
) -> Optional[np.ndarray]:
    """Apply random scaling to blob."""

    scale = random.uniform(scale_range[0], scale_range[1])
    h, w = blob.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    if new_h < min_size or new_w < min_size:
        return None
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(blob, (new_w, new_h), interpolation=interp)


def apply_random_rotation(
    blob: np.ndarray, max_angle: float = 15.0, bg_color: Tuple[int, int, int] = (128, 128, 128)
) -> np.ndarray:
    """Apply random rotation to blob."""

    angle = random.uniform(-max_angle, max_angle)
    h, w = blob.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    return cv2.warpAffine(
        blob,
        M,
        (new_w, new_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=bg_color,
    )


def apply_random_flip(blob: np.ndarray, h_prob: float = 0.5, v_prob: float = 0.0) -> np.ndarray:
    """Apply random horizontal/vertical flip."""

    if random.random() < h_prob:
        blob = cv2.flip(blob, 1)
    if random.random() < v_prob:
        blob = cv2.flip(blob, 0)
    return blob


def apply_color_jitter(
    blob: np.ndarray, brightness: float = 0.2, contrast: float = 0.2, saturation: float = 0.2
) -> np.ndarray:
    """Apply random color jitter to blob."""

    result = blob.astype(np.float32)

    if brightness > 0:
        beta = random.uniform(-brightness, brightness) * 255
        result = np.clip(result + beta, 0, 255)

    if contrast > 0:
        alpha = 1 + random.uniform(-contrast, contrast)
        mean = result.mean()
        result = np.clip((result - mean) * alpha + mean, 0, 255)

    if saturation > 0:
        hsv = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(
            hsv[:, :, 1] * (1 + random.uniform(-saturation, saturation)), 0, 255
        )
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    return result.astype(np.uint8)


def transform_blob(
    blob: Blob, config: AugmentConfig, bg_color: Tuple[int, int, int] = (128, 128, 128)
) -> Optional[Blob]:
    """Apply all enabled transforms to a blob."""

    image = blob.image.copy()

    if config.flip_h_prob > 0 or config.flip_v_prob > 0:
        image = apply_random_flip(image, config.flip_h_prob, config.flip_v_prob)

    if config.rotate_max > 0:
        image = apply_random_rotation(image, config.rotate_max, bg_color)

    if config.scale_range is not None:
        scaled = apply_random_scale(image, config.scale_range)
        if scaled is None:
            return None
        image = scaled

    if config.color_jitter > 0:
        image = apply_color_jitter(
            image, config.color_jitter, config.color_jitter, config.color_jitter
        )

    return Blob(image, blob.class_id)
