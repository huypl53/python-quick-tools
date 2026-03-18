"""Geometry utilities for bounding boxes and collision detection."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from yolo.bg_augment_modules.constants import SPATIAL_GRID_CELL_SIZE


class SpatialGrid:
    """Spatial hash grid for O(1) average-case collision detection."""

    def __init__(self, width: int, height: int, cell_size: int = SPATIAL_GRID_CELL_SIZE):
        """Initialize the spatial grid."""

        self.width = width
        self.height = height
        self.cell_size = max(1, cell_size)
        self.cols = (width + self.cell_size - 1) // self.cell_size
        self.rows = (height + self.cell_size - 1) // self.cell_size
        self.grid: Dict[Tuple[int, int], List[Tuple[int, int, int, int]]] = {}
        self._all_boxes: List[Tuple[int, int, int, int]] = []

    def _get_cells(self, box: Sequence[int]) -> List[Tuple[int, int]]:
        x1, y1, x2, y2 = box
        col1 = max(0, x1 // self.cell_size)
        row1 = max(0, y1 // self.cell_size)
        col2 = min(self.cols - 1, x2 // self.cell_size)
        row2 = min(self.rows - 1, y2 // self.cell_size)
        cells = []
        for row in range(row1, row2 + 1):
            for col in range(col1, col2 + 1):
                cells.append((col, row))
        return cells

    def insert(self, box: Tuple[int, int, int, int]) -> None:
        self._all_boxes.append(box)
        for cell in self._get_cells(box):
            if cell not in self.grid:
                self.grid[cell] = []
            self.grid[cell].append(box)

    def query(self, box: Sequence[int], pad: int = 0) -> List[Tuple[int, int, int, int]]:
        x1, y1, x2, y2 = box
        expanded = (x1 - pad, y1 - pad, x2 + pad, y2 + pad)
        candidates: List[Tuple[int, int, int, int]] = []
        seen: set = set()
        for cell in self._get_cells(expanded):
            for other in self.grid.get(cell, []):
                box_id = id(other)
                if box_id not in seen:
                    seen.add(box_id)
                    candidates.append(other)
        return candidates

    def collides(self, box: Sequence[int], pad: int = 0) -> bool:
        candidates = self.query(box, pad)
        for other in candidates:
            if rects_collide(box, other, pad):
                return True
        return False

    def get_all_boxes(self) -> List[Tuple[int, int, int, int]]:
        return self._all_boxes.copy()


def yolo_to_pixel_box(
    box: Sequence[float], width: int, height: int
) -> Optional[Tuple[int, int, int, int]]:
    """Convert YOLO normalized coordinates to pixel coordinates."""

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
    """Convert pixel coordinates to YOLO normalized format."""

    if width <= 0 or height <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0 / width
    cy = (y1 + y2) / 2.0 / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return cx, cy, w, h


def iou(box_a: Sequence[int], box_b: Sequence[int]) -> float:
    """Calculate Intersection over Union (IoU) between two boxes."""

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


def iou_vectorized(box: Sequence[int], boxes: np.ndarray) -> np.ndarray:
    """Calculate IoU between one box and multiple boxes using vectorization."""

    if boxes.size == 0:
        return np.array([], dtype=np.float64)

    ax1, ay1, ax2, ay2 = box
    bx1 = boxes[:, 0]
    by1 = boxes[:, 1]
    bx2 = boxes[:, 2]
    by2 = boxes[:, 3]

    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a + area_b - inter_area

    iou_vals = np.where(union_area > 0, inter_area / union_area, 0.0)
    return iou_vals


def merge_boxes(
    yolo_boxes: List[Tuple[int, int, int, int]],
    seg_boxes: List[Tuple[int, int, int, int]],
    merge_iou: float,
) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
    """Separate segmentation boxes that overlap with YOLO boxes."""

    remaining_seg: List[Tuple[int, int, int, int]] = []

    if yolo_boxes and len(seg_boxes) > 0:
        yolo_array = np.array(yolo_boxes, dtype=np.int32)
        for seg_box in seg_boxes:
            ious = iou_vectorized(seg_box, yolo_array)
            if not np.any(ious >= merge_iou):
                remaining_seg.append(seg_box)
    else:
        remaining_seg = list(seg_boxes)

    return list(yolo_boxes), remaining_seg


def rects_collide(box_a: Sequence[int], box_b: Sequence[int], pad: int = 0) -> bool:
    """Check if two rectangles collide (overlap)."""

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


def scan_positions(limit: int, step: int) -> List[int]:
    """Generate scan positions for dense placement."""

    if limit < 0:
        return []
    step = max(1, step)
    positions = list(range(0, limit + 1, step))
    if positions and positions[-1] != limit:
        positions.append(limit)
    elif not positions:
        positions = [0]
    return positions


def total_area(boxes: Sequence[Tuple[int, int, int, int]]) -> int:
    """Calculate the total area of all boxes."""

    if not boxes:
        return 0
    arr = np.array(boxes, dtype=np.int64)
    widths = np.maximum(0, arr[:, 2] - arr[:, 0])
    heights = np.maximum(0, arr[:, 3] - arr[:, 1])
    return int(np.sum(widths * heights))
