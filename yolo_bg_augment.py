"""
Augment a YOLO dataset by extracting non-background blobs and rebalancing
selected class boxes via cut-and-paste rearrangement.

Example:
python yolo_bg_augment.py /path/to/yolo_dataset /path/to/output \\
  --classes cat dog --splits train --max-aug-total 500

python yolo_bg_augment.py \
/path/to/yolo_dataset \
/path/to/output \
--classes table shape label \
--class-weights 8 5 1 \
--dense-step 5 --workers 2 \
--dump-bucket-dir /path/to/debug \
--no-seg

"""

# from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TypedDict, Union

import cv2
import numpy as np
import yaml
from tqdm import tqdm
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait


# =============================================================================
# Module-level constants (magic numbers extracted for maintainability)
# =============================================================================

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]

# Color quantization bin size for background estimation
COLOR_QUANT_BIN_SIZE = 16
COLOR_QUANT_BINS = 4096  # 16^3

# Aspect ratio threshold for filtering border-touching contours
ASPECT_RATIO_THRESHOLD = 5.0

# Border tolerance for detecting edge-touching contours
BORDER_TOLERANCE = 3

# Minimum tries for biased placement
MIN_BIASED_PLACEMENT_TRIES = 10

# Maximum consecutive failures before giving up on a class
MAX_AUGMENTATION_FAILURES = 25

# Default grid size for heatmaps
DEFAULT_HEATMAP_GRID_SIZE = 5

# Spatial grid cell size for collision detection
SPATIAL_GRID_CELL_SIZE = 64

# Default maximum placement tries for random placement
DEFAULT_MAX_PLACEMENT_TRIES = 50

# Probability of adding a random seg blob during augmentation
SEG_FILL_PROBABILITY = 0.5


# =============================================================================
# TypedDict definitions for better type safety
# =============================================================================

class WorkerConfigDict(TypedDict):
    """Configuration dictionary passed to worker processes.

    This TypedDict provides proper typing for the worker configuration,
    replacing the generic Dict[str, object] type.
    """
    selected_ids: List[int]
    bucket: Dict[int, List["Blob"]]
    seg_bucket: List["Blob"]
    heatmaps: Dict[int, np.ndarray]
    border_pad: int
    bg_threshold: float
    min_area: int
    merge_iou: float
    drop_rate: float
    collision_pad: int
    keep_background: bool
    placement: str
    dense_step: int
    fill_empty: bool
    fill_ratio: float
    fill_max_blobs: int
    fill_max_tries: int
    dest_root: Path
    seed: int
    no_seg: bool
    feather_radius: int
    scale_range: Optional[Tuple[float, float]]
    rotate_max: float
    flip_h_prob: float
    flip_v_prob: float
    color_jitter: float


class StatsDict(TypedDict, total=False):
    """Statistics dictionary for bucket/dataset stats."""
    version: int
    classes: List[Dict[str, Union[int, str]]]
    heatmap_grid: List[int]
    seg_bucket_count: int
    bucket_dir: str
    dataset_counts: Dict[str, int]
    heatmaps: Dict[str, List[List[int]]]
    bucket_counts: Dict[str, int]


# =============================================================================
# Global worker configuration
# =============================================================================

# Global mutable state for multiprocessing workers.
#
# SAFETY NOTES:
# - This variable is initialized once per worker process via init_worker()
# - Each worker process has its own copy (no cross-process sharing)
# - The main process also initializes this for single-worker mode
# - Always check for None before accessing to catch initialization errors
# - The ProcessPoolExecutor's initializer ensures this is set before work begins
_WORKER_CONFIG: Optional[WorkerConfigDict] = None


@dataclass
class Blob:
    """A cropped image region with optional class label.

    Attributes:
        image: The cropped image data as a numpy array (H, W, C).
        class_id: The YOLO class ID, or None for segmentation-only blobs.
    """
    image: np.ndarray
    class_id: Optional[int]


@dataclass
class AugmentConfig:
    """Configuration for the augmentation pipeline.

    This dataclass consolidates the 14+ parameters passed to augmentation
    functions, making the API cleaner and more maintainable.

    Attributes:
        border_pad: Border padding in pixels for background color sampling.
        bg_threshold: Per-channel tolerance for background masking.
        min_area: Minimum area for color-segment boxes.
        merge_iou: IoU threshold to merge color-segment boxes with YOLO boxes.
        drop_rate: Chance to drop non-target blobs when composing.
        collision_pad: Padding in pixels for collision checking.
        keep_background: Whether to keep original background outside boxes.
        placement: Placement strategy ("dense" or "random").
        dense_step: Pixel step for dense placement scans.
        fill_empty: Whether to fill empty space with extra blobs.
        fill_ratio: Target fill ratio before stopping extra fills.
        fill_max_blobs: Max number of extra blobs to fill per image.
        fill_max_tries: Max placement attempts when filling empty space.
        no_seg: Whether to disable segmentation-based augmentation.
        feather_radius: Edge feathering radius for smooth blending. 0 to disable.
        scale_range: Random scale range (min, max). None to disable.
        rotate_max: Max rotation angle in degrees. 0 to disable.
        flip_h_prob: Horizontal flip probability.
        flip_v_prob: Vertical flip probability.
        color_jitter: Color jitter strength (0-1). 0 to disable.
    """
    border_pad: int = 5
    bg_threshold: float = 30.0
    min_area: int = 200
    merge_iou: float = 0.2
    drop_rate: float = 0.4
    collision_pad: int = 2
    keep_background: bool = False
    placement: str = "dense"
    dense_step: int = 2
    fill_empty: bool = True
    fill_ratio: float = 0.85
    fill_max_blobs: int = 40
    fill_max_tries: int = 500
    no_seg: bool = False
    feather_radius: int = 5
    scale_range: Optional[Tuple[float, float]] = None
    rotate_max: float = 0.0
    flip_h_prob: float = 0.0
    flip_v_prob: float = 0.0
    color_jitter: float = 0.0

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "AugmentConfig":
        """Create an AugmentConfig from parsed command-line arguments."""
        return cls(
            border_pad=args.border_pad,
            bg_threshold=args.bg_threshold,
            min_area=args.min_area,
            merge_iou=args.merge_iou,
            drop_rate=args.drop_rate,
            collision_pad=args.collision_pad,
            keep_background=args.keep_background,
            placement=args.placement,
            dense_step=args.dense_step,
            fill_empty=args.fill_empty,
            fill_ratio=args.fill_ratio,
            fill_max_blobs=args.fill_max_blobs,
            fill_max_tries=args.fill_max_tries,
            no_seg=args.no_seg,
            feather_radius=args.feather_radius,
            scale_range=tuple(args.scale_range) if args.scale_range else None,
            rotate_max=args.rotate_max,
            flip_h_prob=args.flip_h_prob,
            flip_v_prob=args.flip_v_prob,
            color_jitter=args.color_jitter,
        )


class SpatialGrid:
    """Spatial hash grid for O(1) average-case collision detection.

    Instead of checking every placed box for collisions (O(n) per check,
    O(n^2) total), this grid divides the canvas into cells and only checks
    boxes in nearby cells, achieving O(1) average-case performance.

    Attributes:
        width: Canvas width in pixels.
        height: Canvas height in pixels.
        cell_size: Size of each grid cell in pixels.
        grid: Dictionary mapping (cell_x, cell_y) to list of boxes in that cell.
    """

    def __init__(self, width: int, height: int, cell_size: int = SPATIAL_GRID_CELL_SIZE):
        """Initialize the spatial grid.

        Args:
            width: Canvas width in pixels.
            height: Canvas height in pixels.
            cell_size: Size of each grid cell (default SPATIAL_GRID_CELL_SIZE pixels).
        """
        self.width = width
        self.height = height
        self.cell_size = max(1, cell_size)
        self.cols = (width + self.cell_size - 1) // self.cell_size
        self.rows = (height + self.cell_size - 1) // self.cell_size
        self.grid: Dict[Tuple[int, int], List[Tuple[int, int, int, int]]] = {}
        self._all_boxes: List[Tuple[int, int, int, int]] = []

    def _get_cells(self, box: Sequence[int]) -> List[Tuple[int, int]]:
        """Get all grid cells that a box overlaps.

        Args:
            box: Tuple of (x1, y1, x2, y2) pixel coordinates.

        Returns:
            List of (col, row) cell coordinates.
        """
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
        """Insert a box into the spatial grid.

        Args:
            box: Tuple of (x1, y1, x2, y2) pixel coordinates.
        """
        self._all_boxes.append(box)
        for cell in self._get_cells(box):
            if cell not in self.grid:
                self.grid[cell] = []
            self.grid[cell].append(box)

    def query(self, box: Sequence[int], pad: int = 0) -> List[Tuple[int, int, int, int]]:
        """Query all boxes that might collide with the given box.

        Args:
            box: Tuple of (x1, y1, x2, y2) pixel coordinates.
            pad: Additional padding to expand the query region.

        Returns:
            List of potentially colliding boxes (may include false positives).
        """
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
        """Check if a box collides with any existing box.

        Args:
            box: Tuple of (x1, y1, x2, y2) pixel coordinates.
            pad: Additional padding for collision checking.

        Returns:
            True if any collision is detected, False otherwise.
        """
        candidates = self.query(box, pad)
        for other in candidates:
            if rects_collide(box, other, pad):
                return True
        return False

    def get_all_boxes(self) -> List[Tuple[int, int, int, int]]:
        """Get all boxes in the grid.

        Returns:
            List of all inserted boxes.
        """
        return self._all_boxes.copy()


def validate_image(
    image: Optional[np.ndarray], source: str = "image"
) -> Optional[np.ndarray]:
    """Validate that an image was loaded correctly and has valid dimensions.

    Args:
        image: The image array from cv2.imread, or None if loading failed.
        source: Description of the image source for error messages.

    Returns:
        The validated image, or None if invalid.
    """
    if image is None:
        return None
    if image.ndim < 2:
        print(f"[WARN] Invalid image dimensions for {source}: ndim={image.ndim}")
        return None
    if image.ndim == 2:
        # Grayscale image - convert to BGR for consistency
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] not in (3, 4):
        print(f"[WARN] Unexpected channel count for {source}: {image.shape[2]}")
        return None
    if image.shape[0] == 0 or image.shape[1] == 0:
        print(f"[WARN] Zero-dimension image for {source}: {image.shape}")
        return None
    # Convert RGBA to BGR if needed
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


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
        "--mode",
        choices=["bucket", "augment"],
        default="augment",
        help="Run mode: build bucket only or augment dataset (default: augment).",
    )
    parser.add_argument(
        "--bucket-dir",
        type=Path,
        default=None,
        help="Bucket directory to write/read (default: <dest> in bucket mode).",
    )
    parser.add_argument(
        "--bucket-stats",
        type=Path,
        default=None,
        help="Optional JSON stats path to save/load bucket/dataset stats.",
    )
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
        default=5,
        help="Border padding in pixels used to sample background color. Default: 5.",
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
        default=False,
        help="Fill empty space with extra segmentation blobs (default: disabled).",
    )
    parser.add_argument(
        "--no-fill-empty",
        action="store_true",
        default=False,
        help="Disable filling empty space (overrides --fill-empty).",
    )
    parser.add_argument(
        "--fill-ratio",
        type=float,
        default=0.85,
        help="Target fill ratio before stopping extra fills. Default: 0.85.",
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
        default=500,
        help="Max placement attempts when filling empty space. Default: 500.",
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
        "--dump-bucket-dir",
        type=Path,
        default=None,
        help="Optional directory to save bucket crops for debugging.",
    )
    parser.add_argument(
        "--dump-bucket-limit",
        type=int,
        default=0,
        help="Max crops to save per class/seg bucket (0 = no limit).",
    )
    parser.add_argument(
        "--no-seg",
        action="store_true",
        help="Disable segmentation-based augmentation; only use specified class bboxes.",
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
    # Transform arguments
    parser.add_argument(
        "--feather-radius",
        type=int,
        default=5,
        help="Edge feathering radius for blending. 0 to disable. Default: 5.",
    )
    parser.add_argument(
        "--scale-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="Random scale range (e.g., 0.7 1.3). Default: disabled.",
    )
    parser.add_argument(
        "--rotate-max",
        type=float,
        default=0.0,
        help="Max rotation angle in degrees. 0 to disable. Default: 0.",
    )
    parser.add_argument(
        "--flip-h-prob",
        type=float,
        default=0.0,
        help="Horizontal flip probability. Default: 0.",
    )
    parser.add_argument(
        "--flip-v-prob",
        type=float,
        default=0.0,
        help="Vertical flip probability. Default: 0.",
    )
    parser.add_argument(
        "--color-jitter",
        type=float,
        default=0.0,
        help="Color jitter strength (0-1). 0 to disable. Default: 0.",
    )
    args = parser.parse_args()

    # Reconcile --fill-empty and --no-fill-empty flags
    # --no-fill-empty takes precedence if set
    if args.no_fill_empty:
        args.fill_empty = False

    return args


def load_class_names(data_yaml: Path) -> List[str]:
    """Load class names from a YOLO data.yaml file.

    Args:
        data_yaml: Path to the data.yaml file.

    Returns:
        List of class names.

    Raises:
        OSError: If the file cannot be read.
        yaml.YAMLError: If the file is not valid YAML.
        ValueError: If the file does not contain a 'names' list.
    """
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
    """Copy metadata files from source to destination.

    Args:
        src_root: Source dataset root directory.
        dest_root: Destination dataset root directory.
    """
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
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Path to the directory.

    Returns:
        True if the directory exists or was created, False on error.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except OSError as e:
        print(f"[WARN] Failed to create directory {path}: {e}")
        return False


def sanitize_name(name: str) -> str:
    """Sanitize a string to be used as a filename.

    Replaces non-alphanumeric characters (except underscore, period, hyphen)
    with underscores.

    Args:
        name: The string to sanitize.

    Returns:
        A sanitized string safe for use as a filename.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return cleaned or "class"


def list_image_files(folder: Path) -> List[Path]:
    """List all image files in a folder.

    Args:
        folder: Path to the folder to scan.

    Returns:
        Sorted list of image file paths.

    Raises:
        OSError: If the folder cannot be read.
    """
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
    """Find an image file by stem name, trying all supported extensions.

    Args:
        images_dir: Directory to search for the image.
        stem: Base filename without extension.

    Returns:
        Path to the found image, or None if not found.
    """
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def iter_label_images(
    source_root: Path, splits: Sequence[str]
) -> Iterable[Tuple[str, Path, Path]]:
    """Iterate over all label/image pairs in the specified dataset splits.

    Args:
        source_root: Root directory of the YOLO dataset.
        splits: List of splits to process (e.g., ['train', 'val']).

    Yields:
        Tuples of (split_name, image_path, label_path) for each valid pair.
    """
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
    """Parse a single line from a YOLO label file.

    Args:
        line: A line from a YOLO label file.

    Returns:
        Tuple of (class_id, center_x, center_y, width, height) in YOLO format,
        or None if the line is invalid.
    """
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
    """Convert YOLO normalized coordinates to pixel coordinates.

    Args:
        box: YOLO format as (center_x, center_y, width, height), normalized 0-1.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Pixel coordinates as (x1, y1, x2, y2), or None if the box is invalid.
    """
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
    """Convert pixel coordinates to YOLO normalized format.

    Args:
        box: Pixel coordinates as (x1, y1, x2, y2).
        width: Image width for normalization.
        height: Image height for normalization.

    Returns:
        YOLO format as (center_x, center_y, width, height), all normalized 0-1.
        Returns (0, 0, 0, 0) if width or height is zero.
    """
    # Guard against division by zero
    if width <= 0 or height <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0 / width
    cy = (y1 + y2) / 2.0 / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return cx, cy, w, h


def estimate_background_color(image: np.ndarray, pad: int) -> Tuple[int, int, int]:
    """Estimate the background color by sampling border pixels.

    Uses color quantization to find the dominant color in the border region,
    then returns the mean of pixels matching that quantized color.

    Args:
        image: Input image as numpy array (H, W, C).
        pad: Border padding in pixels to sample.

    Returns:
        Tuple of (B, G, R) color values.
    """
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        return (0, 0, 0)
    pad = max(1, min(pad, min(height, width) // 2))

    # Direct indexing instead of creating an 8MB boolean mask
    # Collect border strips: top, bottom, left (excluding corners), right (excluding corners)
    top = image[:pad, :].reshape(-1, image.shape[2])
    bottom = image[-pad:, :].reshape(-1, image.shape[2])
    left = image[pad:-pad, :pad].reshape(-1, image.shape[2]) if height > 2 * pad else np.empty((0, image.shape[2]))
    right = image[pad:-pad, -pad:].reshape(-1, image.shape[2]) if height > 2 * pad else np.empty((0, image.shape[2]))

    samples = np.vstack([top, bottom, left, right]) if left.size > 0 or right.size > 0 else np.vstack([top, bottom])

    if samples.size == 0:
        return int(image[0, 0, 0]), int(image[0, 0, 1]), int(image[0, 0, 2])

    # Quantize colors into bins
    quant = (samples // COLOR_QUANT_BIN_SIZE).astype(np.int32)
    keys = (quant[:, 0] * COLOR_QUANT_BIN_SIZE + quant[:, 1]) * COLOR_QUANT_BIN_SIZE + quant[:, 2]
    counts = np.bincount(keys, minlength=COLOR_QUANT_BINS)
    dominant = int(np.argmax(counts))

    # Find pixels matching the dominant quantized color
    target = np.array(
        [dominant // 256, (dominant // COLOR_QUANT_BIN_SIZE) % COLOR_QUANT_BIN_SIZE, dominant % COLOR_QUANT_BIN_SIZE],
        dtype=np.int32,
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
    """Segment foreground objects and return their bounding boxes.

    Finds connected components of non-background pixels and filters out
    elongated border-touching artifacts.

    Args:
        image: Input image as numpy array (H, W, C).
        bg_color: Background color as (B, G, R) tuple.
        threshold: Per-channel tolerance for background detection.
        min_area: Minimum contour area to include.

    Returns:
        List of bounding boxes as (x1, y1, x2, y2) tuples.
    """
    mask = build_foreground_mask(image, bg_color, threshold)
    h, w = mask.shape
    if h == 0 or w == 0:
        return []
    mask = cv2.rectangle(mask, (0, 0), (w, h), 0, thickness=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(mask)
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        # Guard against division by zero
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


def build_foreground_mask(
    image: np.ndarray, bg_color: Tuple[int, int, int], threshold: float
) -> np.ndarray:
    """Create a binary mask of foreground (non-background) pixels.

    Args:
        image: Input image as numpy array (H, W, C).
        bg_color: Background color as (B, G, R) tuple.
        threshold: Per-channel tolerance for background detection.

    Returns:
        Binary mask where 255 = foreground, 0 = background.
    """
    tol = int(round(threshold))
    lower = np.clip(np.array(bg_color, dtype=np.int16) - tol, 0, 255).astype(np.uint8)
    upper = np.clip(np.array(bg_color, dtype=np.int16) + tol, 0, 255).astype(np.uint8)
    bg_mask = cv2.inRange(image, lower, upper)
    return cv2.bitwise_not(bg_mask)


# =============================================================================
# Transform Functions (Feathering, Scale, Rotation, Flip, Color Jitter)
# =============================================================================


def create_feathered_mask(blob_shape: Tuple[int, int], feather_radius: int = 5) -> np.ndarray:
    """Create a feathered alpha mask for smooth blending.

    Args:
        blob_shape: Tuple of (height, width) for the blob.
        feather_radius: Radius for edge feathering in pixels.

    Returns:
        Feathered mask as float32 array with values in [0, 1].
    """
    h, w = blob_shape
    mask = np.ones((h, w), dtype=np.float32)

    # Create distance from edges
    for i in range(min(feather_radius, h // 2, w // 2)):
        alpha = (i + 1) / feather_radius
        mask[i, :] = np.minimum(mask[i, :], alpha)
        mask[h - 1 - i, :] = np.minimum(mask[h - 1 - i, :], alpha)
        mask[:, i] = np.minimum(mask[:, i], alpha)
        mask[:, w - 1 - i] = np.minimum(mask[:, w - 1 - i], alpha)

    # Apply Gaussian blur for smoother transitions
    ksize = feather_radius * 2 + 1
    return cv2.GaussianBlur(mask, (ksize, ksize), 0)


def paste_blob_with_blending(
    canvas: np.ndarray, blob: np.ndarray, x: int, y: int, feather_radius: int = 5
) -> None:
    """Paste blob onto canvas with feathered edges (modifies canvas in place).

    Args:
        canvas: The canvas image to paste onto.
        blob: The blob image to paste.
        x: X coordinate for placement.
        y: Y coordinate for placement.
        feather_radius: Radius for edge feathering.
    """
    h, w = blob.shape[:2]
    mask = create_feathered_mask((h, w), feather_radius)
    mask = mask[:, :, np.newaxis]  # Add channel dimension

    roi = canvas[y:y+h, x:x+w].astype(np.float32)
    blob_f = blob.astype(np.float32)
    blended = roi * (1 - mask) + blob_f * mask
    canvas[y:y+h, x:x+w] = blended.astype(np.uint8)


def apply_random_scale(
    blob: np.ndarray, scale_range: Tuple[float, float] = (0.7, 1.3), min_size: int = 16
) -> Optional[np.ndarray]:
    """Apply random scaling to blob.

    Args:
        blob: Input image as numpy array.
        scale_range: Tuple of (min_scale, max_scale).
        min_size: Minimum dimension size after scaling.

    Returns:
        Scaled image, or None if result is too small.
    """
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
    """Apply random rotation to blob.

    Args:
        blob: Input image as numpy array.
        max_angle: Maximum rotation angle in degrees.
        bg_color: Background color for filled areas.

    Returns:
        Rotated image with expanded bounds to fit content.
    """
    angle = random.uniform(-max_angle, max_angle)
    h, w = blob.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new bounds
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    return cv2.warpAffine(
        blob, M, (new_w, new_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=bg_color
    )


def apply_random_flip(blob: np.ndarray, h_prob: float = 0.5, v_prob: float = 0.0) -> np.ndarray:
    """Apply random horizontal/vertical flip.

    Args:
        blob: Input image as numpy array.
        h_prob: Probability of horizontal flip.
        v_prob: Probability of vertical flip.

    Returns:
        Potentially flipped image.
    """
    if random.random() < h_prob:
        blob = cv2.flip(blob, 1)
    if random.random() < v_prob:
        blob = cv2.flip(blob, 0)
    return blob


def apply_color_jitter(
    blob: np.ndarray, brightness: float = 0.2, contrast: float = 0.2, saturation: float = 0.2
) -> np.ndarray:
    """Apply random color jitter to blob.

    Args:
        blob: Input image as numpy array (BGR format).
        brightness: Brightness jitter factor (0-1).
        contrast: Contrast jitter factor (0-1).
        saturation: Saturation jitter factor (0-1).

    Returns:
        Color-jittered image.
    """
    result = blob.astype(np.float32)

    # Brightness
    if brightness > 0:
        beta = random.uniform(-brightness, brightness) * 255
        result = np.clip(result + beta, 0, 255)

    # Contrast
    if contrast > 0:
        alpha = 1 + random.uniform(-contrast, contrast)
        mean = result.mean()
        result = np.clip((result - mean) * alpha + mean, 0, 255)

    # Saturation
    if saturation > 0:
        hsv = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + random.uniform(-saturation, saturation)), 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    return result.astype(np.uint8)


def transform_blob(
    blob: "Blob",
    config: "AugmentConfig",
    bg_color: Tuple[int, int, int] = (128, 128, 128)
) -> Optional["Blob"]:
    """Apply all enabled transforms to a blob.

    Args:
        blob: The Blob object to transform.
        config: AugmentConfig with transform settings.
        bg_color: Background color for rotation fill.

    Returns:
        Transformed Blob, or None if transform results in invalid size.
    """
    image = blob.image.copy()

    # Geometric transforms
    if config.flip_h_prob > 0 or config.flip_v_prob > 0:
        image = apply_random_flip(image, config.flip_h_prob, config.flip_v_prob)

    if config.rotate_max > 0:
        image = apply_random_rotation(image, config.rotate_max, bg_color)

    if config.scale_range is not None:
        scaled = apply_random_scale(image, config.scale_range)
        if scaled is None:
            return None
        image = scaled

    # Color transforms
    if config.color_jitter > 0:
        image = apply_color_jitter(image, config.color_jitter, config.color_jitter, config.color_jitter)

    return Blob(image, blob.class_id)


def iou(box_a: Sequence[int], box_b: Sequence[int]) -> float:
    """Calculate Intersection over Union (IoU) between two boxes.

    Args:
        box_a: First box as (x1, y1, x2, y2).
        box_b: Second box as (x1, y1, x2, y2).

    Returns:
        IoU value between 0.0 and 1.0.
    """
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
    # Guard against division by zero
    if denom <= 0:
        return 0.0
    return inter_area / denom


def iou_vectorized(
    box: Sequence[int], boxes: np.ndarray
) -> np.ndarray:
    """Calculate IoU between one box and multiple boxes using vectorization.

    This is significantly faster than calling iou() in a loop when checking
    against many boxes.

    Args:
        box: Single box as (x1, y1, x2, y2).
        boxes: Array of boxes with shape (N, 4), each row is (x1, y1, x2, y2).

    Returns:
        Array of IoU values with shape (N,).
    """
    if boxes.size == 0:
        return np.array([], dtype=np.float64)

    ax1, ay1, ax2, ay2 = box
    bx1 = boxes[:, 0]
    by1 = boxes[:, 1]
    bx2 = boxes[:, 2]
    by2 = boxes[:, 3]

    # Intersection coordinates
    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)

    # Intersection area (clipped to 0)
    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Union area
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a + area_b - inter_area

    # IoU with division by zero guard
    iou_vals = np.where(union_area > 0, inter_area / union_area, 0.0)
    return iou_vals


def merge_boxes(
    yolo_boxes: List[Tuple[int, int, int, int]],
    seg_boxes: List[Tuple[int, int, int, int]],
    merge_iou: float,
) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
    """Separate segmentation boxes that overlap with YOLO boxes.

    Args:
        yolo_boxes: List of YOLO detection boxes.
        seg_boxes: List of segmentation-derived boxes.
        merge_iou: IoU threshold above which a seg box is considered overlapping.

    Returns:
        Tuple of (yolo_boxes, remaining_seg_boxes).
    """
    remaining_seg: List[Tuple[int, int, int, int]] = []

    # Use vectorized IoU if we have many boxes
    if yolo_boxes and len(seg_boxes) > 0:
        yolo_array = np.array(yolo_boxes, dtype=np.int32)
        for seg_box in seg_boxes:
            ious = iou_vectorized(seg_box, yolo_array)
            # Use the merge_iou threshold (not hardcoded > 0.0)
            if not np.any(ious >= merge_iou):
                remaining_seg.append(seg_box)
    else:
        remaining_seg = list(seg_boxes)

    return list(yolo_boxes), remaining_seg


def rects_collide(box_a: Sequence[int], box_b: Sequence[int], pad: int = 0) -> bool:
    """Check if two rectangles collide (overlap).

    Args:
        box_a: First box as (x1, y1, x2, y2).
        box_b: Second box as (x1, y1, x2, y2).
        pad: Additional padding to expand both boxes before checking.

    Returns:
        True if the boxes overlap, False otherwise.
    """
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
    placed: Union[List[Tuple[int, int, int, int]], SpatialGrid],
    collision_pad: int = 0,
    max_tries: int = DEFAULT_MAX_PLACEMENT_TRIES,
    feather_radius: int = 0,
) -> Optional[Tuple[int, int, int, int]]:
    """Place a blob at a random non-colliding position on the canvas.

    Args:
        canvas: The canvas image to place the blob on.
        blob: The blob to place.
        placed: Either a list of placed boxes or a SpatialGrid for collision detection.
        collision_pad: Padding in pixels for collision checking.
        max_tries: Maximum random placement attempts.
        feather_radius: Edge feathering radius for blending. 0 for direct paste.

    Returns:
        The placed box as (x1, y1, x2, y2), or None if placement failed.
    """
    height, width = canvas.shape[:2]
    blob_h, blob_w = blob.image.shape[:2]
    if blob_h >= height or blob_w >= width:
        return None

    # Use SpatialGrid for O(1) collision detection if available
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


def scan_positions(limit: int, step: int) -> List[int]:
    """Generate scan positions for dense placement.

    Creates a list of positions from 0 to limit (inclusive) at the given step,
    always including 0 and the limit value.

    Args:
        limit: Maximum position value (inclusive).
        step: Step size between positions.

    Returns:
        List of position values.
    """
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
    placed: Union[List[Tuple[int, int, int, int]], SpatialGrid],
    collision_pad: int = 0,
    step: int = 2,
    heatmap: Optional[np.ndarray] = None,
    feather_radius: int = 0,
) -> Optional[Tuple[int, int, int, int]]:
    """Place a blob using dense scanning to find optimal position.

    Scans the canvas in a grid pattern to find a non-colliding position.
    If a heatmap is provided, prefers positions with lower heatmap scores.

    Args:
        canvas: The canvas image to place the blob on.
        blob: The blob to place.
        placed: Either a list of placed boxes or a SpatialGrid for collision detection.
        collision_pad: Padding in pixels for collision checking.
        step: Pixel step for dense placement scans.
        heatmap: Optional heatmap for biased placement.
        feather_radius: Edge feathering radius for blending. 0 for direct paste.

    Returns:
        The placed box as (x1, y1, x2, y2), or None if placement failed.
    """
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

    # Use SpatialGrid for O(1) collision detection if available
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
            # Guard against division by zero in grid calculations
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
        # Guard against division by zero in grid calculations
        if width > 0 and height > 0 and grid_w > 0 and grid_h > 0:
            grid_x = min(grid_w - 1, max(0, int(cx / width * grid_w)))
            grid_y = min(grid_h - 1, max(0, int(cy / height * grid_h)))
        else:
            grid_x = grid_y = 0
        heatmap[grid_y, grid_x] += 1
        return best
    return None


def total_area(boxes: Sequence[Tuple[int, int, int, int]]) -> int:
    """Calculate the total area of all boxes.

    Uses numpy vectorization for efficiency with large box lists.

    Args:
        boxes: Sequence of boxes as (x1, y1, x2, y2) tuples.

    Returns:
        Total area in pixels.
    """
    if not boxes:
        return 0
    # Use numpy vectorization for efficiency
    arr = np.array(boxes, dtype=np.int64)
    widths = np.maximum(0, arr[:, 2] - arr[:, 0])
    heights = np.maximum(0, arr[:, 3] - arr[:, 1])
    return int(np.sum(widths * heights))


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
    """Fill empty canvas space with extra segmentation blobs.

    Args:
        canvas: The canvas image to fill.
        placed: Either a list of placed boxes or a SpatialGrid.
        seg_bucket: List of segmentation blobs to sample from.
        collision_pad: Padding in pixels for collision checking.
        placement: Placement strategy ("dense" or "random").
        dense_step: Pixel step for dense placement.
        fill_ratio: Target fill ratio before stopping.
        fill_max_blobs: Maximum blobs to add.
        fill_max_tries: Maximum placement attempts.
        feather_radius: Edge feathering radius for blending. 0 for direct paste.
    """
    if not seg_bucket:
        return
    height, width = canvas.shape[:2]
    # Guard against division by zero
    if width == 0 or height == 0:
        return
    canvas_area = float(width * height)
    target_ratio = max(0.0, min(1.0, fill_ratio))
    added = 0
    tries = 0

    # Get boxes for area calculation
    if isinstance(placed, SpatialGrid):
        boxes_for_area = placed.get_all_boxes()
    else:
        boxes_for_area = placed

    while added < fill_max_blobs and tries < fill_max_tries:
        # Recalculate area with current boxes
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
    """Place a blob with bias toward low-density heatmap regions.

    Samples random positions and selects the one with the lowest heatmap score,
    encouraging more uniform spatial distribution of objects.

    Args:
        canvas: The canvas image to place the blob on.
        blob: The blob to place.
        placed: Either a list of placed boxes or a SpatialGrid.
        heatmap: Heatmap array for biased placement.
        collision_pad: Padding in pixels for collision checking.
        max_tries: Maximum random placement attempts.
        feather_radius: Edge feathering radius for blending. 0 for direct paste.

    Returns:
        The placed box as (x1, y1, x2, y2), or None if placement failed.
    """
    height, width = canvas.shape[:2]
    blob_h, blob_w = blob.image.shape[:2]
    if blob_h >= height or blob_w >= width:
        return None
    grid_h, grid_w = heatmap.shape
    # Guard against division by zero
    if width == 0 or height == 0 or grid_w == 0 or grid_h == 0:
        return None
    candidates: List[Tuple[int, int, int, int, int]] = []
    tries = max(MIN_BIASED_PLACEMENT_TRIES, max_tries)

    # Use SpatialGrid for O(1) collision detection if available
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
            canvas, blob, placed, collision_pad=collision_pad, max_tries=max_tries,
            feather_radius=feather_radius
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


def collect_counts(
    entries: Sequence[Tuple[str, Path, Path]],
    selected_ids: Iterable[int],
) -> Dict[int, int]:
    """Count occurrences of selected classes across label files.

    Args:
        entries: List of (split, image_path, label_path) tuples.
        selected_ids: Class IDs to count.

    Returns:
        Dictionary mapping class ID to count.
    """
    selected = set(selected_ids)
    counts = {class_id: 0 for class_id in selected}
    for _, _, label_path in tqdm(entries, desc="Counting labels", unit="image"):
        try:
            with label_path.open("r", encoding="utf-8") as f:
                for line in f:
                    parsed = parse_label_line(line)
                    if not parsed:
                        continue
                    class_id, x, y, w, h = parsed
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
    """Collect spatial heatmaps showing class distribution across images.

    Args:
        entries: List of (split, image_path, label_path) tuples.
        selected_ids: Class IDs to track.
        grid_w: Heatmap grid width.
        grid_h: Heatmap grid height.

    Returns:
        Dictionary mapping class ID to heatmap array.
    """
    selected = set(selected_ids)
    heatmaps: Dict[int, np.ndarray] = {
        class_id: np.zeros((grid_h, grid_w), dtype=np.int32)
        for class_id in selected
    }
    for _, image_path, label_path in tqdm(entries, desc="Building heatmaps", unit="image"):
        image = cv2.imread(str(image_path))
        image = validate_image(image, str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        # Guard against division by zero
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
    """Print location heatmaps for each class to stdout.

    Args:
        heatmaps: Dictionary mapping class ID to heatmap arrays.
        classes: List of class names to print.
        name_to_idx: Dictionary mapping class name to class ID.
    """
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
    no_seg: bool = False,
) -> Tuple[Dict[int, List[Blob]], List[Blob]]:
    """Build blob buckets by extracting crops from dataset images.

    Processes images to extract YOLO-labeled objects and segmentation-derived
    foreground objects into separate buckets for later augmentation use.

    Args:
        entries: List of (split, image_path, label_path) tuples to process.
        selected_ids: Class IDs to extract into the bucket.
        border_pad: Border padding for background color estimation.
        bg_threshold: Per-channel tolerance for background masking.
        min_area: Minimum area for color-segment boxes.
        merge_iou: IoU threshold to merge seg boxes with YOLO boxes.
        max_per_class: Maximum blobs to store per class.
        max_seg_bucket: Maximum segmentation blobs to store.
        bucket_max_images: Maximum images to process (0 = no limit).
        bucket_sample_rate: Sampling rate for bucket images (0-1].
        debug: Whether to save debug visualization masks.
        debug_dir: Directory for debug output.
        debug_max: Maximum debug images to save.
        no_seg: Whether to skip segmentation-based extraction.

    Returns:
        Tuple of (class_bucket, seg_bucket) where class_bucket maps class ID
        to list of Blobs, and seg_bucket is a list of segmentation Blobs.
    """
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
    """Dump bucket blobs to disk for debugging inspection.

    This is a convenience wrapper around save_bucket_dir that discards the
    return value, used when only the side effect (saving files) is needed.

    Args:
        bucket: Dictionary mapping class ID to list of blobs.
        seg_bucket: List of segmentation blobs.
        class_names: List of class names for folder naming.
        output_dir: Output directory path.
        limit: Maximum blobs to save per class (0 = no limit).
    """
    save_bucket_dir(bucket, seg_bucket, class_names, output_dir, limit=limit)


def save_bucket_dir(
    bucket: Dict[int, List[Blob]],
    seg_bucket: List[Blob],
    class_names: Sequence[str],
    output_dir: Path,
    limit: int = 0,
) -> Tuple[Dict[int, int], int]:
    """Save bucket blobs to disk as images.

    Args:
        bucket: Dictionary mapping class ID to list of blobs.
        seg_bucket: List of segmentation blobs.
        class_names: List of class names for folder naming.
        output_dir: Output directory path.
        limit: Maximum blobs to save per class (0 = no limit).

    Returns:
        Tuple of (saved_counts per class, seg_saved count).
    """
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
    """Load blob buckets from a previously saved bucket directory.

    Reads images from the bucket directory structure and reconstructs the
    bucket and seg_bucket data structures.

    Args:
        bucket_dir: Directory containing the saved bucket.
        selected_ids: Class IDs to load from the bucket.
        class_names: List of class names for folder name matching.
        max_per_class: Maximum blobs to load per class (0 = no limit).
        max_seg_bucket: Maximum segmentation blobs to load (0 = no limit).

    Returns:
        Tuple of (class_bucket, seg_bucket) matching build_bucket's return type.
    """
    bucket: Dict[int, List[Blob]] = {class_id: [] for class_id in selected_ids}
    seg_bucket: List[Blob] = []
    classes_dir = bucket_dir / "classes"
    seg_dir = bucket_dir / "seg"
    if classes_dir.exists():
        class_dirs = [path for path in classes_dir.iterdir() if path.is_dir()]
    else:
        class_dirs = [
            path
            for path in bucket_dir.iterdir()
            if path.is_dir() and path.name != "seg"
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
    """Print bucket statistics to stdout.

    Args:
        bucket_counts: Dictionary mapping class ID to blob count.
        seg_count: Number of segmentation blobs.
        class_names: List of class names for display.
        selected_ids: Class IDs to include in the report.
    """
    print("Bucket stats (saved items):")
    for class_id in selected_ids:
        name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        print(f"  {name}: {bucket_counts.get(class_id, 0)}")
    print(f"  seg: {seg_count}")


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
    """Build a statistics dictionary for JSON serialization.

    Args:
        selected_ids: Class IDs included in the stats.
        class_names: List of class names.
        counts: Optional dataset counts per class.
        heatmaps: Optional heatmap arrays per class.
        grid_w: Heatmap grid width.
        grid_h: Heatmap grid height.
        bucket_counts: Optional bucket blob counts per class.
        seg_count: Number of segmentation blobs.
        bucket_dir: Optional bucket directory path.

    Returns:
        Dictionary ready for JSON serialization.
    """
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
    """Save statistics to a JSON file.

    Args:
        stats: Statistics dictionary to save.
        output_path: Path to the output JSON file.

    Raises:
        OSError: If the file cannot be written.
        TypeError: If the stats contain non-serializable data.
    """
    try:
        ensure_dir(output_path.parent)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, sort_keys=True)
    except (OSError, TypeError) as e:
        print(f"[WARN] Failed to save stats to {output_path}: {e}")
        raise


def load_stats(stats_path: Path) -> Dict[str, object]:
    """Load statistics from a JSON file.

    Args:
        stats_path: Path to the JSON file.

    Returns:
        The loaded statistics dictionary.

    Raises:
        OSError: If the file cannot be read.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    try:
        with stats_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[WARN] Failed to load stats from {stats_path}: {e}")
        raise


def parse_stats_counts(
    stats: Dict[str, object], selected_ids: Sequence[int]
) -> Optional[Dict[int, int]]:
    """Parse dataset counts from a stats dictionary.

    Args:
        stats: Statistics dictionary loaded from JSON.
        selected_ids: Class IDs to extract counts for.

    Returns:
        Dictionary mapping class ID to count, or None if counts are missing.
    """
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
    """Parse heatmaps from a stats dictionary.

    Args:
        stats: Statistics dictionary loaded from JSON.
        selected_ids: Class IDs to extract heatmaps for.

    Returns:
        Dictionary mapping class ID to heatmap array, or None if missing.
    """
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


def copy_original_dataset(
    source_root: Path, dest_root: Path, splits: Sequence[str]
) -> None:
    """Copy the original dataset to the destination directory.

    Copies all images and labels from the specified splits, preserving
    directory structure.

    Args:
        source_root: Source dataset root directory.
        dest_root: Destination dataset root directory.
        splits: List of splits to copy (e.g., ['train', 'val']).
    """
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
    """Create a canvas for blob placement.

    Either preserves the original background with boxes cleared, or creates
    a solid-color canvas for fresh blob arrangement.

    Args:
        image: Original image.
        yolo_boxes: YOLO-labeled boxes to clear.
        seg_boxes: Segmentation boxes to clear.
        bg_color: Background color as (B, G, R).
        keep_background: If True, keep original background outside boxes.

    Returns:
        Canvas image ready for blob placement.
    """
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
    """Extract blobs from an image given a list of bounding boxes.

    Args:
        image: Source image to crop from.
        boxes: List of boxes as (x1, y1, x2, y2) tuples.
        class_id: Class ID to assign to all extracted blobs.

    Returns:
        List of Blob objects extracted from the image.
    """
    blobs: List[Blob] = []
    for box in boxes:
        x1, y1, x2, y2 = box
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        blobs.append(Blob(crop, class_id))
    return blobs


def pick_random_blob(bucket: Dict[int, List[Blob]], class_id: int) -> Optional[Blob]:
    """Pick a random blob of the specified class from the bucket.

    Args:
        bucket: Dictionary mapping class ID to list of blobs.
        class_id: Class ID to pick from.

    Returns:
        A randomly selected Blob, or None if no blobs of that class exist.
    """
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
    no_seg: bool = False,
    feather_radius: int = 0,
    scale_range: Optional[Tuple[float, float]] = None,
    rotate_max: float = 0.0,
    flip_h_prob: float = 0.0,
    flip_v_prob: float = 0.0,
    color_jitter: float = 0.0,
) -> Optional[Tuple[np.ndarray, List[Tuple[int, int, int, int, int]]]]:
    """Build a single augmented image by rearranging blobs.

    Takes an existing image, extracts its objects, adds a target class blob
    from the bucket, and composes them onto a fresh canvas without collisions.

    Args:
        image_path: Path to the source image.
        label_path: Path to the YOLO label file.
        selected_ids: Class IDs to process.
        bucket: Dictionary mapping class ID to list of available blobs.
        seg_bucket: List of segmentation blobs for filling.
        target_class_id: Class ID of the blob to add.
        heatmaps: Spatial heatmaps for biased placement.
        border_pad: Border padding for background color estimation.
        bg_threshold: Per-channel tolerance for background masking.
        min_area: Minimum area for color-segment boxes.
        merge_iou: IoU threshold to merge seg boxes with YOLO boxes.
        drop_rate: Probability to drop non-target blobs.
        collision_pad: Padding for collision detection.
        keep_background: Whether to keep original background.
        placement: Placement strategy ("dense" or "random").
        dense_step: Pixel step for dense placement.
        fill_empty: Whether to fill remaining space with seg blobs.
        fill_ratio: Target fill ratio for empty space filling.
        fill_max_blobs: Maximum blobs to add when filling.
        fill_max_tries: Maximum placement attempts when filling.
        no_seg: Whether to skip segmentation-based processing.
        feather_radius: Edge feathering radius for blending. 0 to disable.
        scale_range: Random scale range (min, max). None to disable.
        rotate_max: Max rotation angle in degrees. 0 to disable.
        flip_h_prob: Horizontal flip probability.
        flip_v_prob: Vertical flip probability.
        color_jitter: Color jitter strength (0-1). 0 to disable.

    Returns:
        Tuple of (canvas, labels) where labels is a list of
        (class_id, x1, y1, x2, y2) tuples, or None on failure.
    """
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

    # Create a temporary config for transform_blob
    # Check if any transforms are enabled
    transforms_enabled = (
        scale_range is not None or
        rotate_max > 0 or
        flip_h_prob > 0 or
        flip_v_prob > 0 or
        color_jitter > 0
    )

    if transforms_enabled:
        # Create a minimal config for transforms
        transform_config = AugmentConfig(
            scale_range=scale_range,
            rotate_max=rotate_max,
            flip_h_prob=flip_h_prob,
            flip_v_prob=flip_v_prob,
            color_jitter=color_jitter,
        )
        # Transform the target blob
        transformed_target = transform_blob(target_blob, transform_config, bg_color)
        if transformed_target is None:
            return None
        target_blob = transformed_target

        # Optionally transform other blobs with 50% probability
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

    # Use SpatialGrid for O(1) collision detection instead of O(n^2) list iteration
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
                canvas, target_blob, spatial_grid,
                collision_pad=collision_pad, feather_radius=feather_radius
            )
        else:
            first = place_blob_biased(
                canvas, target_blob, spatial_grid, target_heat,
                collision_pad=collision_pad, feather_radius=feather_radius
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
                    canvas, blob, spatial_grid,
                    collision_pad=collision_pad, feather_radius=feather_radius
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


def write_label_file(
    path: Path, labels: List[Tuple[int, int, int, int, int]], width: int, height: int
) -> bool:
    """Write YOLO format label file.

    Args:
        path: Path to the output label file.
        labels: List of (class_id, x1, y1, x2, y2) tuples.
        width: Image width for coordinate normalization.
        height: Image height for coordinate normalization.

    Returns:
        True if successful, False on error.
    """
    try:
        with path.open("w", encoding="utf-8") as f:
            for class_id, x1, y1, x2, y2 in labels:
                cx, cy, w, h = pixel_to_yolo_box((x1, y1, x2, y2), width, height)
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        return True
    except OSError as e:
        print(f"[WARN] Failed to write label file {path}: {e}")
        return False


def init_worker(config: WorkerConfigDict) -> None:
    """Initialize a worker process with the given configuration.

    This function is called by ProcessPoolExecutor for each worker process.
    It sets up the global configuration and seeds the random number generators.

    Args:
        config: Worker configuration dictionary.
    """
    global _WORKER_CONFIG
    _WORKER_CONFIG = config
    seed = int(config.get("seed", 13))
    seed = seed + os.getpid()
    random.seed(seed)
    np.random.seed(seed % (2**32))


def augment_worker(entry: Tuple[str, Path, Path], class_id: int, index: int) -> bool:
    """Worker function to build and save a single augmented image.

    This is called from worker processes during parallel augmentation.
    Uses the global _WORKER_CONFIG for configuration parameters.

    Args:
        entry: Tuple of (split, image_path, label_path).
        class_id: Target class ID for augmentation.
        index: Unique index for the output filename.

    Returns:
        True if the augmented image was successfully created and saved.

    Raises:
        RuntimeError: If worker config is not initialized.
    """
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
    """Main entry point for the YOLO background augmentation tool.

    Parses command-line arguments and runs either bucket building or
    dataset augmentation based on the --mode flag.
    """
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

    bucket_dir = args.bucket_dir
    if args.mode == "bucket" and bucket_dir is None:
        bucket_dir = args.dest

    stats_path = args.bucket_stats
    auto_stats_path = bucket_dir / "bucket_stats.json" if bucket_dir is not None else None
    stats: Optional[Dict[str, object]] = None
    counts: Optional[Dict[int, int]] = None
    heatmaps: Optional[Dict[int, np.ndarray]] = None
    if args.mode == "augment":
        if stats_path is None and auto_stats_path is not None and auto_stats_path.exists():
            stats_path = auto_stats_path
        if stats_path is not None and stats_path.exists():
            try:
                stats = load_stats(stats_path)
                counts = parse_stats_counts(stats, selected_ids)
                heatmaps = parse_stats_heatmaps(stats, selected_ids)
                if counts is not None or heatmaps is not None:
                    print(f"[INFO] Using stats from {stats_path}")
            except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
                print(f"[WARN] Failed to read stats file {stats_path}: {exc}")
        elif args.bucket_stats is not None and stats_path is not None and not stats_path.exists():
            print(f"[WARN] Stats file not found: {stats_path}")

    if counts is None:
        counts = collect_counts(entries, selected_ids)
    if heatmaps is None:
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

    if args.mode == "bucket":
        if bucket_dir is None:
            raise SystemExit("--bucket-dir is required in bucket mode.")
        debug_dir = args.debug_dir or (bucket_dir / "debug")
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
            args.no_seg,
        )
        saved_counts, seg_saved = save_bucket_dir(
            bucket, seg_bucket, class_names, bucket_dir
        )
        print_bucket_stats(saved_counts, seg_saved, class_names, selected_ids)
        if stats_path is None:
            stats_path = auto_stats_path
        if stats_path is not None:
            stats = build_stats(
                selected_ids,
                class_names,
                counts,
                heatmaps,
                args.heatmap_grid[0],
                args.heatmap_grid[1],
                saved_counts,
                seg_saved,
                bucket_dir,
            )
            save_stats(stats, stats_path)
            print(f"Saved stats to {stats_path}")
        print(f"Bucket saved to {bucket_dir}")
        return

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

    if args.bucket_dir:
        bucket, seg_bucket = load_bucket_from_dir(
            args.bucket_dir,
            selected_ids,
            class_names,
            args.max_bucket_per_class,
            args.max_seg_bucket,
        )
    else:
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
            args.no_seg,
        )
    if args.dump_bucket_dir:
        dump_bucket(
            bucket,
            seg_bucket,
            class_names,
            args.dump_bucket_dir,
            args.dump_bucket_limit,
        )
    if not any(bucket.values()):
        print("[WARN] No class blobs collected for augmentation.")
        return

    dest_root = args.dest
    aug_index = itertools.count()
    total_aug = 0
    fail_counts: Dict[int, int] = {cid: 0 for cid in needed}
    max_failures = MAX_AUGMENTATION_FAILURES

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
        "no_seg": args.no_seg,
        "feather_radius": args.feather_radius,
        "scale_range": tuple(args.scale_range) if args.scale_range else None,
        "rotate_max": args.rotate_max,
        "flip_h_prob": args.flip_h_prob,
        "flip_v_prob": args.flip_v_prob,
        "color_jitter": args.color_jitter,
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
                        except (RuntimeError, OSError, cv2.error, ValueError) as exc:
                            # RuntimeError: worker config issues
                            # OSError: file I/O errors
                            # cv2.error: OpenCV image processing errors
                            # ValueError: data validation errors
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
