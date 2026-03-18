"""Command-line argument parsing for yolo_bg_augment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


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
    parser.add_argument(
        "--layout-perturb-prob",
        type=float,
        default=0.0,
        help="Probability to use layout-perturb augmentation instead of full recompose. Default: 0.",
    )
    parser.add_argument(
        "--layout-jitter",
        type=int,
        default=12,
        help="Max pixel shift when perturbing layout. Default: 12.",
    )
    parser.add_argument(
        "--layout-max-tries",
        type=int,
        default=20,
        help="Max placement attempts per object in layout-perturb mode. Default: 20.",
    )
    args = parser.parse_args()

    if args.no_fill_empty:
        args.fill_empty = False

    return args
