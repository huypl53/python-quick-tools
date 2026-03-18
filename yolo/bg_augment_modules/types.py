"""Type definitions and configuration objects for the augmentation tool."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict, Union

import numpy as np


@dataclass
class Blob:
    """A cropped image region with optional class label."""

    image: np.ndarray
    class_id: Optional[int]


@dataclass
class AugmentConfig:
    """Configuration for the augmentation pipeline."""

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


class WorkerConfigDict(TypedDict):
    """Configuration dictionary passed to worker processes."""

    selected_ids: List[int]
    bucket: Dict[int, List[Blob]]
    seg_bucket: List[Blob]
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
    layout_perturb_prob: float
    layout_jitter: int
    layout_max_tries: int


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
