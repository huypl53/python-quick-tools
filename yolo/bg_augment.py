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

from __future__ import annotations

import itertools
import json
import multiprocessing as mp
import random
from typing import Dict, Optional

import cv2
import numpy as np
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from tqdm import tqdm

from yolo.bg_augment_modules.bucket import (
    build_bucket,
    dump_bucket,
    load_bucket_from_dir,
    print_bucket_stats,
    save_bucket_dir,
)
from yolo.bg_augment_modules.cli import parse_args
from yolo.bg_augment_modules.constants import MAX_AUGMENTATION_FAILURES
from yolo.bg_augment_modules.io import (
    copy_metadata,
    copy_original_dataset,
    iter_label_images,
    load_class_names,
)
from yolo.bg_augment_modules.stats import (
    build_stats,
    collect_counts,
    collect_heatmaps,
    load_stats,
    parse_stats_counts,
    parse_stats_heatmaps,
    print_heatmaps,
    save_stats,
)
from yolo.bg_augment_modules.worker import augment_worker, init_worker


def main() -> None:
    """Main entry point for the YOLO background augmentation tool."""

    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.bucket_sample_rate <= 0 or args.bucket_sample_rate > 1.0:
        raise SystemExit("--bucket-sample-rate must be in (0, 1].")
    if args.dense_step < 1:
        raise SystemExit("--dense-step must be >= 1.")
    if args.fill_ratio < 0 or args.fill_ratio > 1.0:
        raise SystemExit("--fill-ratio must be in [0, 1].")
    if args.layout_perturb_prob < 0 or args.layout_perturb_prob > 1.0:
        raise SystemExit("--layout-perturb-prob must be in [0, 1].")
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
        saved_counts, seg_saved = save_bucket_dir(bucket, seg_bucket, class_names, bucket_dir)
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

    needed: Dict[int, int] = {}
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
        "layout_perturb_prob": args.layout_perturb_prob,
        "layout_jitter": args.layout_jitter,
        "layout_max_tries": args.layout_max_tries,
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
                    while len(pending) < args.workers and total_aug + len(pending) < target_total:
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
