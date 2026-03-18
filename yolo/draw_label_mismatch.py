"""
Draw mismatched labels between two YOLO datasets.

For each image, compares training (ground truth) labels vs generated (predicted) labels.
Mismatches are visualized with a box and a label:
  <ground_truth|predicted_label|predicted_score>
Missing entries use '---'.

Usage example:
  python yolo_draw_label_mismatch.py /path/to/train /path/to/generated /path/to/output
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont
import yaml
from tqdm import tqdm


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


@dataclass
class YoloBox:
    class_id: int
    xc: float
    yc: float
    w: float
    h: float
    score: Optional[float] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw mismatched labels between YOLO datasets."
    )
    parser.add_argument(
        "train_root",
        type=Path,
        help="Training (ground truth) YOLO dataset root.",
    )
    parser.add_argument(
        "generated_root",
        type=Path,
        help="Generated (predicted) YOLO dataset root.",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output directory for mismatch visualizations.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.6,
        help="IoU threshold to match boxes (default: 0.5).",
    )
    parser.add_argument(
        "--score-index",
        type=int,
        default=5,
        help="Index of score in prediction labels (default: 5; 0-based).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to write a text report of mismatches.",
    )
    return parser.parse_args()


def load_class_names(data_yaml: Path) -> Optional[List[str]]:
    if not data_yaml.exists():
        return None
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names")
    if not isinstance(names, list):
        return None
    return names


def iter_label_files(root: Path) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(root):
        dir_name = Path(dirpath).name
        if dir_name not in {"labels", "images"}:
            continue
        for filename in filenames:
            if filename.endswith(".txt"):
                yield Path(dirpath) / filename


def label_key(label_path: Path) -> Optional[str]:
    labels_root = None
    images_root = None
    for parent in label_path.parents:
        if parent.name == "labels":
            labels_root = parent
            break
        if parent.name == "images":
            images_root = parent
            break
    if labels_root is None and images_root is None:
        return None
    if labels_root is not None:
        split_root = labels_root.parent
        rel = label_path.relative_to(labels_root)
    else:
        split_root = images_root.parent
        rel = label_path.relative_to(images_root)
    return str(Path(split_root.name) / rel)


def find_image_by_key(root: Path, key: str) -> Optional[Path]:
    rel = Path(key)
    if len(rel.parts) < 2:
        return None
    split = rel.parts[0]
    rel_rest = Path(*rel.parts[1:])
    base = root / split / "images" / rel_rest
    base_name = base.name
    if base.suffix:
        base_name = base_name[: -len(base.suffix)]
    for ext in IMAGE_EXTS:
        candidate = base.parent / f"{base_name}{ext}"
        if candidate.exists():
            return candidate
    return None


def parse_label_file(label_path: Path, score_index: int) -> List[YoloBox]:
    boxes: List[YoloBox] = []
    with label_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(parts[0])
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                continue
            score = None
            if len(parts) > score_index:
                try:
                    score = float(parts[score_index])
                except ValueError:
                    score = None
            boxes.append(YoloBox(class_id, xc, yc, w, h, score))
    return boxes


def yolo_to_xyxy(box: YoloBox, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    x0 = int(round((box.xc - box.w / 2.0) * img_w))
    y0 = int(round((box.yc - box.h / 2.0) * img_h))
    x1 = int(round((box.xc + box.w / 2.0) * img_w))
    y1 = int(round((box.yc + box.h / 2.0) * img_h))
    x0 = max(0, min(img_w - 1, x0))
    y0 = max(0, min(img_h - 1, y0))
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0, inter_x1 - inter_x0)
    inter_h = max(0, inter_y1 - inter_y0)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    if union == 0:
        return 0.0
    return inter / union


def _short_label(name: str) -> str:
    return name[:5]


def format_label(
    gt: Optional[int],
    pred: Optional[int],
    score: Optional[float],
    gt_names: Optional[List[str]],
    pred_names: Optional[List[str]],
) -> str:
    if gt is None:
        gt_text = "---"
    else:
        gt_text = gt_names[gt] if gt_names and gt < len(gt_names) else str(gt)
        gt_text = _short_label(gt_text)
    if pred is None:
        pred_text = "---"
    else:
        pred_text = pred_names[pred] if pred_names and pred < len(pred_names) else str(pred)
        pred_text = _short_label(pred_text)
    if score is None:
        score_text = "---"
    else:
        score_text = f"{score:.3f}"
    return f"{gt_text}|{pred_text}|{score_text}"


def build_label_map(root: Path) -> Dict[str, Path]:
    label_map: Dict[str, Path] = {}
    for label_path in iter_label_files(root):
        key = label_key(label_path)
        if key is None:
            continue
        label_map[key] = label_path
    return label_map


def norm_box_stats(
    box: Tuple[int, int, int, int], img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    w = max(0, x1 - x0)
    h = max(0, y1 - y0)
    xc = x0 + w / 2.0
    yc = y0 + h / 2.0
    if img_w <= 0 or img_h <= 0:
        return 0.0, 0.0, 0.0, 0.0
    return w / img_w, h / img_h, xc / img_w, yc / img_h


def summarize(values: List[float]) -> str:
    if not values:
        return "---"
    return f"mean={sum(values)/len(values):.4f} min={min(values):.4f} max={max(values):.4f}"


def main() -> None:
    args = parse_args()
    train_root = args.train_root.resolve()
    gen_root = args.generated_root.resolve()
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    gt_names = load_class_names(train_root / "data.yaml")
    pred_names = load_class_names(gen_root / "data.yaml")
    if pred_names is None:
        pred_names = gt_names

    gt_label_map = build_label_map(train_root)
    gen_label_map = build_label_map(gen_root)

    font = ImageFont.load_default()
    total_images = 0
    written_images = 0
    report_lines: List[str] = []
    missed_gt = 0
    false_pos = 0
    class_mismatch = 0
    stat_w: List[float] = []
    stat_h: List[float] = []
    stat_xc: List[float] = []
    stat_yc: List[float] = []

    all_keys = sorted(set(gt_label_map) | set(gen_label_map))
    for key in tqdm(all_keys):
        image_path = find_image_by_key(train_root, key)
        if not image_path:
            continue
        gt_label_path = gt_label_map.get(key)
        gen_label_path = gen_label_map.get(key)
        gt_boxes = parse_label_file(gt_label_path, args.score_index) if gt_label_path else []
        pred_boxes = (
            parse_label_file(gen_label_path, args.score_index) if gen_label_path else []
        )

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img_w, img_h = img.size

            gt_xyxy = [yolo_to_xyxy(box, img_w, img_h) for box in gt_boxes]
            pred_xyxy = [yolo_to_xyxy(box, img_w, img_h) for box in pred_boxes]

            used_pred: Dict[int, int] = {}
            mismatches: List[Tuple[Tuple[int, int, int, int], str, Tuple[int, int, int]]] = []
            matches: List[Tuple[Tuple[int, int, int, int], str]] = []

            for gt_idx, gt_box in enumerate(gt_boxes):
                best_iou = 0.0
                best_pred = None
                for pred_idx, pred_box in enumerate(pred_boxes):
                    if pred_idx in used_pred:
                        continue
                    iou_val = iou(gt_xyxy[gt_idx], pred_xyxy[pred_idx])
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_pred = pred_idx
                if best_pred is None or best_iou < args.iou:
                    label = format_label(
                        gt_box.class_id,
                        None,
                        None,
                        gt_names,
                        pred_names,
                    )
                    mismatches.append((gt_xyxy[gt_idx], label, (0, 200, 0)))
                    missed_gt += 1
                else:
                    used_pred[best_pred] = gt_idx
                    pred_box = pred_boxes[best_pred]
                    if pred_box.class_id != gt_box.class_id:
                        label = format_label(
                            gt_box.class_id,
                            pred_box.class_id,
                            pred_box.score,
                            gt_names,
                            pred_names,
                        )
                        mismatches.append((gt_xyxy[gt_idx], label, (0, 200, 0)))
                        mismatches.append((pred_xyxy[best_pred], label, (220, 0, 0)))
                        class_mismatch += 1
                    else:
                        label = format_label(
                            gt_box.class_id,
                            pred_box.class_id,
                            pred_box.score,
                            gt_names,
                            pred_names,
                        )
                        matches.append((gt_xyxy[gt_idx], label))

            for pred_idx, pred_box in enumerate(pred_boxes):
                if pred_idx in used_pred:
                    continue
                label = format_label(
                    None,
                    pred_box.class_id,
                    pred_box.score,
                    gt_names,
                    pred_names,
                )
                mismatches.append((pred_xyxy[pred_idx], label, (220, 0, 0)))
                false_pos += 1

            total_images += 1
            if not mismatches and not matches:
                continue

            draw = ImageDraw.Draw(img)
            for (x0, y0, x1, y1), text, color in mismatches:
                draw.rectangle([x0, y0, x1, y1], outline=color, width=1)
                text_pos = (x0, max(0, y0 - 18))
                text_bbox = draw.textbbox(text_pos, text, font=font)
                draw.rectangle(text_bbox, fill=(200, 0, 0))
                draw.text(text_pos, text, fill=(0, 0, 0), font=font)
                w, h, xc, yc = norm_box_stats((x0, y0, x1, y1), img_w, img_h)
                stat_w.append(w)
                stat_h.append(h)
                stat_xc.append(xc)
                stat_yc.append(yc)
            for (x0, y0, x1, y1), text in matches:
                draw.rectangle([x0, y0, x1, y1], outline=(0, 140, 255), width=1)
                text_pos = (x0, max(0, y0 - 18))
                text_bbox = draw.textbbox(text_pos, text, font=font)
                draw.rectangle(text_bbox, fill=(0, 0, 0))
                draw.text(text_pos, text, fill=(255, 255, 255), font=font)

            out_path = output_root / image_path.name
            img.save(out_path)
            written_images += 1
            report_lines.append(f"{key}: {len(mismatches)} mismatches")

    print(f"Processed images: {total_images}")
    print(f"Images with mismatches: {written_images}")
    print(f"Output written to: {output_root}")
    summary_lines = [
        f"Failed boxes: {len(stat_w)}",
        f"Missed GT: {missed_gt}",
        f"False positives: {false_pos}",
        f"Class mismatches: {class_mismatch}",
        f"Size (w): {summarize(stat_w)}",
        f"Size (h): {summarize(stat_h)}",
        f"Location (xc): {summarize(stat_xc)}",
        f"Location (yc): {summarize(stat_yc)}",
    ]
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        print(f"Report written to: {args.report}")
    else:
        print("Report:")
        for line in summary_lines:
            print(line)


if __name__ == "__main__":
    main()
