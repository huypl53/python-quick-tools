"""
YOLO evaluation insight tool — identify poor predictions for dataset enhancement.

Runs YOLO inference on a dataset split, compares predictions to ground truth,
and produces actionable outputs:
  - Confusion matrix image
  - Per-image error ranking (CSV) — worst images first
  - Visual gallery of top-N worst images with GT (green) vs prediction (red)
  - Class-level error summary
  - Error catalog linking each failure back to source image

Usage:
  python3 yolo/eval_insight.py data.yaml best.pt output/ --split test
  python3 yolo/eval_insight.py data.yaml best.pt output/ --conf 0.25 --iou 0.5 --top-n 50
  python3 yolo/eval_insight.py data.yaml best.pt output/ --split valid --img-size 640

Requirements:
  pip install ultralytics opencv-python numpy matplotlib tqdm pyyaml
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm
from ultralytics import YOLO


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Box:
    class_id: int
    xc: float
    yc: float
    w: float
    h: float
    conf: float = 1.0

    def xyxy(self, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        x0 = int(round((self.xc - self.w / 2) * img_w))
        y0 = int(round((self.yc - self.h / 2) * img_h))
        x1 = int(round((self.xc + self.w / 2) * img_w))
        y1 = int(round((self.yc + self.h / 2) * img_h))
        x0 = max(0, min(img_w, x0))
        y0 = max(0, min(img_h, y0))
        x1 = max(0, min(img_w, x1))
        y1 = max(0, min(img_h, y1))
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        return x0, y0, x1, y1


@dataclass
class ImageErrors:
    path: Path
    missed: int = 0              # GT boxes with no matching pred (FN)
    false_pos: int = 0           # Pred boxes with no matching GT (FP)
    misclassified: int = 0       # Matched but wrong class
    low_iou: int = 0             # Matched + correct class but borderline IoU
    total_gt: int = 0
    total_pred: int = 0
    details: List[str] = field(default_factory=list)

    @property
    def error_score(self) -> float:
        """Higher = worse. Weighted sum of error types."""
        return self.missed * 3 + self.false_pos * 2 + self.misclassified * 2 + self.low_iou * 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YOLO evaluation insight — identify poor cases for dataset enhancement."
    )
    p.add_argument("data_yaml", type=Path, help="Path to data.yaml")
    p.add_argument("weights", type=Path, help="Path to YOLO weights (.pt)")
    p.add_argument("output", type=Path, help="Output directory")
    p.add_argument("--split", default="test", help="Dataset split (default: test)")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold for GT-pred matching (default: 0.5)")
    p.add_argument("--nms-iou", type=float, default=0.7, help="NMS IoU threshold for inference (default: 0.7)")
    p.add_argument("--top-n", type=int, default=50, help="Number of worst images to visualize (default: 50)")
    p.add_argument("--img-size", type=int, default=640, help="Inference image size (default: 640)")
    p.add_argument("--batch", type=int, default=16, help="Inference batch size (default: 16)")
    p.add_argument("--device", default=None, help="Device: cpu, 0, 0,1, etc.")
    return p.parse_args()


def load_class_names(data_yaml: Path) -> List[str]:
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names")
    if isinstance(names, dict):
        return [names[k] for k in sorted(names.keys())]
    if isinstance(names, list):
        return names
    raise ValueError(f"Cannot parse class names from {data_yaml}")


def resolve_split_dir(data_yaml: Path, split: str) -> Path:
    """Resolve the image directory for a given split from data.yaml."""
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    split_path = data.get(split)
    if split_path is None:
        raise ValueError(f"Split '{split}' not found in {data_yaml}")
    split_path = Path(split_path)
    if not split_path.is_absolute():
        split_path = (data_yaml.parent / split_path).resolve()
    # data.yaml may point to images/ dir or split root
    if split_path.name != "images" and (split_path / "images").is_dir():
        return split_path / "images"
    return split_path


def find_label_for_image(image_path: Path) -> Optional[Path]:
    """Given an image path, find its corresponding label .txt file."""
    parts = list(image_path.parts)
    try:
        idx = len(parts) - 1 - parts[::-1].index("images")
    except ValueError:
        return None
    parts[idx] = "labels"
    label_path = Path(*parts).with_suffix(".txt")
    return label_path if label_path.exists() else None


def parse_label_file(label_path: Path) -> List[Box]:
    boxes: List[Box] = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                boxes.append(Box(
                    class_id=int(parts[0]),
                    xc=float(parts[1]),
                    yc=float(parts[2]),
                    w=float(parts[3]),
                    h=float(parts[4]),
                ))
            except ValueError:
                continue
    return boxes


def compute_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter == 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def collect_images(images_dir: Path) -> List[Path]:
    result = []
    for root, _, files in os.walk(images_dir):
        for f in sorted(files):
            if Path(f).suffix.lower() in IMAGE_EXTS:
                result.append(Path(root) / f)
    return result


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def run_inference(model: YOLO, image_paths: List[Path], args: argparse.Namespace) -> Dict[str, List[Box]]:
    """Run YOLO inference and return predictions keyed by image path string."""
    preds: Dict[Path, List[Box]] = {}
    kwargs = dict(
        conf=args.conf,
        iou=args.nms_iou,
        imgsz=args.img_size,
        batch=args.batch,
        verbose=False,
    )
    if args.device is not None:
        kwargs["device"] = args.device

    results = model.predict(source=[str(p) for p in image_paths], stream=True, **kwargs)
    for result in tqdm(results, total=len(image_paths), desc="Inference"):
        img_path = Path(result.path).resolve()
        boxes = []
        if result.boxes is not None:
            for box_data in result.boxes:
                xyxy = box_data.xyxyn[0].cpu().numpy()  # normalized
                xc = (xyxy[0] + xyxy[2]) / 2
                yc = (xyxy[1] + xyxy[3]) / 2
                w = xyxy[2] - xyxy[0]
                h = xyxy[3] - xyxy[1]
                boxes.append(Box(
                    class_id=int(box_data.cls[0].item()),
                    xc=float(xc),
                    yc=float(yc),
                    w=float(w),
                    h=float(h),
                    conf=float(box_data.conf[0].item()),
                ))
        preds[img_path] = boxes
    return preds


def _resolve_preds_key(img_path: Path, preds: Dict[Path, List[Box]]) -> List[Box]:
    return preds.get(img_path.resolve(), [])


def evaluate_image(
    gt_boxes: List[Box],
    pred_boxes: List[Box],
    img_w: int,
    img_h: int,
    iou_thresh: float,
    class_names: List[str],
) -> Tuple[ImageErrors, List[Tuple[str, int, int]]]:
    """
    Compare GT vs predictions for one image.
    Returns ImageErrors and list of (error_type, gt_class, pred_class) for confusion matrix.
    """
    errors = ImageErrors(path=Path())
    errors.total_gt = len(gt_boxes)
    errors.total_pred = len(pred_boxes)
    confusion_entries: List[Tuple[str, int, int]] = []  # (type, gt_cls, pred_cls)

    gt_xyxy = [b.xyxy(img_w, img_h) for b in gt_boxes]
    pred_xyxy = [b.xyxy(img_w, img_h) for b in pred_boxes]

    used_pred = set()

    for gi, gb in enumerate(gt_boxes):
        best_iou = 0.0
        best_pi = -1
        for pi, pb in enumerate(pred_boxes):
            if pi in used_pred:
                continue
            iou_val = compute_iou(gt_xyxy[gi], pred_xyxy[pi])
            if iou_val > best_iou:
                best_iou = iou_val
                best_pi = pi

        if best_pi < 0 or best_iou < iou_thresh:
            errors.missed += 1
            gt_name = class_names[gb.class_id] if gb.class_id < len(class_names) else str(gb.class_id)
            errors.details.append(f"MISSED: GT={gt_name}")
            # Confusion: GT class → background (no detection)
            confusion_entries.append(("missed", gb.class_id, -1))
        else:
            used_pred.add(best_pi)
            pb = pred_boxes[best_pi]
            gt_name = class_names[gb.class_id] if gb.class_id < len(class_names) else str(gb.class_id)
            pred_name = class_names[pb.class_id] if pb.class_id < len(class_names) else str(pb.class_id)
            if pb.class_id != gb.class_id:
                errors.misclassified += 1
                errors.details.append(
                    f"MISCLASS: GT={gt_name} Pred={pred_name} conf={pb.conf:.3f} IoU={best_iou:.3f}"
                )
                confusion_entries.append(("misclass", gb.class_id, pb.class_id))
            elif best_iou < iou_thresh + 0.15:
                # Correct class but borderline IoU — flag as low quality
                errors.low_iou += 1
                errors.details.append(
                    f"LOW_IOU: GT={gt_name} IoU={best_iou:.3f} conf={pb.conf:.3f}"
                )
                confusion_entries.append(("correct", gb.class_id, pb.class_id))
            else:
                confusion_entries.append(("correct", gb.class_id, pb.class_id))

    # Unmatched predictions = false positives
    for pi, pb in enumerate(pred_boxes):
        if pi in used_pred:
            continue
        errors.false_pos += 1
        pred_name = class_names[pb.class_id] if pb.class_id < len(class_names) else str(pb.class_id)
        errors.details.append(f"FALSE_POS: Pred={pred_name} conf={pb.conf:.3f}")
        confusion_entries.append(("false_pos", -1, pb.class_id))

    return errors, confusion_entries


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------

def draw_eval_image(
    image_path: Path,
    gt_boxes: List[Box],
    pred_boxes: List[Box],
    class_names: List[str],
    iou_thresh: float,
) -> np.ndarray:
    """Draw GT (green) and predictions (red=error, blue=correct) on image."""
    img = cv2.imread(str(image_path))
    if img is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    h, w = img.shape[:2]

    gt_xyxy = [b.xyxy(w, h) for b in gt_boxes]
    pred_xyxy = [b.xyxy(w, h) for b in pred_boxes]

    # Match pred to GT
    used_pred = set()
    pred_status: Dict[int, str] = {}  # pi -> "correct" | "misclass"

    for gi, gb in enumerate(gt_boxes):
        best_iou = 0.0
        best_pi = -1
        for pi in range(len(pred_boxes)):
            if pi in used_pred:
                continue
            iou_val = compute_iou(gt_xyxy[gi], pred_xyxy[pi])
            if iou_val > best_iou:
                best_iou = iou_val
                best_pi = pi
        if best_pi >= 0 and best_iou >= iou_thresh:
            used_pred.add(best_pi)
            if pred_boxes[best_pi].class_id == gb.class_id:
                pred_status[best_pi] = "correct"
            else:
                pred_status[best_pi] = "misclass"

    # Draw GT boxes in green
    for gi, gb in enumerate(gt_boxes):
        x0, y0, x1, y1 = gt_xyxy[gi]
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 0), 2)
        name = class_names[gb.class_id] if gb.class_id < len(class_names) else str(gb.class_id)
        label = f"GT:{name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x0, y0 - th - 6), (x0 + tw, y0), (0, 200, 0), -1)
        cv2.putText(img, label, (x0, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Draw pred boxes
    for pi, pb in enumerate(pred_boxes):
        x0, y0, x1, y1 = pred_xyxy[pi]
        status = pred_status.get(pi, "false_pos")
        if status == "correct":
            color = (200, 150, 0)  # blue-ish = correct
        elif status == "misclass":
            color = (0, 0, 220)    # red = misclassified
        else:
            color = (0, 0, 220)    # red = false positive

        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        name = class_names[pb.class_id] if pb.class_id < len(class_names) else str(pb.class_id)
        label = f"P:{name} {pb.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = y1 + th + 6
        cv2.rectangle(img, (x0, y1), (x0 + tw, ly), color, -1)
        cv2.putText(img, label, (x0, y1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: List[str],
    output_path: Path,
) -> None:
    """Save confusion matrix as image. Rows=GT, Cols=Predicted. Extra row/col for BG."""
    labels = class_names + ["BG"]
    n = len(labels)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(8, n * 0.8)))
    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix")

    # Annotate cells
    thresh = matrix.max() / 2
    for i in range(n):
        for j in range(n):
            val = int(matrix[i, j])
            if val == 0:
                continue
            color = "white" if matrix[i, j] > thresh else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=7)

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"[INFO] Confusion matrix saved: {output_path}")


def write_error_csv(
    all_errors: List[ImageErrors],
    output_path: Path,
) -> None:
    """Write per-image error CSV (expects pre-sorted by error_score descending)."""
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "image", "error_score", "missed", "false_pos",
            "misclassified", "low_iou", "total_gt", "total_pred", "details",
        ])
        for i, err in enumerate(all_errors, 1):
            writer.writerow([
                i, err.path.name, err.error_score, err.missed, err.false_pos,
                err.misclassified, err.low_iou, err.total_gt, err.total_pred,
                "; ".join(err.details),
            ])
    print(f"[INFO] Error ranking CSV saved: {output_path}")


def write_class_summary(
    class_stats: Dict[int, Dict[str, int]],
    class_names: List[str],
    output_path: Path,
) -> None:
    """Write class-level summary CSV."""
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class_id", "class_name", "total_gt", "correct", "missed",
            "misclassified", "false_pos", "recall", "precision",
        ])
        for cid in sorted(class_stats.keys()):
            s = class_stats[cid]
            name = class_names[cid] if cid < len(class_names) else str(cid)
            total_gt = s.get("total_gt", 0)
            correct = s.get("correct", 0)
            missed = s.get("missed", 0)
            misclass = s.get("misclass", 0)
            fp = s.get("false_pos", 0)
            recall = correct / total_gt if total_gt > 0 else 0.0
            precision = correct / (correct + fp + misclass) if (correct + fp + misclass) > 0 else 0.0
            writer.writerow([
                cid, name, total_gt, correct, missed, misclass, fp,
                f"{recall:.4f}", f"{precision:.4f}",
            ])
    print(f"[INFO] Class summary saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    data_yaml = args.data_yaml.resolve()
    weights = args.weights.resolve()
    output_dir = args.output.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    gallery_dir = output_dir / "gallery"
    gallery_dir.mkdir(exist_ok=True)

    # Load model & class names
    print(f"[INFO] Loading model: {weights}")
    model = YOLO(str(weights))
    class_names = load_class_names(data_yaml)
    nc = len(class_names)
    print(f"[INFO] Classes ({nc}): {class_names}")

    # Find images
    images_dir = resolve_split_dir(data_yaml, args.split)
    image_paths = collect_images(images_dir)
    print(f"[INFO] Found {len(image_paths)} images in {images_dir}")
    if not image_paths:
        print("[ERROR] No images found. Check data.yaml split paths.")
        return

    # Run inference
    preds = run_inference(model, image_paths, args)

    # Evaluate each image
    all_errors: List[ImageErrors] = []
    # confusion matrix: (nc+1) x (nc+1), last row/col = background
    conf_matrix = np.zeros((nc + 1, nc + 1), dtype=np.int64)
    class_stats: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    print("[INFO] Evaluating predictions vs ground truth...")
    for img_path in tqdm(image_paths, desc="Evaluating"):
        label_path = find_label_for_image(img_path)
        if label_path is None and not hasattr(main, "_label_warn_shown"):
            print(f"[WARN] No label file found for {img_path.name} (and possibly others)")
            main._label_warn_shown = True
        gt_boxes = parse_label_file(label_path) if label_path else []
        pred_boxes = _resolve_preds_key(img_path, preds)

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        errors, confusion_entries = evaluate_image(
            gt_boxes, pred_boxes, img_w, img_h, args.iou, class_names
        )
        errors.path = img_path
        all_errors.append(errors)

        # Update confusion matrix & class stats
        for entry_type, gt_cls, pred_cls in confusion_entries:
            gt_idx = gt_cls if gt_cls >= 0 else nc    # -1 → BG row
            pred_idx = pred_cls if pred_cls >= 0 else nc  # -1 → BG col
            conf_matrix[gt_idx, pred_idx] += 1

            if entry_type == "correct":
                if gt_cls >= 0:
                    class_stats[gt_cls]["correct"] += 1
            elif entry_type == "missed":
                if gt_cls >= 0:
                    class_stats[gt_cls]["missed"] += 1
            elif entry_type == "misclass":
                if gt_cls >= 0:
                    class_stats[gt_cls]["misclass"] += 1
            elif entry_type == "false_pos":
                if pred_cls >= 0:
                    class_stats[pred_cls]["false_pos"] += 1

        # Track total GT per class
        for gb in gt_boxes:
            class_stats[gb.class_id]["total_gt"] += 1

    # Sort by error score
    all_errors.sort(key=lambda e: e.error_score, reverse=True)

    # --- Outputs ---

    # 1. Confusion matrix
    plot_confusion_matrix(conf_matrix, class_names, output_dir / "confusion_matrix.png")

    # 2. Per-image error CSV
    write_error_csv(all_errors, output_dir / "error_ranking.csv")

    # 3. Class summary
    write_class_summary(class_stats, class_names, output_dir / "class_summary.csv")

    # 4. Visual gallery of top-N worst images
    top_n = min(args.top_n, len(all_errors))
    print(f"[INFO] Generating gallery for top {top_n} worst images...")
    for i, err in enumerate(tqdm(all_errors[:top_n], desc="Gallery")):
        label_path = find_label_for_image(err.path)
        gt_boxes = parse_label_file(label_path) if label_path else []
        pred_boxes = _resolve_preds_key(err.path, preds)
        vis = draw_eval_image(err.path, gt_boxes, pred_boxes, class_names, args.iou)

        # Add error info banner at top
        banner_h = 30
        banner = np.zeros((banner_h, vis.shape[1], 3), dtype=np.uint8)
        text = (
            f"#{i+1} score={err.error_score} | "
            f"missed={err.missed} fp={err.false_pos} "
            f"misclass={err.misclassified} low_iou={err.low_iou} | "
            f"{err.path.name}"
        )
        cv2.putText(banner, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        vis = np.vstack([banner, vis])

        out_path = gallery_dir / f"{i+1:04d}_{err.path.stem}.jpg"
        cv2.imwrite(str(out_path), vis)

    # 5. Summary report
    total_gt = sum(e.total_gt for e in all_errors)
    total_pred = sum(e.total_pred for e in all_errors)
    total_missed = sum(e.missed for e in all_errors)
    total_fp = sum(e.false_pos for e in all_errors)
    total_misclass = sum(e.misclassified for e in all_errors)
    total_low_iou = sum(e.low_iou for e in all_errors)
    total_correct = total_gt - total_missed - total_misclass - total_low_iou

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Images evaluated:  {len(all_errors)}")
    print(f"Total GT boxes:    {total_gt}")
    print(f"Total predictions: {total_pred}")
    print(f"Correct matches:   {total_correct} ({total_correct/max(1,total_gt)*100:.1f}%)")
    print(f"Missed (FN):       {total_missed} ({total_missed/max(1,total_gt)*100:.1f}%)")
    print(f"False pos (FP):    {total_fp}")
    print(f"Misclassified:     {total_misclass}")
    print(f"Low IoU:           {total_low_iou}")
    print(f"\nOutputs in: {output_dir}")
    print(f"  confusion_matrix.png  — confusion matrix")
    print(f"  error_ranking.csv     — per-image errors, worst first")
    print(f"  class_summary.csv     — per-class recall/precision")
    print(f"  gallery/              — top {top_n} worst images visualized")
    print("=" * 60)


if __name__ == "__main__":
    main()
