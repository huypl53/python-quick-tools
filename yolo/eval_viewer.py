"""
FiftyOne-based YOLO evaluation viewer — interactive visual analysis for dataset improvement.

Loads a YOLO dataset, runs inference with trained weights, and launches
an interactive FiftyOne web app with GT vs prediction overlays, confusion
matrix, per-sample TP/FP/FN metrics, and filtering by error type/class.

Inference results are cached in FiftyOne's database. Subsequent launches
skip inference and load instantly. Use --force to re-run inference.

Usage:
  python3 yolo/eval_viewer.py data.yaml best.pt --split valid
  python3 yolo/eval_viewer.py data.yaml best.pt --split valid --force
  python3 yolo/eval_viewer.py data.yaml best.pt --split test --conf 0.25 --iou 0.5
  python3 yolo/eval_viewer.py data.yaml best.pt --split valid --device 0 --port 5151

Requirements:
  pip install fiftyone ultralytics pyyaml tqdm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Remove project dir from sys.path to avoid csv/ shadowing stdlib csv module
_project_dir = str(Path(__file__).resolve().parent.parent)
sys.path = [p for p in sys.path if p != _project_dir]

import fiftyone as fo
import fiftyone.utils.ultralytics as fou
import yaml
from tqdm import tqdm
from ultralytics import YOLO


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    # Try common aliases: val/valid, test
    aliases = {
        "val": ["val", "valid", "validation"],
        "valid": ["valid", "val", "validation"],
        "test": ["test"],
        "train": ["train"],
    }
    candidates = aliases.get(split, [split])
    split_path = None
    for name in candidates:
        if name in data:
            split_path = data[name]
            break
    if split_path is None:
        raise ValueError(f"Split '{split}' not found in {data_yaml}")
    split_path = Path(split_path)
    if not split_path.is_absolute():
        resolved = (data_yaml.parent / split_path).resolve()
        if not resolved.exists():
            # Roboflow data.yaml uses ../split/images but dirs are local
            stripped = Path(str(split_path).lstrip("./").lstrip("../"))
            fallback = (data_yaml.parent / stripped).resolve()
            if fallback.exists():
                resolved = fallback
            else:
                print(f"[WARN] Split path not found: {resolved} (also tried {fallback})")
        split_path = resolved
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


def parse_label_file(label_path: Path, class_names: List[str]) -> List[fo.Detection]:
    """Parse a YOLO label file into FiftyOne Detections."""
    detections = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(parts[0])
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                # Convert YOLO center-format to FiftyOne top-left format
                x = xc - w / 2
                y = yc - h / 2
                label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                detections.append(fo.Detection(
                    label=label,
                    bounding_box=[x, y, w, h],
                ))
            except (ValueError, IndexError):
                continue
    return detections


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def build_dataset(
    data_yaml: Path,
    split: str,
    class_names: List[str],
    dataset_name: str,
) -> fo.Dataset:
    """Load images + GT labels into a FiftyOne dataset."""
    images_dir = resolve_split_dir(data_yaml, split)
    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")
    print(f"Found {len(image_paths)} images in {images_dir}")

    dataset = fo.Dataset(name=dataset_name)
    dataset.persistent = True
    samples = []
    missing_labels = 0
    for img_path in tqdm(image_paths, desc="Loading GT"):
        sample = fo.Sample(filepath=str(img_path))
        label_path = find_label_for_image(img_path)
        if label_path is not None:
            dets = parse_label_file(label_path, class_names)
            sample["ground_truth"] = fo.Detections(detections=dets)
        else:
            sample["ground_truth"] = fo.Detections(detections=[])
            missing_labels += 1
        samples.append(sample)

    dataset.add_samples(samples)
    if missing_labels:
        print(f"[WARN] {missing_labels} images had no label file")
    return dataset


def run_inference(dataset: fo.Dataset, model: YOLO, conf: float, iou: float, device: Optional[str], img_size: int):
    """Run per-image inference and add predictions to dataset."""
    kwargs = dict(conf=conf, iou=iou, imgsz=img_size, verbose=False)
    if device is not None:
        kwargs["device"] = device

    for sample in tqdm(dataset.iter_samples(progress=False), total=len(dataset), desc="Inference"):
        results = model.predict(source=sample.filepath, **kwargs)
        sample["predictions"] = fou.to_detections(results[0])
        sample.save()


def run_evaluation(dataset: fo.Dataset, iou: float):
    """Run FiftyOne evaluation to compute TP/FP/FN per sample."""
    eval_key = "eval"
    if eval_key in dataset.list_evaluations():
        print(f"Evaluation '{eval_key}' already exists, loading cached results")
        results = dataset.load_evaluation_results(eval_key)
    else:
        print("Evaluating detections...")
        results = dataset.evaluate_detections(
            "predictions",
            gt_field="ground_truth",
            eval_key=eval_key,
            iou=iou,
            compute_mAP=True,
        )
    results.print_report()
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FiftyOne YOLO evaluation viewer — interactive visual analysis."
    )
    p.add_argument("data_yaml", type=Path, help="Path to data.yaml")
    p.add_argument("weights", type=Path, help="Path to YOLO weights (.pt)")
    p.add_argument("--split", default="valid", help="Dataset split (default: valid)")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold for eval matching (default: 0.5)")
    p.add_argument("--nms-iou", type=float, default=0.7, help="NMS IoU threshold for inference (default: 0.7)")
    p.add_argument("--img-size", type=int, default=640, help="Inference image size (default: 640)")
    p.add_argument("--device", default=None, help="Device: cpu, 0, 0,1, etc.")
    p.add_argument("--port", type=int, default=5151, help="FiftyOne app port (default: 5151)")
    p.add_argument("--name", default=None, help="Dataset name (default: auto from data.yaml)")
    p.add_argument("--force", action="store_true", help="Force re-run inference even if cached")
    return p.parse_args()


def main():
    args = parse_args()

    # Dataset name
    dataset_name = args.name or f"yolo-eval-{args.data_yaml.stem}-{args.split}"

    # Check for cached dataset
    if fo.dataset_exists(dataset_name) and not args.force:
        print(f"Loading cached dataset '{dataset_name}' (use --force to re-run inference)")
        dataset = fo.load_dataset(dataset_name)
        print(f"  {len(dataset)} samples loaded")
    else:
        # Delete old dataset if forcing
        if fo.dataset_exists(dataset_name):
            fo.delete_dataset(dataset_name)
            print(f"Deleted old dataset '{dataset_name}'")

        # Load class names
        class_names = load_class_names(args.data_yaml)
        print(f"Classes ({len(class_names)}): {class_names}")

        # Build FiftyOne dataset with GT labels
        dataset = build_dataset(args.data_yaml, args.split, class_names, dataset_name)

        # Load model and run inference
        print(f"Loading model: {args.weights}")
        model = YOLO(str(args.weights))
        run_inference(dataset, model, args.conf, args.nms_iou, args.device, args.img_size)

    # Always run evaluation — it's fast (no inference) and needed for
    # the Model Evaluation panel in FiftyOne UI
    run_evaluation(dataset, args.iou)

    # Launch interactive viewer
    print(f"\nLaunching FiftyOne app on port {args.port}...")
    print("Tips:")
    print("  - Use sidebar to filter by class, confidence, TP/FP/FN")
    print("  - Click images to see bbox overlays (green=GT, blue=predictions)")
    print("  - Use Plots > Confusion Matrix for interactive error analysis")
    print("  - Sort by eval_fn (descending) to find worst images")

    # Exclude eval primitives from sidebar to prevent default range filters
    # from hiding detections. The per-detection eval field (tp/fp/fn tags)
    # inside ground_truth/predictions is still available for filtering.
    dataset.app_config.sidebar_groups = [
        fo.SidebarGroupDocument(name="metadata", paths=["metadata"]),
        fo.SidebarGroupDocument(name="labels", paths=["ground_truth", "predictions"]),
        fo.SidebarGroupDocument(
            name="eval (per-sample)",
            paths=["eval_tp", "eval_fp", "eval_fn"],
            expanded=False,
        ),
    ]
    dataset.save()

    session = fo.launch_app(dataset, port=args.port, address="0.0.0.0")

    # Color by class label instead of field (default colors all GT same, all preds same)
    session.color_scheme = fo.ColorScheme(color_by="value")

    session.wait()


if __name__ == "__main__":
    main()
