# FiftyOne YOLO Evaluation Viewer - Research Report

## Summary
FiftyOne has native support for YOLO datasets via `fo.types.YOLOv5Dataset` and direct Ultralytics integration. BBox format is `[top_left_x, top_left_y, width, height]` with normalized coords (0-1). Use `fo.launch_app()` for viewer and `evaluate_detections()` for metrics.

---

## 1. Loading YOLO Dataset from data.yaml

FiftyOne can load Roboflow/Ultralytics-format datasets directly:

```python
import fiftyone as fo

# Load entire dataset with splits
dataset = fo.Dataset.from_dir(
    dataset_dir="path/to/dataset",
    dataset_type=fo.types.YOLOv5Dataset,
)

# Or load specific split
train_dataset = fo.Dataset.from_dir(
    dataset_dir="path/to/dataset",
    dataset_type=fo.types.YOLOv5Dataset,
    split="train",
)

# data.yaml is auto-detected; if elsewhere, pass:
# yaml_path="path/to/data.yaml"
```

**Format expectation**: data.yaml + images/{train/val/test} + labels/{train/val/test}

---

## 2. Running YOLO Inference with Ultralytics

```python
from ultralytics import YOLO
import fiftyone.utils.ultralytics as fou

# Load model (auto-downloads)
model = YOLO("yolov8m.pt")

# Option A: Batch apply to dataset
dataset.apply_model(model, label_field="predictions", batch_size=16)

# Option B: Manual inference (more control)
for sample in dataset.iter_samples(progress=True):
    results = model(sample.filepath)[0]
    # Convert YOLO output to FiftyOne Detections
    sample["predictions"] = fou.to_detections(results)
    sample.save()
```

**Note**: `fou.to_detections()` handles YOLO format conversion automatically.

---

## 3. YOLO Format Conversion

**YOLO format** (from labels/*.txt): `[class_id, x_center, y_center, w, h]` (normalized)

**FiftyOne Detection format**: `[top_left_x, top_left_y, width, height]` (normalized)

Manual conversion if needed:

```python
from fiftyone import Detection, Detections

def yolo_to_fiftyone(yolo_result, image_width, image_height):
    """Convert ultralytics result to FiftyOne detections."""
    detections = []
    for box in yolo_result.boxes:
        # YOLO gives [x1, y1, x2, y2] in pixel coords
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])

        # Convert to [top_left_x, top_left_y, w, h] normalized
        x = (x1 / image_width).item()
        y = (y1 / image_height).item()
        w = ((x2 - x1) / image_width).item()
        h = ((y2 - y1) / image_height).item()

        detections.append(
            Detection(
                label=str(cls_id),  # or map to class name
                bounding_box=[x, y, w, h],
                confidence=conf,
            )
        )
    return Detections(detections=detections)
```

---

## 4. Running evaluate_detections()

```python
# Compute evaluation metrics
results = dataset.evaluate_detections(
    predictions_field="predictions",
    gt_field="ground_truth",  # FiftyOne auto-loads labels as this
    eval_key="eval",  # stores TP/FP/FN per sample
    compute_mAP=True,  # includes mAP computation
)

# Get confusion matrix
confusion_matrix = results.confusion_matrix(classes=dataset.distinct("ground_truth.detections.label"))

# Per-class metrics
for class_name in dataset.distinct("ground_truth.detections.label"):
    metrics = results.metrics(class_name)
    print(f"{class_name}: AP={metrics['AP']:.3f}")

# Sample-level stats now available
for sample in dataset:
    print(f"{sample.filepath}: TP={sample.eval_tp}, FP={sample.eval_fp}, FN={sample.eval_fn}")
```

**Key Parameters**:
- `predictions_field`: your predictions label
- `gt_field`: ground truth field (FiftyOne uses "ground_truth" by default)
- `eval_key`: prefix for per-sample stats (eval_tp, eval_fp, eval_fn)
- `compute_mAP=True`: enables precision/recall computation

---

## 5. Launching FiftyOne App

```python
# Simplest: just launch
session = fo.launch_app(dataset)

# Or with custom name and port
session = fo.launch_app(
    dataset,
    name="yolo_eval",
    port=5151,
    address="127.0.0.1",
)

# Keep session running for browser interaction
session.wait()
```

---

## 6. Complete Working Example

```python
import fiftyone as fo
from ultralytics import YOLO

# 1. Load dataset
dataset = fo.Dataset.from_dir(
    "path/to/dataset",
    dataset_type=fo.types.YOLOv5Dataset,
)

# 2. Run inference
model = YOLO("yolov8m.pt")
dataset.apply_model(model, label_field="predictions")

# 3. Evaluate
results = dataset.evaluate_detections(
    "predictions",
    gt_field="ground_truth",
    eval_key="eval",
    compute_mAP=True,
)

# 4. Print metrics
print("mAP:", results.metrics()["mAP"])

# 5. Launch viewer
fo.launch_app(dataset)
```

---

## Detection Format Reference

FiftyOne Detection object:
```python
Detection(
    label="person",                    # class name or id
    bounding_box=[0.1, 0.2, 0.3, 0.4], # [x_norm, y_norm, w_norm, h_norm]
    confidence=0.95,                   # model confidence
)
```

**Bounding box coords**:
- Range: [0, 1] × [0, 1]
- Format: top-left corner + width/height
- **NOT** center-based (that's YOLO's internal format)

---

## Key Answers

| Question | Answer |
|----------|--------|
| **YOLO→FiftyOne conversion** | `fou.to_detections(yolo_result)` or manual [x_norm, y_norm, w_norm, h_norm] |
| **eval_key usage** | Specify string to store TP/FP/FN as sample-level fields |
| **Native YOLO loading** | Yes: `fo.Dataset.from_dir(..., dataset_type=fo.types.YOLOv5Dataset)` |
| **Minimal code** | Load + infer + evaluate + `fo.launch_app(dataset)` (~10 lines) |
| **Coordinate format** | Normalized [0,1], top-left corner, [x, y, w, h] |

---

## Installation

```bash
pip install fiftyone ultralytics torch
```

---

## Sources
- [FiftyOne Ultralytics Integration](https://docs.voxel51.com/integrations/ultralytics.html)
- [FiftyOne YOLOv8 Tutorial](https://docs.voxel51.com/tutorials/yolov8.html)
- [FiftyOne YOLO Utils API](https://docs.voxel51.com/api/fiftyone.utils.yolo.html)
- [FiftyOne Evaluation Guide](https://docs.voxel51.com/user_guide/evaluation.html)
- [FiftyOne Dataset Import](https://docs.voxel51.com/user_guide/import_datasets.html)
- [Ultralytics Traffic Safety Pipeline](https://aftabgazali001.medium.com/from-bdd100k-to-yolov8-a-practical-traffic-safety-detection-pipeline-fiftyone-ultralytics-256ec1fd1437)
