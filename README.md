# Python Tools

Collection of CLI tools for YOLO dataset manipulation and data processing.

## Structure

```
python_tools/
├── yolo/                          # YOLO dataset tools
│   ├── bg_augment.py              # Background augmentation (cut-paste rebalancing)
│   ├── bg_augment_modules/        # bg_augment subpackage
│   ├── visualizer.py              # Visualize YOLO annotations
│   ├── classification_crops.py    # Crop images for classification
│   ├── draw_label_mismatch.py     # Draw label mismatches
│   ├── ds_merge.py                # Merge YOLO datasets
│   ├── ds_squeeze.py              # Squeeze/compact datasets
│   ├── extract_classification.py  # Extract classification data
│   ├── extract_ref_crops.py       # Extract reference crops
│   ├── filter_label.py            # Filter by label
│   ├── filter_ref_boxes.py        # Filter reference boxes
│   ├── filter_small_thin_boxes.py # Filter small/thin boxes
│   ├── polygon_to_bbox.py         # Convert polygons to bboxes
│   ├── seg_to_bbox.py             # Convert segmentation to bboxes
│   └── resize_padding.py          # Resize with padding
├── csv/                           # CSV tools
│   └── to_payload.py              # Convert CSV to payload
```

## Usage

Most tools are standalone CLI scripts. Run from the repo root:

```bash
# Standalone scripts (no internal imports)
python3 yolo/visualizer.py --help
python3 yolo/ds_merge.py --help
python3 yolo/filter_label.py --help
python3 csv/to_payload.py --help

# bg_augment (has subpackage imports, run as module)
python3 -m yolo.bg_augment --help
```

## Requirements

- Python 3.8+
- opencv-python (cv2)
- numpy
- tqdm
- pyyaml
