"""
Interactive YOLO dataset visualizer with bounding box overlays.

Browse through images using keyboard navigation. Type image name in CLI to jump.

Usage:
  python yolo_visualizer.py <dataset_path> [options]

Arguments:
  dataset              YOLO dataset root (contains data.yaml)

Options:
  --split SPLIT        Split to view: train/valid/test (default: train)
  --start NAME         Image name to start from
  --max-window W H     Max window size (default: auto-detect screen)
  --box-thickness N    Line thickness (default: 2)
  --font-scale F       Label font scale (default: 0.6)
  --no-labels          Hide class labels

Examples:
  python yolo_visualizer.py /path/to/yolo/dataset
  python yolo_visualizer.py /path/to/dataset --split valid
  python yolo_visualizer.py /path/to/dataset --start image_001.jpg
  python yolo_visualizer.py /path/to/dataset --max-window 1280 720 --no-labels

Keyboard controls:
  Right / d / n  - Next image
  Left / a / p   - Previous image
  Home / h       - First image
  End / e        - Last image
  l              - Toggle labels
  q / Escape     - Quit

CLI input:
  Type image name (partial match supported) and press Enter to jump.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

WINDOW_NAME = "YOLO Visualizer"


@dataclass
class YoloBox:
    class_id: int
    xc: float
    yc: float
    w: float
    h: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive YOLO dataset visualizer with bounding box overlays."
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="YOLO dataset root (contains data.yaml).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split to view: train/valid/test (default: train).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Image name to start from.",
    )
    parser.add_argument(
        "--max-window",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=None,
        help="Max window size (default: auto-detect screen).",
    )
    parser.add_argument(
        "--box-thickness",
        type=int,
        default=2,
        help="Bounding box line thickness (default: 2).",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=0.6,
        help="Label font scale (default: 0.6).",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Hide class labels.",
    )
    return parser.parse_args()


def load_class_names(data_yaml: Path) -> List[str]:
    """Load class names from data.yaml."""
    if not data_yaml.exists():
        return []
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if isinstance(names, dict):
        max_idx = max(names.keys()) if names else -1
        result = [""] * (max_idx + 1)
        for idx, name in names.items():
            result[idx] = name
        return result
    if isinstance(names, list):
        return names
    return []


def collect_images(dataset: Path, split: str) -> List[Path]:
    """Collect all images from the split, sorted by name."""
    images_dir = dataset / split / "images"
    if not images_dir.exists():
        return []
    images = [
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return sorted(images, key=lambda p: p.name.lower())


def get_label_path(image_path: Path) -> Path:
    """Get the corresponding label file path for an image."""
    parts = list(image_path.parts)
    for i, part in enumerate(parts):
        if part == "images":
            parts[i] = "labels"
            break
    label_path = Path(*parts).with_suffix(".txt")
    return label_path


def parse_label_file(path: Path) -> List[YoloBox]:
    """Parse YOLO format label file."""
    boxes: List[YoloBox] = []
    if not path.exists():
        return boxes
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
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
            boxes.append(YoloBox(class_id, xc, yc, w, h))
    return boxes


def yolo_to_xyxy(box: YoloBox, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Convert YOLO normalized coords to pixel coords (x1, y1, x2, y2)."""
    x1 = int(round((box.xc - box.w / 2.0) * img_w))
    y1 = int(round((box.yc - box.h / 2.0) * img_h))
    x2 = int(round((box.xc + box.w / 2.0) * img_w))
    y2 = int(round((box.yc + box.h / 2.0) * img_h))
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))
    return x1, y1, x2, y2


def get_screen_size() -> Tuple[int, int]:
    """Auto-detect screen size, fallback to 1920x1080."""
    try:
        from screeninfo import get_monitors
        monitors = get_monitors()
        if monitors:
            m = monitors[0]
            return m.width, m.height
    except ImportError:
        pass
    except Exception:
        pass
    return 1920, 1080


def calculate_display_size(
    img_w: int, img_h: int, max_w: int, max_h: int
) -> Tuple[int, int]:
    """Scale image to fit within max dimensions while preserving aspect ratio."""
    if img_w <= max_w and img_h <= max_h:
        return img_w, img_h
    scale_w = max_w / img_w
    scale_h = max_h / img_h
    scale = min(scale_w, scale_h)
    return int(img_w * scale), int(img_h * scale)


def generate_colors(n: int) -> List[Tuple[int, int, int]]:
    """Generate n distinct colors using HSV colorspace."""
    colors: List[Tuple[int, int, int]] = []
    for i in range(n):
        hue = int(180 * i / max(n, 1))
        hsv = np.array([[[hue, 255, 220]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        b, g, r = int(bgr[0, 0, 0]), int(bgr[0, 0, 1]), int(bgr[0, 0, 2])
        colors.append((b, g, r))
    return colors


def draw_boxes(
    image: np.ndarray,
    boxes: List[YoloBox],
    class_names: List[str],
    colors: List[Tuple[int, int, int]],
    thickness: int,
    font_scale: float,
    show_labels: bool,
) -> np.ndarray:
    """Draw bounding boxes and labels on image."""
    img = image.copy()
    img_h, img_w = img.shape[:2]

    for box in boxes:
        x1, y1, x2, y2 = yolo_to_xyxy(box, img_w, img_h)
        color = colors[box.class_id % len(colors)] if colors else (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        if show_labels:
            if box.class_id < len(class_names):
                label = class_names[box.class_id]
            else:
                label = str(box.class_id)

            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            label_y = y1 - 5 if y1 - text_h - 5 > 0 else y2 + text_h + 5
            label_x = x1

            cv2.rectangle(
                img,
                (label_x, label_y - text_h - baseline),
                (label_x + text_w, label_y + baseline),
                color,
                -1,
            )
            brightness = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
            text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)
            cv2.putText(
                img, label, (label_x, label_y), font, font_scale, text_color, thickness
            )

    return img


def build_name_index(images: List[Path]) -> Dict[str, int]:
    """Build index mapping image stems to their indices."""
    index: Dict[str, int] = {}
    for i, path in enumerate(images):
        index[path.stem.lower()] = i
    return index


def find_image_index(
    query: str, images: List[Path], name_index: Dict[str, int]
) -> Optional[int]:
    """Find image index by name (exact, prefix, or contains match)."""
    query_lower = query.lower().strip()
    if not query_lower:
        return None

    stem_query = Path(query_lower).stem

    if stem_query in name_index:
        return name_index[stem_query]

    for i, path in enumerate(images):
        if path.stem.lower().startswith(stem_query):
            return i

    for i, path in enumerate(images):
        if stem_query in path.stem.lower():
            return i

    return None


def cli_input_thread(input_queue: queue.Queue) -> None:
    """Background thread to read CLI input."""
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            input_queue.put(line.strip())
        except EOFError:
            break
        except Exception:
            break


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()

    if not dataset.exists():
        print(f"Error: Dataset path does not exist: {dataset}")
        sys.exit(1)

    class_names = load_class_names(dataset / "data.yaml")
    images = collect_images(dataset, args.split)

    if not images:
        print(f"Error: No images found in {dataset / args.split / 'images'}")
        sys.exit(1)

    print(f"Loaded {len(images)} images from {args.split} split")
    print(f"Classes: {len(class_names)}")

    name_index = build_name_index(images)

    current_idx = 0
    if args.start:
        found = find_image_index(args.start, images, name_index)
        if found is not None:
            current_idx = found
            print(f"Starting from: {images[current_idx].name}")
        else:
            print(f"Warning: Image '{args.start}' not found, starting from first image")

    if args.max_window:
        max_w, max_h = args.max_window
    else:
        screen_w, screen_h = get_screen_size()
        max_w = int(screen_w * 0.9)
        max_h = int(screen_h * 0.9)

    num_classes = max(len(class_names), 1)
    colors = generate_colors(num_classes)
    show_labels = not args.no_labels

    input_queue: queue.Queue = queue.Queue()
    input_thread = threading.Thread(target=cli_input_thread, args=(input_queue,), daemon=True)
    input_thread.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    print("\nControls:")
    print("  Right/d/n - Next | Left/a/p - Previous")
    print("  Home/h - First | End/e - Last")
    print("  l - Toggle labels | q/Esc - Quit")
    print("\nType image name and press Enter to jump.\n")

    last_idx = -1
    running = True

    while running:
        if current_idx != last_idx:
            image_path = images[current_idx]
            label_path = get_label_path(image_path)

            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Error reading image: {image_path}")
                current_idx = (current_idx + 1) % len(images)
                continue

            boxes = parse_label_file(label_path)

            display = draw_boxes(
                image,
                boxes,
                class_names,
                colors,
                args.box_thickness,
                args.font_scale,
                show_labels,
            )

            img_h, img_w = display.shape[:2]
            disp_w, disp_h = calculate_display_size(img_w, img_h, max_w, max_h)
            cv2.resizeWindow(WINDOW_NAME, disp_w, disp_h)

            cv2.imshow(WINDOW_NAME, display)

            print(f"[{current_idx + 1}/{len(images)}] {image_path.name} ({len(boxes)} boxes)")
            last_idx = current_idx

        try:
            while not input_queue.empty():
                query = input_queue.get_nowait()
                if query:
                    found = find_image_index(query, images, name_index)
                    if found is not None:
                        current_idx = found
                        print(f"Jumping to: {images[current_idx].name}")
                    else:
                        print(f"Image not found: {query}")
        except queue.Empty:
            pass

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q") or key == 27:
            running = False
        elif key == ord("d") or key == ord("n") or key == 83 or key == 3:
            current_idx = (current_idx + 1) % len(images)
        elif key == ord("a") or key == ord("p") or key == 81 or key == 2:
            current_idx = (current_idx - 1) % len(images)
        elif key == ord("h") or key == 80 or key == 106:
            current_idx = 0
        elif key == ord("e") or key == 87 or key == 107:
            current_idx = len(images) - 1
        elif key == ord("l"):
            show_labels = not show_labels
            last_idx = -1
            print(f"Labels: {'ON' if show_labels else 'OFF'}")

        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            running = False

    cv2.destroyAllWindows()
    print("\nViewer closed.")


if __name__ == "__main__":
    main()
