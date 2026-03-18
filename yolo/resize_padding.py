"""
Resize YOLO datasets to a fixed canvas with centered padding.

Each image is letterboxed to the target size (default 640x640) with black padding,
and label boxes are remapped to the new coordinates.

Example:
python yolo_resize_padding.py \\
  /path/to/source_dataset \\
  /path/to/output_dataset \\
  --img-size 640 640
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resize YOLO dataset with centered padding.")
    parser.add_argument("source", type=Path, help="Source YOLO dataset root.")
    parser.add_argument("dest", type=Path, help="Destination dataset root.")
    parser.add_argument(
        "--img-size",
        type=int,
        nargs="+",
        default=[640, 640],
        help="Output width height (one value applies to both). Default: 640 640.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_size(values: List[int]) -> Tuple[int, int]:
    if len(values) == 1:
        return values[0], values[0]
    if len(values) == 2:
        return values[0], values[1]
    raise ValueError("img-size expects one (square) or two values (width height)")


def parse_label_line(line: str) -> Optional[Tuple[int, float, float, float, float, List[str]]]:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        class_id = int(parts[0])
        x, y, w, h = map(float, parts[1:5])
    except ValueError:
        return None
    extras = parts[5:]
    return class_id, x, y, w, h, extras


def find_image(images_dir: Path, stem: str) -> Optional[Path]:
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def remap_box(
    x: float,
    y: float,
    w: float,
    h: float,
    orig_w: int,
    orig_h: int,
    scale: float,
    pad_x: int,
    pad_y: int,
    out_w: int,
    out_h: int,
) -> Optional[Tuple[float, float, float, float]]:
    cx = x * orig_w
    cy = y * orig_h
    bw = w * orig_w
    bh = h * orig_h

    cx = cx * scale + pad_x
    cy = cy * scale + pad_y
    bw = bw * scale
    bh = bh * scale

    if bw <= 0 or bh <= 0:
        return None

    new_x = cx / out_w
    new_y = cy / out_h
    new_w = bw / out_w
    new_h = bh / out_h

    # Clamp to valid range in case of tiny numeric drift.
    new_x = min(max(new_x, 0.0), 1.0)
    new_y = min(max(new_y, 0.0), 1.0)
    new_w = min(max(new_w, 0.0), 1.0)
    new_h = min(max(new_h, 0.0), 1.0)
    return new_x, new_y, new_w, new_h


def copy_metadata(src_root: Path, dest_root: Path) -> None:
    for name in ["data.yaml", "README.dataset.txt", "README.roboflow.txt"]:
        candidate = src_root / name
        if candidate.exists():
            ensure_dir(dest_root)
            dest = dest_root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(candidate.read_bytes())


def process_dataset(source_root: Path, dest_root: Path, out_w: int, out_h: int) -> None:
    stats = {
        "processed_labels": 0,
        "missing_image": 0,
        "labels_written": 0,
        "images_written": 0,
    }

    for dirpath, _, filenames in os.walk(source_root):
        if Path(dirpath).name != "labels":
            continue

        labels_dir = Path(dirpath)
        images_dir = labels_dir.parent / "images"

        for filename in filenames:
            if not filename.endswith(".txt"):
                continue
            stats["processed_labels"] += 1

            label_path = labels_dir / filename
            stem = Path(filename).stem

            image_path = find_image(images_dir, stem)
            if not image_path:
                stats["missing_image"] += 1
                continue

            with Image.open(image_path) as img:
                orig_w, orig_h = img.size
                scale = min(out_w / orig_w, out_h / orig_h)
                new_w = int(round(orig_w * scale))
                new_h = int(round(orig_h * scale))
                pad_x = (out_w - new_w) // 2
                pad_y = (out_h - new_h) // 2

                canvas = Image.new("RGB", (out_w, out_h), color=(0, 0, 0))
                canvas.paste(img.resize((new_w, new_h)), (pad_x, pad_y))

                dest_images_dir = dest_root / images_dir.relative_to(source_root)
                dest_labels_dir = dest_root / labels_dir.relative_to(source_root)
                ensure_dir(dest_images_dir)
                ensure_dir(dest_labels_dir)

                dest_image_path = dest_images_dir / image_path.name
                dest_label_path = dest_labels_dir / filename

                new_lines: List[str] = []
                with label_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        parsed = parse_label_line(line)
                        if not parsed:
                            continue
                        class_id, x, y, w, h, extras = parsed
                        mapped = remap_box(
                            x, y, w, h, orig_w, orig_h, scale, pad_x, pad_y, out_w, out_h
                        )
                        if not mapped:
                            continue
                        nx, ny, nw, nh = mapped
                        tokens = [
                            str(class_id),
                            f"{nx}",
                            f"{ny}",
                            f"{nw}",
                            f"{nh}",
                        ] + extras
                        new_lines.append(" ".join(tokens))

                canvas.save(dest_image_path)
                stats["images_written"] += 1

                with dest_label_path.open("w", encoding="utf-8") as out:
                    out.write("\n".join(new_lines) + ("\n" if new_lines else ""))
                stats["labels_written"] += 1

    print("--- Summary ---")
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').capitalize()}: {value}")
    print(f"Output: {dest_root}")


def main() -> None:
    args = parse_args()
    out_w, out_h = parse_size(args.img_size)
    source_root = args.source.resolve()
    dest_root = args.dest.resolve()
    ensure_dir(dest_root)

    copy_metadata(source_root, dest_root)
    process_dataset(source_root=source_root, dest_root=dest_root, out_w=out_w, out_h=out_h)


if __name__ == "__main__":
    main()
