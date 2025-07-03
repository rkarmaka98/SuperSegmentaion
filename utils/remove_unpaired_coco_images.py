#!/usr/bin/env python3
"""Delete COCO train images without a matching CS-34 annotation."""

import argparse
from pathlib import Path

# Supported image extensions in COCO
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def main():
    parser = argparse.ArgumentParser(
        description="Delete images in train2017 that lack a CS-34 mask"
    )
    parser.add_argument(
        "--coco-root",
        type=Path,
        default=Path("datasets/COCO"),
        help="Root folder of the COCO dataset",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be removed without deleting them",
    )
    args = parser.parse_args()

    images_root = args.coco_root / "train2017"
    masks_root = args.coco_root / "panoptic_cs34_train2017"

    if not images_root.exists():
        print(f"Image folder not found: {images_root}")
        return

    removed = 0
    for img_path in images_root.iterdir():
        if img_path.suffix.lower() not in IMG_EXTS:
            continue  # skip non-image files
        mask_path = masks_root / (img_path.stem + ".png")
        if not mask_path.exists():
            if args.dry_run:
                print(f"Would remove {img_path}")
            else:
                print(f"Removing {img_path}")
                img_path.unlink()
            removed += 1

    print(f"Total images removed: {removed}")


if __name__ == "__main__":
    main()
