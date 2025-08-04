"""Resize Cityscapes ground-truth masks to a fixed resolution.

This utility downsamples the ``gtFine`` annotations to ``512x1024`` by default
and stores them in a separate folder (e.g. ``gtFine_1024``). Having masks at
the same resolution as the training images keeps predictions and labels
aligned, which improves metric accuracy and visualization clarity.
"""

import argparse
from pathlib import Path

import cv2


def resize_masks(root, out_dir="gtFine_1024", size=(512, 1024)):
    """Resize all label masks under ``root/gtFine``.

    Parameters
    ----------
    root : Path or str
        Path to the Cityscapes dataset containing the original ``gtFine``
        directory.
    out_dir : str
        Name of the folder to write resized masks to.
    size : tuple[int, int]
        Output size as ``(height, width)``.
    """

    root = Path(root)
    src = root / "gtFine"
    dst = root / out_dir

    for split in ["train", "val", "test"]:
        src_split = src / split
        dst_split = dst / split
        for mask_path in src_split.rglob("*_gtFine_labelIds.png"):
            rel = mask_path.relative_to(src_split)
            out_path = dst_split / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            # nearest-neighbor keeps label ids intact during resize
            resized = cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(out_path), resized)


def main():
    parser = argparse.ArgumentParser(description="Downscale Cityscapes gtFine masks")
    parser.add_argument("root", type=str, help="Path to Cityscapes dataset root")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="gtFine_1024",
        help="Folder name for resized masks",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Target mask height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Target mask width",
    )
    args = parser.parse_args()

    resize_masks(args.root, args.out_dir, size=(args.height, args.width))


if __name__ == "__main__":
    main()

