# Convert COCO panoptic masks to Cityscapes-34 labels
# Usage: python coco_panoptic_to_cs34.py --src panoptic_val2017 --dst cs34_masks
# The script reads each COCO panoptic PNG and writes a grayscale mask with
# Cityscapes labelIds. Unknown categories become 0.

import argparse
import cv2
import numpy as np
import pathlib
from panopticapi.utils import rgb2id
import tqdm

# Mapping from COCO category id to Cityscapes-34 labelId
COCO_TO_CS34 = {
    # vehicles
    3: 26,  # car
    8: 27,  # truck
    6: 28,  # bus
    7: 31,  # train
    4: 32,  # motorcycle
    2: 33,  # bicycle
    9: 5,   # boat -> dynamic
    5: 5,   # airplane -> dynamic

    # humans
    1: 24,  # person
    18: 25, 19: 25, 24: 25, 25: 25,  # dog/horse/zebra/giraffe -> rider
    # accessories -> dynamic (labelId 5)
    27: 5, 28: 5, 31: 5, 32: 5, 33: 5,
    34: 5, 35: 5, 36: 5, 37: 5, 38: 5,
    39: 5, 40: 5, 41: 5, 42: 5, 43: 5,

    # traffic objects
    10: 19,  # traffic light
    11: 17,  # fire hydrant -> pole
    13: 20,  # stop sign
    14: 17, 15: 17,  # parking meter, bench -> pole

    # ground / construction / nature
    149: 7,  # road
    144: 9, 147: 10, 190: 6, 191: 9,  # platform -> parking; rail track; ground
    128: 11, 151: 11, 197: 11,  # house / building
    171: 12, 175: 12, 176: 12, 177: 12,  # walls
    185: 13, 133: 13,  # fence / mirror
    14: 14,  # guard rail
    95: 15,  # bridge
    166: 16,  # tent -> tunnel
    17: 21, 184: 21, 193: 21, 119: 21,  # vegetation / tree / grass / flower
    125: 22, 154: 22, 194: 22,  # gravel / sand / dirt -> terrain
    187: 23, 178: 23,  # sky
    148: 5, 155: 5,  # river / sea -> dynamic water

    # poles & signs groups
    112: 17, 138: 17, 180: 17, 181: 17,  # doors/net/window-blind -> polegroup
    100: 4, 122: 4, 196: 4,  # cardboard / fruit / food-other -> static
}

COCO_TO_CS34_DEFAULT = 0


def remap_png(coco_png_path: pathlib.Path, out_path: pathlib.Path):
    """Convert a COCO panoptic mask to Cityscapes-34 labelIds."""
    coco_rgb = cv2.imread(str(coco_png_path))[:, :, ::-1]  # BGR -> RGB
    coco_id32 = rgb2id(coco_rgb).astype(np.int32)

    # initialize output mask with default value 0
    cs_mask = np.full_like(coco_id32, COCO_TO_CS34_DEFAULT, dtype=np.uint8)
    for coco_id, cs_id in COCO_TO_CS34.items():
        cs_mask[coco_id32 == coco_id] = cs_id

    cv2.imwrite(str(out_path), cs_mask)


def main():
    parser = argparse.ArgumentParser(description="Remap COCO panoptic masks to Cityscapes-34 labels")
    parser.add_argument("--src", type=pathlib.Path, required=True, help="Folder with COCO panoptic PNGs")
    parser.add_argument("--dst", type=pathlib.Path, required=True, help="Output folder for CS-34 masks")
    args = parser.parse_args()

    args.dst.mkdir(exist_ok=True)
    for fn in tqdm.tqdm(list(args.src.glob("*.png"))):
        remap_png(fn, args.dst / fn.name)


if __name__ == "__main__":
    main()
