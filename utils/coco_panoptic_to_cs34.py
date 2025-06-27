# Convert COCO panoptic masks to Cityscapes-34 labels
# Usage example:
#   python coco_panoptic_to_cs34.py \
#       --src panoptic_val2017 \
#       --ann annotations/panoptic_val2017.json \
#       --categories annotations/panoptic_coco_categories.json \
#       --dst cs34_masks
#
# The script reads COCO panoptic PNGs together with their JSON annotations and
# writes grayscale masks with Cityscapes labelIds. Unknown categories become 0.

import argparse
import cv2
import json
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

def load_categories(cat_json: pathlib.Path):
    """Load COCO categories as a mapping from id to name."""
    with open(cat_json, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "categories" in data:
        data = data["categories"]
    return {c["id"]: c["name"] for c in data}


def load_segment_annotations(ann_json: pathlib.Path):
    """Return mapping from file_name to segment id -> category id."""
    with open(ann_json, "r") as f:
        data = json.load(f)
    mapping = {}
    for ann in data.get("annotations", []):
        seg_map = {seg["id"]: seg["category_id"] for seg in ann["segments_info"]}
        mapping[ann["file_name"]] = seg_map
    return mapping


def remap_png(
    coco_png_path: pathlib.Path,
    seg_mapping: dict,
    cat_map: dict,
    out_path: pathlib.Path,
):
    """Convert a COCO panoptic mask to Cityscapes-34 labelIds."""
    coco_rgb = cv2.imread(str(coco_png_path))[:, :, ::-1]  # BGR -> RGB
    coco_id32 = rgb2id(coco_rgb).astype(np.int32)

    # initialize output mask with default value 0
    cs_mask = np.full_like(coco_id32, COCO_TO_CS34_DEFAULT, dtype=np.uint8)
    for seg_id, cat_id in seg_mapping.items():
        cs_id = cat_map.get(cat_id, COCO_TO_CS34_DEFAULT)
        cs_mask[coco_id32 == seg_id] = cs_id

    cv2.imwrite(str(out_path), cs_mask)


def main():
    parser = argparse.ArgumentParser(
        description="Remap COCO panoptic masks to Cityscapes-34 labels"
    )
    parser.add_argument(
        "--src", type=pathlib.Path, required=True, help="Folder with COCO panoptic PNGs"
    )
    parser.add_argument(
        "--ann", type=pathlib.Path, required=True, help="Panoptic annotations JSON"
    )
    parser.add_argument(
        "--categories", type=pathlib.Path, required=True, help="COCO categories JSON"
    )
    parser.add_argument(
        "--dst", type=pathlib.Path, required=True, help="Output folder for CS-34 masks"
    )
    args = parser.parse_args()

    cat_names = load_categories(args.categories)
    cat_map = {cid: COCO_TO_CS34.get(cid, COCO_TO_CS34_DEFAULT) for cid in cat_names}
    seg_maps = load_segment_annotations(args.ann)

    args.dst.mkdir(exist_ok=True)
    for fn in tqdm.tqdm(list(args.src.glob("*.png"))):
        segments = seg_maps.get(fn.name)
        if segments is None:
            continue
        remap_png(fn, segments, cat_map, args.dst / fn.name)


if __name__ == "__main__":
    main()

