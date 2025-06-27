#!/usr/bin/env python3
"""Write a Cityscapes-34 categories JSON for evaluation.

The output format follows ``panoptic_coco_categories.json`` with ``id``,
``name`` and ``color`` fields so ``evaluation.py`` can display correct
class names and colors when using CS-34 masks.
"""

import argparse
import json
import pathlib

# Cityscapes labelIds with their standard names and RGB colors
CS34_CATEGORIES = [
    {"id": 0, "name": "unlabeled", "color": [0, 0, 0]},
    {"id": 1, "name": "ego vehicle", "color": [0, 0, 0]},
    {"id": 2, "name": "rectification border", "color": [0, 0, 0]},
    {"id": 3, "name": "out of roi", "color": [0, 0, 0]},
    {"id": 4, "name": "static", "color": [0, 0, 0]},
    {"id": 5, "name": "dynamic", "color": [111, 74, 0]},
    {"id": 6, "name": "ground", "color": [81, 0, 81]},
    {"id": 7, "name": "road", "color": [128, 64, 128]},
    {"id": 8, "name": "sidewalk", "color": [244, 35, 232]},
    {"id": 9, "name": "parking", "color": [250, 170, 160]},
    {"id": 10, "name": "rail track", "color": [230, 150, 140]},
    {"id": 11, "name": "building", "color": [70, 70, 70]},
    {"id": 12, "name": "wall", "color": [102, 102, 156]},
    {"id": 13, "name": "fence", "color": [190, 153, 153]},
    {"id": 14, "name": "guard rail", "color": [180, 165, 180]},
    {"id": 15, "name": "bridge", "color": [150, 100, 100]},
    {"id": 16, "name": "tunnel", "color": [150, 120, 90]},
    {"id": 17, "name": "pole", "color": [153, 153, 153]},
    {"id": 18, "name": "polegroup", "color": [153, 153, 153]},
    {"id": 19, "name": "traffic light", "color": [250, 170, 30]},
    {"id": 20, "name": "traffic sign", "color": [220, 220, 0]},
    {"id": 21, "name": "vegetation", "color": [107, 142, 35]},
    {"id": 22, "name": "terrain", "color": [152, 251, 152]},
    {"id": 23, "name": "sky", "color": [70, 130, 180]},
    {"id": 24, "name": "person", "color": [220, 20, 60]},
    {"id": 25, "name": "rider", "color": [255, 0, 0]},
    {"id": 26, "name": "car", "color": [0, 0, 142]},
    {"id": 27, "name": "truck", "color": [0, 0, 70]},
    {"id": 28, "name": "bus", "color": [0, 60, 100]},
    {"id": 29, "name": "caravan", "color": [0, 0, 90]},
    {"id": 30, "name": "trailer", "color": [0, 0, 110]},
    {"id": 31, "name": "train", "color": [0, 80, 100]},
    {"id": 32, "name": "motorcycle", "color": [0, 0, 230]},
    {"id": 33, "name": "bicycle", "color": [119, 11, 32]},
]


def main():
    parser = argparse.ArgumentParser(
        description="Create a panoptic_cs34_categories.json file",
    )
    parser.add_argument(
        "--dst",
        type=pathlib.Path,
        default=pathlib.Path("annotations/panoptic_cs34_categories.json"),
        help="Output path for the JSON file",
    )
    args = parser.parse_args()

    args.dst.parent.mkdir(parents=True, exist_ok=True)

    with open(args.dst, "w") as f:
        json.dump(CS34_CATEGORIES, f, indent=2)
    print(f"Wrote {args.dst}")


if __name__ == "__main__":
    main()
