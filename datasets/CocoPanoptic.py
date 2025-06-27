import json
from pathlib import Path

import cv2
import numpy as np
import torch

from panopticapi.utils import rgb2id

from .Coco import Coco
from settings import DATA_PATH


class CocoPanoptic(Coco):
    """COCO dataset with panoptic segmentation support."""

    default_config = Coco.default_config.copy()
    # extra switch to enable panoptic loading
    default_config.update({'load_panoptic': False})

    def __init__(self, export=False, transform=None, task='train', **config):
        # turn off instance masks when panoptic segmentation is requested
        if config.get('load_panoptic'):
            config['load_segmentation'] = False

        # initialize base Coco dataset
        super().__init__(export=export, transform=transform, task=task, **config)

        self.panoptic_segments = {}
        self.panoptic_root = None

        if self.config.get('load_panoptic', False):
            # panoptic annotations json
            ann_file = Path(DATA_PATH, 'COCO/annotations', f'panoptic_{task}2017.json')
            if ann_file.exists():
                with open(ann_file, 'r') as f:
                    data = json.load(f)
                self.panoptic_root = ann_file.parent / f'panoptic_{task}2017'
                for ann in data['annotations']:
                    seg_map = {seg['id']: seg['category_id'] for seg in ann['segments_info']}
                    self.panoptic_segments[ann['file_name']] = seg_map
            else:
                print(f'Panoptic annotation file not found: {ann_file}')

    def __getitem__(self, index):
        # get sample from base class
        input_dict = super().__getitem__(index)

        if self.config.get('load_panoptic', False) and self.panoptic_root is not None:
            sample = self.samples[index]
            image_name = Path(sample['image']).with_suffix('.png').name
            pan_path = self.panoptic_root / image_name
            H, W = input_dict['image'].shape[-2:]
            if pan_path.exists():
                pan_img = cv2.imread(str(pan_path), cv2.IMREAD_COLOR)
                # convert BGR image returned by OpenCV to RGB for rgb2id
                pan_img = cv2.cvtColor(pan_img, cv2.COLOR_BGR2RGB)
                seg_ids = rgb2id(pan_img)
                cat_map = np.zeros_like(seg_ids, dtype=np.int32)
                mapping = self.panoptic_segments.get(image_name, {})
                for seg_id, cat_id in mapping.items():
                    cat_map[seg_ids == seg_id] = cat_id
                cat_map = cv2.resize(cat_map, (W, H), interpolation=cv2.INTER_NEAREST)
                num_cls = self.config.get('num_segmentation_classes', 0)
                if num_cls > 0:
                    cat_map = np.clip(cat_map, 0, num_cls - 1)
                input_dict['segmentation_mask'] = torch.tensor(cat_map, dtype=torch.long)
            else:
                # skip mask if panoptic file missing
                pass

        return input_dict
