import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from panopticapi.utils import rgb2id

from .Coco import Coco
from settings import DATA_PATH


class CocoPanoptic(Coco):
    """COCO dataset with panoptic segmentation support.

    When ``load_panoptic`` is enabled, ``segmentation_mask`` is returned as a
    ``torch.long`` tensor of shape ``(H, W)``.
    """

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
                if not self.panoptic_root.exists():
                    logging.warning('Panoptic folder missing: %s', self.panoptic_root)
                for ann in data['annotations']:
                    seg_map = {seg['id']: seg['category_id'] for seg in ann['segments_info']}
                    self.panoptic_segments[ann['file_name']] = seg_map
            else:
                logging.warning('Panoptic annotation file not found: %s', ann_file)

    def __getitem__(self, index):
        # get sample from base class
        input_dict = super().__getitem__(index)

        if self.config.get('load_panoptic', False) and self.panoptic_root is not None:
            sample = self.samples[index]
            image_name = Path(sample['image']).with_suffix('.png').name
            pan_path = self.panoptic_root / image_name
            H, W = input_dict['image'].shape[-2:]
            seg_mask = torch.zeros((H, W), dtype=torch.long)  # fallback mask prevents KeyError
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
                    max_val = int(cat_map.max())
                    if max_val >= num_cls:
                        logging.warning(
                            "Segmentation label %d exceeds num_segmentation_classes=%d; clipping",
                            max_val,
                            num_cls,
                        )
                    cat_map = np.clip(cat_map, 0, num_cls - 1)

                seg_mask = torch.tensor(cat_map, dtype=torch.long)
                
            else:
                # skip mask if panoptic file missing
                logging.warning('Missing panoptic file for image %s', image_name)
                # seg_mask stays all zeros so later code can access segmentation_mask safely

            input_dict['segmentation_mask'] = seg_mask

        return input_dict
