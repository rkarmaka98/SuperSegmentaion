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
    default_config.update({
        'load_panoptic': False,
        # when true, use grayscale CS-34 masks instead of RGB panoptic labels
        'use_cs34_masks': False,
    })

    def __init__(self, export=False, transform=None, task='train', **config):
        # turn off instance masks when panoptic segmentation is requested
        if config.get('load_panoptic'):
            config['load_segmentation'] = False

        # initialize base Coco dataset
        super().__init__(export=export, transform=transform, task=task, **config)

        self.panoptic_segments = {}
        self.panoptic_root = None  # original COCO panoptic RGB masks
        self.cs34_root = None  # grayscale CS-34 masks

        if self.config.get('load_panoptic', False):
            # allow explicit request for CS-34 grayscale masks via config
            if self.config.get('use_cs34_masks', False):
                cs34_root = Path(DATA_PATH, 'COCO', f'panoptic_cs34_{task}2017')
                if cs34_root.exists():
                    self.cs34_root = cs34_root
                    self.num_segmentation_classes = self.config.get('num_segmentation_classes', 34)
                else:
                    logging.warning('Requested CS-34 masks but folder not found: %s', cs34_root)
                return

            # otherwise try standard COCO panoptic setup first
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
                # fallback: look for CS-34 masks without JSON annotations
                cs34_root = Path(DATA_PATH, 'COCO', f'panoptic_cs34_{task}2017')
                if cs34_root.exists():
                    self.cs34_root = cs34_root
                    self.num_segmentation_classes = self.config.get('num_segmentation_classes', 34)
                else:
                    logging.warning('Panoptic annotation file not found: %s', ann_file)

    def __getitem__(self, index):
        # get sample from base class
        input_dict = super().__getitem__(index)

        if self.config.get('load_panoptic', False):
            sample = self.samples[index]
            image_name = Path(sample['image']).with_suffix('.png').name
            H, W = input_dict['image'].shape[-2:]

            if self.panoptic_root is not None:
                pan_path = self.panoptic_root / image_name
                seg_mask = torch.zeros((H, W), dtype=torch.long)
                if pan_path.exists():
                    pan_img = cv2.imread(str(pan_path), cv2.IMREAD_COLOR)
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
                    logging.warning('Missing panoptic file for image %s', image_name)

            elif self.cs34_root is not None:
                cs34_path = self.cs34_root / image_name
                seg_mask = torch.zeros((H, W), dtype=torch.long)
                if cs34_path.exists():
                    seg_img = cv2.imread(str(cs34_path), cv2.IMREAD_GRAYSCALE)
                    seg_img = cv2.resize(seg_img, (W, H), interpolation=cv2.INTER_NEAREST)
                    seg_mask = torch.tensor(seg_img, dtype=torch.long)
                    num_cls = self.config.get('num_segmentation_classes', 0)
                    if num_cls > 0:
                        max_val = int(seg_mask.max())
                        if max_val >= num_cls:
                            logging.warning(
                                "Segmentation label %d exceeds num_segmentation_classes=%d; clipping for cs34",
                                max_val,
                                num_cls,
                            )
                            seg_mask = torch.clamp(seg_mask, 0, num_cls - 1)
                else:
                    logging.warning('Missing cs34 file for image %s', image_name)
                
            else:
                seg_mask = torch.zeros((H, W), dtype=torch.long)

            input_dict['segmentation_mask'] = seg_mask

        return input_dict
