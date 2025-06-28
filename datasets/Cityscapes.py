import logging
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.utils.data as data

from settings import DATA_PATH
from utils.tools import dict_update


# mapping from 34 Cityscapes labelIds to 4 broad categories
# 0 -> Static Structure, 1 -> Flat Surfaces,
# 2 -> Dynamic Objects, 3 -> Unstable/Ambiguous
CS34_TO_4 = {
    11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0,
    7: 1, 8: 1, 9: 1, 10: 1, 22: 1,
    24: 2, 25: 2, 26: 2, 27: 2, 28: 2, 29: 2, 30: 2, 31: 2, 32: 2, 33: 2,
    21: 3, 23: 3, 6: 3, 5: 3, 4: 3, 0: 3, 1: 3, 2: 3, 3: 3,
}


class Cityscapes(data.Dataset):
    """Dataset loader for Cityscapes images and semantic labels.

    ``segmentation_mask`` is returned as a ``torch.long`` tensor with shape
    ``(H, W)`` when ``load_segmentation`` is enabled.
    """

    # default configuration similar to Coco dataset
    default_config = {
        'labels': None,
        'segmentation_labels': None,
        'num_segmentation_classes': 34,
        # optionally map the 34 labelIds to 4 coarse categories
        'reduce_to_4_categories': False,
        'cache_in_memory': False,
        'validation_size': 100,
        'truncate': 0,
        'load_segmentation': True,
        'preprocessing': {
            'resize': [256, 512]
        },
        'num_parallel_calls': 10,
    }

    def __init__(self, transform=None, task='train', **config):
        """Initialize dataset by crawling Cityscapes folders."""
        self.config = dict_update(self.default_config, config)
        self.transforms = transform
        self.split = 'train' if task == 'train' else 'val'

        # root directory with leftImg8bit/ and gtFine/ folders
        self.root = Path(self.config.get('root', Path(DATA_PATH, 'Cityscapes')))
        img_root = self.root / 'leftImg8bit' / self.split
        self.mask_root = self.root / 'gtFine' / self.split

        image_paths = sorted(img_root.rglob('*_leftImg8bit.png'))
        if self.config.get('truncate'):
            image_paths = image_paths[: self.config['truncate']]

        self.samples = []
        for img_path in image_paths:
            rel = img_path.relative_to(img_root)
            name = img_path.stem.replace('_leftImg8bit', '')
            mask_name = img_path.stem.replace('_leftImg8bit', '_gtFine_labelIds.png')
            mask_path = self.mask_root / rel.parent / mask_name
            self.samples.append({'image': str(img_path), 'mask': str(mask_path), 'name': name})

        self.sizer = self.config['preprocessing']['resize']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        img = cv2.imread(sample['image'], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (self.sizer[1], self.sizer[0]), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        image_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        output = {'image': image_tensor, 'name': sample['name']}

        if self.config.get('load_segmentation', False):
            mask_path = Path(sample['mask'])
            H, W = image_tensor.shape[-2:]
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                seg_mask = torch.tensor(mask, dtype=torch.long)
                max_val = int(seg_mask.max())
                num_cls = int(self.config.get('num_segmentation_classes', 0))
                # if max_val >= num_cls > 0:
                #     logging.warning(
                #         "Segmentation label %d exceeds num_segmentation_classes=%d in %s",
                #         max_val, num_cls, mask_path,
                #     )
            else:
                logging.warning('Missing segmentation label file: %s', mask_path)
                seg_mask = torch.zeros((H, W), dtype=torch.long)
            if self.config.get('reduce_to_4_categories', False):
                # convert CS-34 labels to the 4-category scheme
                mask_np = seg_mask.numpy()
                mapped = np.full_like(mask_np, 3)
                for k, v in CS34_TO_4.items():
                    mapped[mask_np == k] = v
                seg_mask = torch.from_numpy(mapped)

            # semantic segmentation mask with dtype long and shape (H, W)
            output['segmentation_mask'] = seg_mask

        return output
