import logging
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.utils.data as data

# for generating warped pairs
from utils.homographies import sample_homography_np
from utils.utils import inv_warp_image, compute_valid_mask
from datasets.data_tools import warpLabels, np_to_tensor
from utils.var_dim import squeezeToNumpy

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
            'resize': [240, 320]
        },
        'num_parallel_calls': 10,
        # optional random homography generation for descriptor export
        'warped_pair': {
            'enable': False,
            'params': {},
            'valid_border_margin': 0,
        },
        # gaussian heatmap generation for keypoints
        'gaussian_label': {
            'enable': False,
            'params': {},
        },
    }

    def __init__(self, transform=None, task='train', **config):
        """Initialize dataset by crawling Cityscapes folders."""
        self.config = dict_update(self.default_config, config)
        self.transforms = transform
        self.split = 'train' if task == 'train' else 'val'

        # enable keypoint labels when path provided
        self.labels = False
        if self.config.get('labels'):
            self.labels = True

        # gaussian heatmap flag
        self.gaussian_label = self.config.get('gaussian_label', {}).get('enable', False)

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
            sample = {
                'image': str(img_path),
                'mask': str(mask_path),
                'name': name,
                # city identifier used as scene name for export
                'scene_name': rel.parent.name,
            }
            if self.labels:
                label_path = Path(self.config['labels'], self.split, f"{name}.npz")
                sample['points'] = str(label_path)
            self.samples.append(sample)

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

        output = {
            'image': image_tensor,
            'image_2D': image_tensor,
            'name': sample['name'],
            # scene identifier required by some export scripts
            'scene_name': sample['scene_name'],
        }

        H, W = image_tensor.shape[-2:]

        # load keypoint labels if available
        pnts = None
        if self.labels:
            pnts = np.load(sample['points'])['pts']
            labels = self.points_to_2D(pnts, H, W)
            output['labels_2D'] = torch.tensor(labels, dtype=torch.float32).unsqueeze(0)
            output['labels_res'] = torch.zeros((2, H, W), dtype=torch.float32)
            if self.gaussian_label:
                labels_gaussian = self.gaussian_blur(squeezeToNumpy(output['labels_2D']))
                output['labels_2D_gaussian'] = np_to_tensor(labels_gaussian, H, W)

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

        # optionally generate a warped pair and provide fields compatible with
        # the training pipeline
        if self.config.get('warped_pair', {}).get('enable', False):
            H, W = image_tensor.shape[-2:]
            # sample homography mapping warped image to original
            homo_inv = sample_homography_np(
                np.array([H, W]), shift=-1,
                **self.config['warped_pair'].get('params', {})
            )
            # invert to obtain transformation from original to warped
            homography = np.linalg.inv(homo_inv)
            # warp original image using the inverse matrix
            warped = inv_warp_image(
                image_tensor.squeeze(0),
                torch.tensor(homo_inv, dtype=torch.float32),
            )
            # store both naming conventions for compatibility
            output['warped_image'] = warped.unsqueeze(0)
            output['warped_img'] = output['warped_image']
            # homographies in both directions for descriptor loss
            H_mat = torch.tensor(homography, dtype=torch.float32)
            H_inv_mat = torch.tensor(homo_inv, dtype=torch.float32)
            output['homography'] = H_mat
            output['homographies'] = H_mat.unsqueeze(0)
            output['inv_homographies'] = H_inv_mat.unsqueeze(0)
            # valid mask used when computing descriptor loss
            margin = self.config['warped_pair'].get('valid_border_margin', 0)
            valid_mask = compute_valid_mask(torch.tensor([H, W]), H_inv_mat, erosion_radius=margin)
            output['warped_valid_mask'] = valid_mask
            # print('[compute_valid_mask] sum:', valid_mask.sum().item(), 'min/max:', valid_mask.min().item(), valid_mask.max().item())

            # warp keypoint labels when available
            if self.labels:
                warped_set = warpLabels(pnts, H, W, torch.tensor(homography, dtype=torch.float32), bilinear=True)
                output['warped_labels'] = warped_set['labels']
                warped_res = warped_set['res'].transpose(1,2).transpose(0,1)
                output['warped_res'] = warped_res
                if self.gaussian_label:
                    warped_labels_gaussian = self.gaussian_blur(squeezeToNumpy(warped_set['labels_bi']))
                    output['warped_labels_gaussian'] = np_to_tensor(warped_labels_gaussian, H, W)
                    output['warped_labels_bi'] = warped_set['labels_bi']

        return output

    @staticmethod
    def points_to_2D(pnts, H, W):
        labels = np.zeros((H, W))
        pnts = pnts.astype(int)
        labels[pnts[:, 1], pnts[:, 0]] = 1
        return labels

    def gaussian_blur(self, image):
        """Apply Gaussian blur augmentation to generate heatmaps."""
        from utils.photometric import ImgAugTransform
        aug_par = {'photometric': {}}
        aug_par['photometric']['enable'] = True
        aug_par['photometric']['params'] = self.config['gaussian_label']['params']
        augmentation = ImgAugTransform(**aug_par)
        image = image[:, :, np.newaxis]
        heatmaps = augmentation(image)
        return heatmaps.squeeze()
