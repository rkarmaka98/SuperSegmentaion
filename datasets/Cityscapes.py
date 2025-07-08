import logging
from pathlib import Path
import json  # for reading camera files
import cv2
import numpy as np
import torch
import torch.utils.data as data

# for generating warped pairs
# planar homography sampler used for homography adaptation
from utils.homographies import sample_homography_np
# utils for warping and computing valid masks
from utils.utils import inv_warp_image, compute_valid_mask, inv_warp_image_batch
from datasets.data_tools import warpLabels, np_to_tensor
from utils.var_dim import squeezeToNumpy

from settings import DATA_PATH
from utils.tools import dict_update

# Mean and standard deviation of RGB channels computed on the Cityscapes dataset
CITYSCAPES_MEAN = (0.28689554, 0.32513303, 0.28389177)
CITYSCAPES_STD = (0.18696375, 0.19017339, 0.18720214)


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
        # optional data augmentation similar to the COCO loader
        'augmentation': {
            'photometric': {
                'enable': False,
                'primitives': 'all',
                'params': {},
                'random_order': True,
            },
            'homographic': {
                'enable': False,
                'params': {},
                'valid_border_margin': 0,
            },
        },
        # optional random homography generation for descriptor export
        'warped_pair': {
            'enable': False,
            'params': {},
            'valid_border_margin': 0,
        },
        # enable homography adaptation at export time
        'homography_adaptation': {
            'enable': False
        },
        # gaussian heatmap generation for keypoints
        'gaussian_label': {
            'enable': False,
            'params': {},
        },
    }

    # To recompute the dataset mean and standard deviation run the following
    # snippet. It loads the dataset with the same preprocessing parameters and
    # iterates over all images to aggregate statistics.
    #
    # from datasets.Cityscapes import Cityscapes
    # from torch.utils.data import DataLoader
    # import numpy as np
    #
    # ds = Cityscapes(root='/path/to/Cityscapes', load_segmentation=False)
    # loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    # means, stds = [], []
    # for batch in loader:
    #     img = batch['image']
    #     means.append(img.mean().item())
    #     stds.append(img.std().item())
    # print('mean:', np.mean(means))
    # print('std:', np.mean(stds))

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
            camera_dir = self.root / 'camera' / self.split / rel.parent
            json_path = camera_dir / f'{name}_camera.json'
            txt_path = camera_dir / f'{name}_camera.txt'
            # store whichever camera file exists
            if json_path.exists():
                cam_file = json_path
            elif txt_path.exists():
                cam_file = txt_path
            else:
                logging.warning('Missing camera file for %s', img_path)
                cam_file = None
            sample = {
                'image': str(img_path),
                'mask': str(mask_path),
                'name': name,
                # city identifier used as scene name for export
                'scene_name': rel.parent.name,
                'camera': str(cam_file) if cam_file is not None else None,
            }
            if self.labels:
                label_path = Path(self.config['labels'], self.split, f"{name}.npz")
                sample['points'] = str(label_path)
            self.samples.append(sample)

        self.sizer = self.config['preprocessing']['resize']

        # expose utils as attributes for easier access in __getitem__
        self.inv_warp_image_batch = inv_warp_image_batch

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        img = cv2.imread(sample['image'], cv2.IMREAD_COLOR)
        # resize and keep RGB information
        img = cv2.resize(img, (self.sizer[1], self.sizer[0]), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        # convert BGR to RGB for consistency with mean/std
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # normalize each channel
        img = (img - CITYSCAPES_MEAN) / CITYSCAPES_STD
        # torch expects channel-first tensors
        image_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()

        output = {
            'image': image_tensor,
            'name': sample['name'],
            # scene identifier required by some export scripts
            'scene_name': sample['scene_name'],
        }

        H, W = image_tensor.shape[-2:]

        # always provide a valid mask for the current image
        valid_mask = compute_valid_mask(torch.tensor([H, W]), torch.eye(3))
        output['valid_mask'] = valid_mask

        # camera intrinsics and extrinsics for this frame
        if 'K' not in sample:
            K, R_C_to_V, t_C_to_V = self._load_camera_matrices(sample.get('camera'))
            try:
                # standard inverse of intrinsics
                K_inv = np.linalg.inv(K)
            except np.linalg.LinAlgError:
                # fallback to pseudo-inverse when K is singular
                logging.warning('Camera intrinsics singular; using pseudo-inverse')
                K_inv = np.linalg.pinv(K)
            R_V_to_C = R_C_to_V.T
            t_V_to_C = -R_V_to_C @ t_C_to_V
            Rt = np.hstack((R_C_to_V, t_C_to_V.reshape(3, 1)))
            sample.update({
                'K': K,
                'K_inv': K_inv,
                'R_C_to_V': R_C_to_V,
                't_C_to_V': t_C_to_V,
                'R_V_to_C': R_V_to_C,
                't_V_to_C': t_V_to_C,
                'P': K @ Rt,
            })
        K = sample['K']
        K_inv = sample['K_inv']
        output['K'] = torch.from_numpy(K)
        output['K_inv'] = torch.from_numpy(K_inv)
        output['R_C_to_V'] = torch.from_numpy(sample['R_C_to_V'])
        output['t_C_to_V'] = torch.from_numpy(sample['t_C_to_V'])
        output['R_V_to_C'] = torch.from_numpy(sample['R_V_to_C'])
        output['t_V_to_C'] = torch.from_numpy(sample['t_V_to_C'])
        output['P'] = torch.from_numpy(sample['P'])

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

        # orientation-based augmentation of image and labels
        if self.config.get('augmentation', {}).get('homographic', {}).get('enable', False):
            params = self.config['augmentation']['homographic'].get('params', {})
            R_perturb = self._sample_rotation(params)
            homography = sample['K'] @ R_perturb @ sample['K_inv']
            try:
                # standard inverse for augmentation
                homo_inv = np.linalg.inv(homography)
            except np.linalg.LinAlgError:
                logging.warning('Augmentation homography singular; using pseudo-inverse')
                homo_inv = np.linalg.pinv(homography)
            image_tensor = inv_warp_image_batch(
                image_tensor.unsqueeze(0),
                torch.tensor(homo_inv, dtype=torch.float32),
                mode='bilinear',
            ).squeeze(0)
            valid_mask = compute_valid_mask(
                torch.tensor([H, W]),
                torch.tensor(homo_inv, dtype=torch.float32),
                erosion_radius=self.config['augmentation']['homographic']['valid_border_margin'],
            )
            output['valid_mask'] = valid_mask
            if 'segmentation_mask' in output:
                seg_warp = inv_warp_image_batch(
                    output['segmentation_mask'].float().unsqueeze(0),
                    torch.tensor(homo_inv, dtype=torch.float32),
                    mode='nearest',
                ).squeeze(0).long()
                output['segmentation_mask'] = seg_warp
            if self.labels:
                warped_set = warpLabels(pnts, H, W, torch.tensor(homography, dtype=torch.float32), bilinear=True)
                output['labels_2D'] = warped_set['labels']
                output['labels_res'] = warped_set['res'].transpose(1,2).transpose(0,1)
                if self.gaussian_label:
                    warped_labels_gaussian = self.gaussian_blur(squeezeToNumpy(warped_set['labels_bi']))
                    output['labels_2D_gaussian'] = np_to_tensor(warped_labels_gaussian, H, W)
            output['image'] = image_tensor

        # homography adaptation to generate multiple warped views
        if self.config.get('homography_adaptation', {}).get('enable', False):
            homoAdapt_iter = self.config['homography_adaptation']['num']
            homographies = np.stack([
                sample_homography_np(
                    np.array([H, W]), shift=-1,
                    **self.config['homography_adaptation']['homographies']['params']
                )
                for _ in range(homoAdapt_iter)
            ])
            # use inverse homographies as defined by the loader
            inv_list = []
            for h in homographies:
                try:
                    inv_list.append(np.linalg.inv(h))
                except np.linalg.LinAlgError:
                    logging.warning('Homography adaptation singular; using pseudo-inverse')
                    inv_list.append(np.linalg.pinv(h))
            homographies = np.stack(inv_list)
            homographies[0, :, :] = np.identity(3)
            homographies = torch.tensor(homographies, dtype=torch.float32)
            inv_homographies = torch.stack([
                torch.inverse(homographies[i]) for i in range(homoAdapt_iter)
            ])

            # warp original image for each homography
            warped_img = self.inv_warp_image_batch(
                image_tensor.repeat(homoAdapt_iter, 1, 1, 1),
                inv_homographies,
                mode='bilinear'
            ).unsqueeze(0).squeeze()

            valid_mask = compute_valid_mask(
                torch.tensor([H, W]),
                inv_homography=inv_homographies,
                erosion_radius=self.config['augmentation']['homographic'][
                    'valid_border_margin']
            )
            output.update({
                'image': warped_img,
                'image_2D': image_tensor,
                'valid_mask': valid_mask,
            })
            output.update({
                'homographies': homographies,
                'inv_homographies': inv_homographies,
            })

        # optionally generate a warped pair using small orientation perturbations
        if self.config.get('warped_pair', {}).get('enable', False):
            H, W = image_tensor.shape[-2:]
            R_perturb = self._sample_rotation(self.config['warped_pair'].get('params', {}))
            homography = sample['K'] @ R_perturb @ sample['K_inv']
            try:
                homo_inv = np.linalg.inv(homography)
            except np.linalg.LinAlgError:
                logging.warning('Warped pair homography singular; using pseudo-inverse')
                homo_inv = np.linalg.pinv(homography)
            warped = inv_warp_image_batch(
                image_tensor.unsqueeze(0),
                torch.tensor(homo_inv, dtype=torch.float32),
            ).squeeze(0)
            output['warped_image'] = warped.unsqueeze(0)
            output['warped_img'] = output['warped_image']
            H_mat = torch.tensor(homography, dtype=torch.float32)
            H_inv_mat = torch.tensor(homo_inv, dtype=torch.float32)
            output['homography'] = H_mat
            output['homographies'] = H_mat.unsqueeze(0)
            output['inv_homographies'] = H_inv_mat.unsqueeze(0)
            margin = self.config['warped_pair'].get('valid_border_margin', 0)
            valid_mask = compute_valid_mask(torch.tensor([H, W]), H_inv_mat, erosion_radius=margin)
            output['warped_valid_mask'] = valid_mask

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

    @staticmethod
    def _euler_to_matrix(yaw, pitch, roll):
        """Convert yaw/pitch/roll angles to a rotation matrix."""
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ], dtype=np.float32)
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ], dtype=np.float32)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ], dtype=np.float32)
        return Rz @ Ry @ Rx

    @staticmethod
    def _sample_rotation(params):
        """Sample a random rotation matrix using Euler angle ranges in degrees."""
        yaw_rng = params.get('yaw_range', 0)
        pitch_rng = params.get('pitch_range', 0)
        roll_rng = params.get('roll_range', 0)
        yaw = np.deg2rad(np.random.uniform(-yaw_rng, yaw_rng))
        pitch = np.deg2rad(np.random.uniform(-pitch_rng, pitch_rng))
        roll = np.deg2rad(np.random.uniform(-roll_rng, roll_rng))
        return Cityscapes._euler_to_matrix(yaw, pitch, roll)

    def _load_camera_matrices(self, cam_path):
        """Load intrinsics and extrinsics from a camera file."""
        if cam_path is None or not Path(cam_path).exists():
            # fallback to identity matrices when camera file is missing
            K = np.eye(3, dtype=np.float32)
            R = np.eye(3, dtype=np.float32)
            t = np.zeros(3, dtype=np.float32)
            return K, R, t

        # read json or simple txt format
        try:
            with open(cam_path, 'r') as f:
                data = json.load(f)
        except Exception:
            data = {}
            with open(cam_path, 'r') as f:
                for line in f:
                    if ':' in line:
                        k, v = line.split(':', 1)
                        data[k.strip()] = float(v.strip())

        fx = float(data.get('fx', 0))
        fy = float(data.get('fy', 0))
        u0 = float(data.get('u0', 0))
        v0 = float(data.get('v0', 0))
        K = np.array([[fx, 0, u0], [0, fy, v0], [0, 0, 1]], dtype=np.float32)

        yaw = float(data.get('yawextrinsic', 0))
        pitch = float(data.get('pitchextrinsic', 0))
        roll = float(data.get('rollextrinsic', 0))
        x = float(data.get('xextrinsic', 0))
        y = float(data.get('yextrinsic', 0))
        z = float(data.get('zextrinsic', 0))

        # convert degrees to radians if needed
        if max(abs(yaw), abs(pitch), abs(roll)) > 2 * np.pi:
            yaw = np.deg2rad(yaw)
            pitch = np.deg2rad(pitch)
            roll = np.deg2rad(roll)

        # rotation matrices around each axis
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ], dtype=np.float32)
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ], dtype=np.float32)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ], dtype=np.float32)
        R = Rz @ Ry @ Rx
        t = np.array([x, y, z], dtype=np.float32)
        return K, R, t

