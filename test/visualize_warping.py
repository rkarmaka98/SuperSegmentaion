"""Testing file (not sorted yet)

import torch
import numpy as np


from utils.utils import inv_warp_image_batch
from numpy.linalg import inv
import cv2
import matplotlib.pyplot as plt
from utils.draw import plot_imgs

from utils.utils import pltImshow
path = 'logs/magicpoint_synth_homoAdapt_cityscape/predictions/train'
for i in range(10):
    data = np.load(path + str(i) + '.npz')
    # p1 = '/home/yoyee/Documents/deepSfm/datasets/HPatches/v_abstract/1.ppm'
    # p2 = '/home/yoyee/Documents/deepSfm/datasets/HPatches/v_abstract/2.ppm'
    # H = '/home/yoyee/Documents/deepSfm/datasets/HPatches/v_abstract/H_1_2'
    # img = np.load(p1)
    # warped_img = np.load(p2)

    H = data['homography']
    img1 = data['image'][:,:,np.newaxis]
    img2 = data['warped_image'][:,:,np.newaxis]
    # warped_img_H = inv_warp_image_batch(torch.tensor(img), torch.tensor(inv(H)))
    warped_img1 = cv2.warpPerspective(img1, H, (img1.shape[1], img1.shape[0]))


    # img_cat = np.concatenate((img, warped_img, warped_img_H), axis=1)
    # pltImshow(img_cat)

    # from numpy.linalg import inv
    # warped_img1 = cv2.warpPerspective(img1, inv(H), (img2.shape[1], img2.shape[0]))
    img1 = np.concatenate([img1, img1, img1], axis=2)
    warped_img1 = np.stack([warped_img1, warped_img1, warped_img1], axis=2)
    img2 = np.concatenate([img2, img2, img2], axis=2)
    plot_imgs([img1, img2, warped_img1], titles=['img1', 'img2', 'warped_img1'], dpi=200)
    plt.savefig( 'test' + str(i) + '.png')

    """
    
import os,sys

# Add the project’s root directory (one level up) to Python’s import path:
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)

import glob

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from numpy.linalg import inv
from utils.utils import inv_warp_image_batch, pltImshow
from utils.draw import plot_imgs

# point this to your folder containing the .npz predictions
base_dir = r'.'

# find and sort all .npz files
npz_paths = sorted(glob.glob(os.path.join(base_dir, '*.npz')))

for npz_path in tqdm(npz_paths):
    # load the archive
    data = np.load(npz_path)
    H = data['homography']            # (3×3)
    img1 = data['image'][..., np.newaxis]       # (H×W×1)
    img2 = data['warped_image'][..., np.newaxis]  # (H×W×1)

    # warp img1 by H
    # OpenCV wants (width, height)
    h, w = img1.shape[:2]
    warped_img1 = cv2.warpPerspective(img1, H.astype(np.float32), (w, h))

    # convert to 3-channel for plotting
    img1_rgb       = np.repeat(img1,       3, axis=2)
    img2_rgb       = np.repeat(img2,       3, axis=2)
    warped_img1_rgb = np.repeat(warped_img1, 3, axis=2)

    # plot and save
    plot_imgs(
        [img1_rgb, img2_rgb, warped_img1_rgb],
        titles=['img1', 'img2', 'warped_img1'],
        dpi=200
    )

    # build a matching save name, e.g. test_aachen_000000_000019.png
    base_name = os.path.splitext(os.path.basename(npz_path))[0]
    out_path = os.path.join(base_dir, f'test_{base_name}.png')
    plt.savefig(out_path)
    print(f'Saved visualization to {out_path}')