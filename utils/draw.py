"""util functions for visualization

"""

import argparse
import time
import csv
import yaml
import os
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import cv2
import matplotlib.pyplot as plt


def plot_imgs(imgs, titles=None, cmap='brg', ylabel='', normalize=False, ax=None, dpi=100):
    n = len(imgs)
    if not isinstance(cmap, list):
        cmap = [cmap]*n
    if ax is None:
        fig, ax = plt.subplots(1, n, figsize=(6*n, 6), dpi=dpi)
        if n == 1:
            ax = [ax]
    else:
        if not isinstance(ax, list):
            ax = [ax]
        assert len(ax) == len(imgs)
    for i in range(n):
        if imgs[i].shape[-1] == 3:
            imgs[i] = imgs[i][..., ::-1]  # BGR to RGB
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmap[i]),
                     vmin=None if normalize else 0,
                     vmax=None if normalize else 1)
        if titles:
            ax[i].set_title(titles[i])
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    ax[0].set_ylabel(ylabel)
    plt.tight_layout()


# from utils.draw import img_overlap
def img_overlap(img_r, img_g, img_gray):  # img_b repeat
    img = np.concatenate((img_gray, img_gray, img_gray), axis=0)
    img[0, :, :] += img_r[0, :, :]
    img[1, :, :] += img_g[0, :, :]
    img[img > 1] = 1
    img[img < 0] = 0
    return img

def draw_keypoints(img, corners, color=(0, 255, 0), radius=3, s=3):
    '''

    :param img:
        image:
        numpy [H, W]
    :param corners:
        Points
        numpy [N, 2]
    :param color:
    :param radius:
    :param s:
    :return:
        overlaying image
        numpy [H, W]
    '''
    img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
    for c in np.stack(corners).T:
        # cv2.circle(img, tuple(s * np.flip(c, 0)), radius, color, thickness=-1)
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, color, thickness=-1)
    return img

# def draw_keypoints(img, corners, color=(0, 255, 0), radius=3, s=3):
#     '''

#     :param img:
#         np (H, W)
#     :param corners:
#         np (3, N)
#     :param color:
#     :param radius:
#     :param s:
#     :return:
#     '''
#     img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
#     for c in np.stack(corners).T:
#         # cv2.circle(img, tuple(s * np.flip(c, 0)), radius, color, thickness=-1)
#         cv2.circle(img, tuple((s*c[:2]).astype(int)), radius, color, thickness=-1)
#     return img

def draw_matches(rgb1, rgb2, match_pairs, lw = 0.5, color='g', if_fig=True,
                filename='matches.png', show=False):
    '''

    :param rgb1:
        image1
        numpy (H, W)
    :param rgb2:
        image2
        numpy (H, W)
    :param match_pairs:
        numpy (keypoiny1 x, keypoint1 y, keypoint2 x, keypoint 2 y)
    :return:
        None
    '''
    from matplotlib import pyplot as plt

    h1, w1 = rgb1.shape[:2]
    h2, w2 = rgb2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=rgb1.dtype)
    canvas[:h1, :w1] = rgb1[:,:,np.newaxis]
    canvas[:h2, w1:] = rgb2[:,:,np.newaxis]
    # fig = plt.figure(frameon=False)
    if if_fig:
        fig = plt.figure(figsize=(15,5))
    plt.axis("off")
    plt.imshow(canvas, zorder=1)

    xs = match_pairs[:, [0, 2]]
    xs[:, 1] += w1
    ys = match_pairs[:, [1, 3]]

    alpha = 1
    sf = 5
    # lw = 0.5
    # markersize = 1
    markersize = 2

    plt.plot(
        xs.T, ys.T,
        alpha=alpha,
        linestyle="-",
        linewidth=lw,
        aa=False,
        marker='o',
        markersize=markersize,
        fillstyle='none',
        color=color,
        zorder=2,
        # color=[0.0, 0.8, 0.0],
    );
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    print('#Matches = {}'.format(len(match_pairs)))
    if show:
        plt.show()


def draw_matches_overlay(img1, img2, matches, color=(0, 255, 0), radius=2, thickness=1):
    """Return an image showing ``img1`` and ``img2`` side by side with match lines.

    Parameters
    ----------
    img1 : np.ndarray
        First image in ``H x W`` or ``H x W x 3`` format.
    img2 : np.ndarray
        Second image in ``H x W`` or ``H x W x 3`` format.
    matches : np.ndarray
        Array of shape ``(N, 4)`` containing ``[x1, y1, x2, y2]`` pairs.
    color : tuple, optional
        BGR color of the lines.
    radius : int, optional
        Radius of circles drawn on match points.
    thickness : int, optional
        Line thickness.

    Returns
    -------
    np.ndarray
        Combined overlay image.
    """

    # Ensure both images are 3 channel BGR
    img1 = img1.copy()
    img2 = img2.copy()
    if img1.ndim == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if img2.ndim == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # create canvas wide enough for both images
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=img1.dtype)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1 + w2] = img2

    # draw each match as a line between the two images
    for m in matches:
        x1, y1, x2, y2 = map(int, m[:4])
        pt1 = (x1, y1)
        # second point is shifted by ``w1`` so that it aligns with ``img2``
        pt2 = (x2 + w1, y2)
        # draw connection line and endpoints for visualization
        cv2.line(canvas, pt1, pt2, color, thickness)
        cv2.circle(canvas, pt1, radius, color, -1)
        cv2.circle(canvas, pt2, radius, color, -1)

    return canvas



# from utils.draw import draw_matches_cv
def draw_matches_cv(data):
    keypoints1 = [cv2.KeyPoint(p[1], p[0], 1) for p in data['keypoints1']]
    keypoints2 = [cv2.KeyPoint(p[1], p[0], 1) for p in data['keypoints2']]
    inliers = data['inliers'].astype(bool)
    matches = np.array(data['matches'])[inliers].tolist()
    def to3dim(img):
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        return img
    img1 = to3dim(data['image1'])
    img2 = to3dim(data['image2'])
    img1 = np.concatenate([img1, img1, img1], axis=2)
    img2 = np.concatenate([img2, img2, img2], axis=2)
    return cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches,
                           None, matchColor=(0,255,0), singlePointColor=(0, 0, 255))


def drawBox(points, img, offset=np.array([0,0]), color=(0,255,0)):
#     print("origin", points)
    offset = offset[::-1]
    points = points + offset    
    points = points.astype(int)
    for i in range(len(points)):
        img = img + cv2.line(np.zeros_like(img),tuple(points[-1+i]), tuple(points[i]), color,5)
    return img

