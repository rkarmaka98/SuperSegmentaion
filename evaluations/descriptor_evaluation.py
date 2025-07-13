"""Script for descriptor evaluation

Updated by You-Yi from https://github.com/eric-yyjau/image_denoising_matching
Date: 2020/08/05

"""

import numpy as np
import cv2
from os import path as osp
from glob import glob

from settings import EXPER_PATH


def get_paths(exper_name):
    """
    Return a list of paths to the outputs of the experiment.
    """
    return glob(osp.join(EXPER_PATH, 'outputs/{}/*.npz'.format(exper_name)))


def keep_shared_points(keypoint_map, H, keep_k_points=1000):
    """
    Compute a list of keypoints from the map, filter the list of points by keeping
    only the points that once mapped by H are still inside the shape of the map
    and keep at most 'keep_k_points' keypoints in the image.
    """
    def select_k_best(points, k):
        """ Select the k most probable points (and strip their proba).
        points has shape (num_points, 3) where the last coordinate is the proba. """
        sorted_prob = points[points[:, 2].argsort(), :2]
        start = min(k, points.shape[0])
        return sorted_prob[-start:, :]

    def warp_keypoints(keypoints, H):
        num_points = keypoints.shape[0]
        homogeneous_points = np.concatenate([keypoints, np.ones((num_points, 1))],
                                            axis=1)
        warped_points = np.dot(homogeneous_points, np.transpose(H))
        return warped_points[:, :2] / warped_points[:, 2:]

    def keep_true_keypoints(points, H, shape):
        """ Keep only the points whose warped coordinates by H
        are still inside shape. """
        warped_points = warp_keypoints(points[:, [1, 0]], H)
        warped_points[:, [0, 1]] = warped_points[:, [1, 0]]
        mask = (warped_points[:, 0] >= 0) & (warped_points[:, 0] < shape[0]) &\
               (warped_points[:, 1] >= 0) & (warped_points[:, 1] < shape[1])
        return points[mask, :]

    keypoints = np.where(keypoint_map > 0)
    prob = keypoint_map[keypoints[0], keypoints[1]]
    keypoints = np.stack([keypoints[0], keypoints[1], prob], axis=-1)
    keypoints = keep_true_keypoints(keypoints, H, keypoint_map.shape)
    keypoints = select_k_best(keypoints, keep_k_points)

    return keypoints.astype(int)

def compute_homography(data, keep_k_points=1000, correctness_thresh=3,
                       orb=False, shape=(240, 320)):
    """Estimate a homography from descriptor matches.

    Parameters
    ----------
    data : dict
        Dictionary produced by the evaluation pipeline containing ``prob``,
        ``warped_prob``, ``desc``, ``warped_desc`` and the ground truth
        ``homography``.
    keep_k_points : int, optional
        Maximum number of keypoints used for the estimation.
    correctness_thresh : int, optional
        Threshold in pixels for considering the estimated homography correct.
    orb : bool, optional
        If ``True`` the descriptors are interpreted as ORB descriptors and
        matched using the Hamming distance. Otherwise L2 distance is used.
    shape : tuple, optional
        Shape ``(H, W)`` of the original image. Used for corner error
        computation.

    Returns
    -------
    dict
        Dictionary with the computed homography, boolean correctness flag,
        inlier mask and match information.

    The function performs nearest neighbour matching between descriptors,
    estimates the homography with RANSAC and finally evaluates how close the
    predicted homography is to the ground truth one.
    """
    # shape = data['prob'].shape
    print("shape: ", shape)
    real_H = data['homography']

    # Keeps only the points shared between the two views
    # keypoints = keep_shared_points(data['prob'],
    #                                real_H, keep_k_points)
    # warped_keypoints = keep_shared_points(data['warped_prob'],
    #                                       np.linalg.inv(real_H), keep_k_points)
    # keypoints = data['prob'][:,:2]
    keypoints = data['prob'][:,[1, 0]]
    # warped_keypoints = data['warped_prob'][:,:2]
    warped_keypoints = data['warped_prob'][:,[1, 0]]
    # desc = data['desc'][keypoints[:, 0], keypoints[:, 1]]
    # warped_desc = data['warped_desc'][warped_keypoints[:, 0],
    #                                   warped_keypoints[:, 1]]
    desc = data['desc']
    warped_desc = data['warped_desc']

    # Match the keypoints with the warped_keypoints with nearest neighbor search
    # def get_matches():
    if orb:
        desc = desc.astype(np.uint8)
        warped_desc = warped_desc.astype(np.uint8)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    print("desc: ", desc.shape)
    print("w desc: ", warped_desc.shape)
    # Perform brute force matching between descriptors
    cv2_matches = bf.match(desc, warped_desc)
    matches_idx = np.array([m.queryIdx for m in cv2_matches])
    m_keypoints = keypoints[matches_idx, :]
    matches_idx = np.array([m.trainIdx for m in cv2_matches])
    # Distance between matched descriptors (lower is better)
    m_dist = np.array([m.distance for m in cv2_matches])
    m_warped_keypoints = warped_keypoints[matches_idx, :]
    # Store matches as (y, x) coordinates to be consistent with other code
    matches = np.hstack((m_keypoints[:, [1, 0]], m_warped_keypoints[:, [1, 0]]))
    print(f"matches: {matches.shape}")
    # get_matches()
    # from export_classical import get_sift_match
    # data = get_sift_match(sift_kps_ii=keypoints, sift_des_ii=desc, 
            # sift_kps_jj=warped_keypoints, sift_des_jj=warped_desc, if_BF_matcher=True) 
    # matches_pts = data['match_quality_good']
    # cv_matches = data['cv_matches']
    # print(f"matches: {matches_pts.shape}")
    

    # Estimate the homography between the matches using RANSAC
    # Estimate homography only when there are enough matches. OpenCV requires
    # at least four correspondences, otherwise it raises an exception. The
    # original code did not check for this which caused a crash during
    # evaluation when few matches were found.
    # OpenCV requires at least 4 correspondences to estimate a homography
    if m_keypoints.shape[0] >= 4 and m_warped_keypoints.shape[0] >= 4:
        H, inliers = cv2.findHomography(m_keypoints[:, [1, 0]],
                                        m_warped_keypoints[:, [1, 0]],
                                        cv2.RANSAC)
    else:
        H, inliers = None, None

    # H, inliers = cv2.findHomography(matches_pts[:, [1, 0]],
    #                                 matches_pts[:, [3, 2]],
    #                                 cv2.RANSAC)
                                    
    # inliers might be None when not enough matches were provided for
    # homography estimation. To keep the downstream code simple we
    # return an empty array in that case.
    if inliers is not None:
        inliers = inliers.flatten()
    else:
        inliers = np.array([])
    # print(f"cv_matches: {np.array(cv_matches).shape}, inliers: {inliers.shape}")

    # Compute correctness
    if H is None:
        correctness = 0
        H = np.identity(3)
        mean_dist = np.inf
        print("no valid estimation")
    else:
        corners = np.array([[0, 0, 1],
                            [0, shape[0] - 1, 1],
                            [shape[1] - 1, 0, 1],
                            [shape[1] - 1, shape[0] - 1, 1]])
        print("corner: ", corners)
        # corners = np.array([[0, 0, 1],
        #             [0, shape[1] - 1, 1],
        #             [shape[0] - 1, 0, 1],
        #             [shape[0] - 1, shape[1] - 1, 1]])
        real_warped_corners = np.dot(corners, np.transpose(real_H))
        real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
        print("real_warped_corners: ", real_warped_corners)
        
        warped_corners = np.dot(corners, np.transpose(H))
        warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
        print("warped_corners: ", warped_corners)
        
        mean_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners,
                                           axis=1))
        # The estimation is considered correct if the average corner error is
        # below the threshold
        correctness = mean_dist <= correctness_thresh

    return {'correctness': correctness,
            'keypoints1': keypoints,
            'keypoints2': warped_keypoints,
            'matches': matches,  # cv2.match
            'cv2_matches': cv2_matches,
            # Normalize descriptor distances to the range [0, 1]
            'mscores': m_dist / (m_dist.max()),
            'inliers': inliers,
            'homography': H,
            'mean_dist': mean_dist
            }




def homography_estimation(exper_name, keep_k_points=1000,
                          correctness_thresh=3, orb=False):
    """Compute the mean homography correctness for a set of predictions.

    Parameters
    ----------
    exper_name : str
        Name of the experiment folder under ``EXPER_PATH/outputs``.
    keep_k_points : int, optional
        Maximum number of keypoints kept from each image when estimating the
        homography.
    correctness_thresh : int, optional
        Pixel threshold used to decide if a homography estimate is correct.
    orb : bool, optional
        Passed through to :func:`compute_homography` to select the descriptor
        matching metric.

    Returns
    -------
    float
        Average correctness over all image pairs of the experiment.

    The function iterates over the saved predictions, estimates the homography
    for each pair using :func:`compute_homography` and averages the correctness
    indicator.
    """
    paths = get_paths(exper_name)
    correctness = []
    for path in paths:
        data = np.load(path)
        estimates = compute_homography(data, keep_k_points, correctness_thresh, orb)
        correctness.append(estimates['correctness'])
    return np.mean(correctness)


def get_homography_matches(exper_name, keep_k_points=1000,
                           correctness_thresh=3, num_images=1, orb=False):
    """
    Estimates the homography between two images given the predictions.
    The experiment must contain in its output the prediction on 2 images, an original
    image and a warped version of it, plus the homography linking the 2 images.
    Outputs the keypoints shared between the two views,
    a mask of inliers points in the first image, and a list of matches meaning that
    keypoints1[i] is matched with keypoints2[matches[i]]
    """
    paths = get_paths(exper_name)
    outputs = []
    for path in paths[:num_images]:
        data = np.load(path)
        output = compute_homography(data, keep_k_points, correctness_thresh, orb)
        output['image1'] = data['image']
        output['image2'] = data['warped_image']
        outputs.append(output)
    return outputs
