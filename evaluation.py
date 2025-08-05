"""Script for evaluation
This is the evaluation script for image denoising project.

Author: You-Yi Jau, Yiqian Wang
Date: 2020/03/30
"""

import matplotlib
matplotlib.use('Agg') # solve error of tk

import numpy as np
from evaluations.descriptor_evaluation import compute_homography
from evaluations.detector_evaluation import compute_repeatability, warp_keypoints
from torch.utils.tensorboard import SummaryWriter
from utils.utils import getWriterPath
import cv2
import matplotlib.pyplot as plt
import json

import logging
import os
from tqdm import tqdm
from utils.draw import plot_imgs
from utils.custom_logging import *

def draw_matches_cv(data, matches, plot_points=True):
    if plot_points:
        keypoints1 = [cv2.KeyPoint(p[1], p[0], 0.1) for p in data['keypoints1']]
        keypoints2 = [cv2.KeyPoint(p[1], p[0], 0.1) for p in data['keypoints2']]
    else:
        matches_pts = data['matches']
        keypoints1 = [cv2.KeyPoint(p[0], p[1], 0.1) for p in matches_pts]
        keypoints2 = [cv2.KeyPoint(p[2], p[3], 0.1) for p in matches_pts]
        print(f"matches_pts: {matches_pts}")
        # keypoints1, keypoints2 = [], []

    inliers = data['inliers'].astype(bool)
    # matches = np.array(data['matches'])[inliers].tolist()
    # matches = matches[inliers].tolist()
    def ensure_color(img):
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            # Convert grayscale to BGR to avoid cv2.drawMatches errors
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img
    img1 = ensure_color(data['image1'])
    img2 = ensure_color(data['image2'])

    def to_uint8(img32):
        if img32.dtype == np.uint8:
            # Avoid scaling already uint8 images to prevent value overflow
            return img32
        return cv2.convertScaleAbs(img32, alpha=255.0)

    return cv2.drawMatches(to_uint8(img1), keypoints1, to_uint8(img2), keypoints2, matches,
                           None, matchColor=(0,200,0), singlePointColor=(200, 200, 200))

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def find_files_with_ext(directory, extension='.npz', if_int=True):
    # print(os.listdir(directory))
    list_of_files = []
    import os
    if extension == ".npz":
        for l in os.listdir(directory):
            if l.endswith(extension):
                list_of_files.append(l)
                # print(l)
    if if_int:
        list_of_files = [e for e in list_of_files if isfloat(e[:-4])]
    return list_of_files


def to3dim(img):
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    return img


def compute_miou(pred_mask, gt_mask, num_classes=None):
    """Compute mean Intersection over Union for segmentation masks.

    Both ``pred_mask`` and ``gt_mask`` are expected to have shape ``(H, W)``
    with integer class ids. ``pred_mask`` will be resized to match ``gt_mask``
    using nearest-neighbor interpolation if their dimensions differ.

    Parameters
    ----------
    pred_mask : np.ndarray
        Predicted segmentation mask.
    gt_mask : np.ndarray
        Ground truth segmentation mask.
    num_classes : int, optional
        Number of classes. If ``None`` the union of unique values from both
        masks is used.

    Returns
    -------
    float
        The mean IoU score.
    """
    if pred_mask.shape != gt_mask.shape:
        # resize predicted mask to ground truth size when dimensions differ
        pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                               interpolation=cv2.INTER_NEAREST)

    if num_classes is None:
        classes = np.unique(np.concatenate([pred_mask.ravel(), gt_mask.ravel()]))
    else:
        classes = range(num_classes)

    ious = []
    for cls in classes:
        pred_c = pred_mask == cls
        gt_c = gt_mask == cls
        union = np.logical_or(pred_c, gt_c).sum()
        if union == 0:
            continue
        intersection = np.logical_and(pred_c, gt_c).sum()
        ious.append(intersection / union)

    return float(np.mean(ious)) if ious else 0.0


def compute_pixel_accuracy(pred_mask, gt_mask, num_classes=None):
    """Return overall pixel accuracy and mean class accuracy."""
    if pred_mask.shape != gt_mask.shape:
        pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                               interpolation=cv2.INTER_NEAREST)

    overall = np.mean(pred_mask == gt_mask)

    if num_classes is None:
        classes = np.unique(gt_mask)
    else:
        classes = range(num_classes)

    accs = []
    for cls in classes:
        cls_mask = gt_mask == cls
        if cls_mask.sum() == 0:
            continue
        accs.append(np.mean(pred_mask[cls_mask] == cls))

    mean_acc = float(np.mean(accs)) if accs else 0.0
    return float(overall), mean_acc


def colorize_mask(mask, num_classes=None, class_colors=None):
    """Convert a segmentation mask to a color image for visualization.

    * Generates unique colors for any number of classes.
    * Returns BGR output so it can be passed directly to ``plot_imgs`` which
      expects OpenCV style images.
    """
    if num_classes is None:
        num_classes = int(mask.max()) + 1 if mask.size > 0 else 1

    if class_colors is not None:
        colors = np.asarray(class_colors)
        # ensure user-supplied colors are RGB triples
        assert colors.shape[1] == 3
        # convert to 0-255 range if they were normalized
        if colors.max() <= 1:
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)
        colors = colors[:, [2, 1, 0]]  # convert RGB -> BGR
    else:
        if num_classes == 4:
            # sensible default palette for the 4 Cityscapes categories
            colors = np.array(
                [
                    [255,   0, 170],   # static structure → violet
                    [255, 191,   0],   # flat surfaces   → sky blue (BGR)
                    [0,   69, 255],    # dynamic objects → strong orange-red
                    [128, 128, 128],   # unstable        → gray
                ],
                dtype=float,
            ) / 255.0
        elif num_classes <= 20:
            # use matplotlib's tab20 for small numbers of classes
            cmap = plt.get_cmap("tab20")
            colors = cmap(np.arange(num_classes))[:, :3]
        else:
            # Generate distinct colors in HSV space to avoid repetition
            hsv = np.stack([
                np.linspace(0, 1, num_classes, endpoint=False),
                np.ones(num_classes),
                np.ones(num_classes)
            ], axis=1)
            colors = matplotlib.colors.hsv_to_rgb(hsv)

        # convert from float RGB [0,1] to uint8 BGR
        colors = (colors * 255).astype(np.uint8)
        colors = colors[:, [2, 1, 0]]

    return colors[mask.astype(int)]


def overlay_mask(image, mask, alpha=0.5, num_classes=None, class_names=None, class_colors=None):
    """Overlay a colorized mask on top of an image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale or color image in range [0, 1] or uint8.
    mask : np.ndarray
        Segmentation mask to overlay.
    alpha : float, optional
        Opacity of the mask, by default 0.5.
    num_classes : int, optional
        Number of classes for ``mask``.

    Returns
    -------
    np.ndarray
        BGR image showing ``image`` with the colorized ``mask`` overlaid.
    """
    color_mask = colorize_mask(mask, num_classes, class_colors)

    # Ensure the base image has three channels without changing its dimensions
    if image.ndim == 2:
        # (H, W) -> (H, W, 3)
        img_color = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    elif image.shape[-1] == 1:
        # (H, W, 1) -> (H, W, 3)
        img_color = np.repeat(image, 3, axis=2)
    else:
        img_color = image

    # Resize the mask if dimensions differ
    if color_mask.shape[:2] != img_color.shape[:2]:
        # Nearest neighbour keeps class labels intact
        color_mask = cv2.resize(
            color_mask,
            (img_color.shape[1], img_color.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )


    # Convert floating point images to 8-bit and blend with mask
    if img_color.dtype != np.uint8:
        img_color = cv2.convertScaleAbs(img_color, alpha=255.0)
    overlay = cv2.addWeighted(img_color, 1 - alpha, color_mask, alpha, 0)

    # Draw legend for the classes present in the mask
    unique_classes = np.unique(mask)
    if num_classes is None:
        num_classes = int(unique_classes.max()) + 1
    if class_names is None:
        class_names = [f"class {i}" for i in range(num_classes)]

    patch = 20  # size of color squares
    margin = 5
    x = margin
    y = margin
    row_height = patch + 15
    for cls in unique_classes:
        cls = int(cls)
        color = colorize_mask(np.array([[cls]]), num_classes, class_colors)[0, 0].tolist()
        cv2.rectangle(overlay, (x, y), (x + patch, y + patch), color, -1)
        cv2.putText(
            overlay,
            class_names[cls],
            (x + patch + 5, y + patch - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
        text_size, _ = cv2.getTextSize(
            class_names[cls], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        x += patch + 5 + text_size[0] + margin
        if x + patch > overlay.shape[1]:
            x = margin
            y += row_height
            if y + row_height > overlay.shape[0]:
                break

    return overlay

def smooth_mask(mask, kernel_size=3):
    """Apply simple morphological post-processing to clean up a mask.

    ``mask`` should be an array of shape ``(H, W)`` containing class ids.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def evaluate(args, **options):
    # path = '/home/yoyee/Documents/SuperPoint/superpoint/logs/outputs/superpoint_coco/'
    path = args.path
    files = find_files_with_ext(path)
    writer = SummaryWriter(getWriterPath(task='eval', exper_name=os.path.basename(path), date=True))
    correctness = []
    est_H_mean_dist = []
    repeatability = []
    mscore = []
    mAP = []
    localization_err = []
    segmentation_iou = []
    # default value to avoid UnboundLocalError when homography evaluation is disabled
    homography_thresh = []
    rep_thd = 3
    save_file = path + "/result.txt"
    inliers_method = 'cv'
    compute_map = True
    verbose = True
    top_K = 1000
    print("top_K: ", top_K)

    class_colors = None
    class_names = None
    if args.category_file:
        try:
            with open(args.category_file, 'r') as f:
                categories = json.load(f)
            max_id = max(c['id'] for c in categories) + 1
            class_colors = np.zeros((max_id, 3), dtype=np.uint8)
            class_names = [''] * max_id
            for c in categories:
                cid = c['id']
                if cid >= max_id:
                    continue
                class_colors[cid] = np.array(c['color'][::-1], dtype=np.uint8)
                class_names[cid] = c.get('name', str(cid))
        except Exception as e:
            logging.warning(f"Failed to load categories from {args.category_file}: {e}")

    reproduce = True
    if reproduce:
        logging.info("reproduce = True")
        np.random.seed(0)
        print(f"test random # : np({np.random.rand(1)})")


    # create output dir
    if args.outputImg:
        path_warp = path+'/warping'
        os.makedirs(path_warp, exist_ok=True)
        path_match = path + '/matching'
        os.makedirs(path_match, exist_ok=True)
        path_rep = path + '/repeatibility' + str(rep_thd)
        os.makedirs(path_rep, exist_ok=True)
        if args.evaluate_segmentation:
            path_seg = path + '/segmentation'
            os.makedirs(path_seg, exist_ok=True)
        if args.stable_matching:
            # save visualizations restricted to static/flat classes
            path_stable = path + '/stable_matching'
            os.makedirs(path_stable, exist_ok=True)

    # for i in range(2):
    #     f = files[i]
    print(f"file: {files[0]}")
    files.sort(key=lambda x: int(x[:-4]))
    from numpy.linalg import norm
    from utils.draw import draw_keypoints
    from utils.utils import saveImg
    step = 0  # step counter for TensorBoard

    for f in tqdm(files):
        f_num = f[:-4]
        data = np.load(path + '/' + f)
        print("load successfully. ", f)

        pred_mask = None  # store predicted mask for reuse outside this block

        if args.evaluate_segmentation:
            # Look for typical predicted and ground-truth mask keys
            pred_key = next((k for k in ['pred_mask', 'segmentation_mask', 'mask_pred']
                             if k in data.files), None)
            gt_key = next((k for k in ['gt_mask', 'segmentation_gt', 'mask_gt']
                           if k in data.files), None)

            if pred_key is None:
                logging.warning(f"No predicted mask found in {f}")
            else:
                pred_mask = smooth_mask(data[pred_key])  # keep in scope for stable matching
                # compute segmentation metrics when gt mask is available
                if gt_key:
                    miou = compute_miou(pred_mask, data[gt_key])
                    segmentation_iou.append(miou)
                    pixel_acc, class_acc = compute_pixel_accuracy(pred_mask, data[gt_key])
                    writer.add_scalar("miou", miou, step)
                    writer.add_scalar("pixel_acc", pixel_acc, step)
                    writer.add_scalar("class_acc", class_acc, step)

                if args.outputImg:
                    # visualize predicted mask over the original image
                    base_img = data['image']

                    # Upsample predicted mask to match original image resolution before visualization
                    pred_mask_vis = pred_mask
                    if pred_mask_vis.shape != base_img.shape[:2]:
                        pred_mask_vis = cv2.resize(
                            pred_mask_vis,
                            (base_img.shape[1], base_img.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    imgs = [
                        overlay_mask(
                            base_img,
                            pred_mask_vis,
                            alpha=0.3,  # blend mask at 30%
                            class_names=class_names,
                            class_colors=class_colors,
                        )
                    ]
                    titles = ['pred overlay']
                    if gt_key:
                        # Upsample ground truth mask to original resolution for fair comparison
                        gt_mask_vis = data[gt_key]
                        if gt_mask_vis.shape != base_img.shape[:2]:
                            gt_mask_vis = cv2.resize(
                                gt_mask_vis,
                                (base_img.shape[1], base_img.shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        imgs.append(
                            overlay_mask(
                                base_img,
                                gt_mask_vis,
                                alpha=0.3,  # blend mask at 30%
                                class_names=class_names,
                                class_colors=class_colors,
                            )
                        )
                        titles.append('gt overlay')

                    plot_imgs(imgs, titles=titles, dpi=200)
                    plt.tight_layout()
                    plt.savefig(os.path.join(path_seg, f_num + '.png'), dpi=300,
                                bbox_inches='tight')
                    plt.close('all')

        # unwarp
        # prob = data['prob']
        # warped_prob = data['prob']
        # desc = data['desc']
        # warped_desc = data['warped_desc']
        # homography = data['homography']
        real_H = data['homography']
        image = data['image']
        # gracefully handle missing or empty warped images
        if 'warped_image' in data.files:
            warped_image = data['warped_image']
        else:
            warped_image = None

        if warped_image is None or not np.any(warped_image):
            try:
                warped_image = cv2.warpPerspective(
                    image,
                    real_H,
                    (image.shape[1], image.shape[0]),
                )
            except Exception as e:
                logging.warning(f"Failed to compute warped image for {f}: {e}")
                warped_image = np.zeros_like(image)
        keypoints = data['prob'][:, [1, 0]]
        print("keypoints: ", keypoints[:3,:])
        warped_keypoints = data['warped_prob'][:, [1, 0]]
        print("warped_keypoints: ", warped_keypoints[:3,:])
        # print("Unwrap successfully.")

        if args.repeatibility:
            rep, local_err = compute_repeatability(data, keep_k_points=top_K, distance_thresh=rep_thd, verbose=False)
            repeatability.append(rep)
            print("repeatability: %.2f"%(rep))
            if local_err > 0:
                localization_err.append(local_err)
                print('local_err: ', local_err)
            writer.add_scalar("repeatability", rep, step)
            if args.outputImg:
                # img = to3dim(image)
                img = image
                pts = data['prob']
                img1 = draw_keypoints(img*255, pts.transpose())

                # img = to3dim(warped_image)
                img = warped_image
                pts = data['warped_prob']
                img2 = draw_keypoints(img*255, pts.transpose())

                plot_imgs([img1.astype(np.uint8), img2.astype(np.uint8)], titles=['img1', 'img2'], dpi=200)
                plt.title("rep: " + str(repeatability[-1]))
                plt.tight_layout()
                
                plt.savefig(path_rep + '/' + f_num + '.png', dpi=300, bbox_inches='tight')
                pass


        if args.homography:
            # estimate result
            ##### check
            homography_thresh = [1,3,5,10,20,50]
            #####
            result = compute_homography(data, correctness_thresh=homography_thresh)
            correctness.append(result['correctness'])
            # est_H_mean_dist.append(result['mean_dist'])
            # Re-label matches within a small pixel error as inliers to cope with
            # the 1 px noise present in Cityscapes ground truth.
            if result['matches'].size > 0 and args.inlier_pixel_threshold >= 0:
                # Warp keypoints using ground-truth homography to measure error
                gt_warped = warp_keypoints(result['matches'][:, :2], real_H)
                err = np.linalg.norm(gt_warped - result['matches'][:, 2:4], axis=1)
                close = err <= args.inlier_pixel_threshold
                if result['inliers'].size == err.size:
                    result['inliers'] = np.logical_or(result['inliers'].astype(bool), close)
                else:
                    result['inliers'] = close
            # compute matching score
            def warpLabels(pnts, homography, H, W):
                import torch
                """
                input:
                    pnts: numpy
                    homography: numpy
                output:
                    warped_pnts: numpy
                """
                from utils.utils import warp_points
                from utils.utils import filter_points
                pnts = torch.tensor(pnts).float()
                homography = torch.tensor(homography, dtype=torch.float32)
                warped_pnts = warp_points(torch.stack((pnts[:, 0], pnts[:, 1]), dim=1),
                                          homography)  # check the (x, y)
                warped_pnts = filter_points(warped_pnts, torch.tensor([W, H])).float()
                return warped_pnts.numpy()

            from numpy.linalg import inv
            H, W = image.shape
            unwarped_pnts = warpLabels(warped_keypoints, inv(real_H), H, W)
            # score = (result['inliers'].sum() * 2) / (keypoints.shape[0] + unwarped_pnts.shape[0])
            # deduplicate here since repeated points inflate the denominator
            # when computing the matching score
            unique_kpts = np.unique(keypoints, axis=0)
            unique_unwarp = np.unique(unwarped_pnts, axis=0)
            denom = unique_kpts.shape[0] + unique_unwarp.shape[0]
            score = (result['inliers'].sum() * 2) / denom if denom > 0 else 0.0
            score = min(score, 1.0)
            print("m. score: ", score)
            mscore.append(score)
            if result['inliers'].size > 0:
                match_precision = result['inliers'].sum() / result['inliers'].shape[0]
            else:
                match_precision = 0.0
            match_recall = result['inliers'].sum() / keypoints.shape[0] if keypoints.shape[0] > 0 else 0.0
            writer.add_scalar("match-precision", match_precision, step)
            writer.add_scalar("match-recall", match_recall, step)
            # compute map
            if compute_map:
                def getMatches(data):
                    from models.model_wrap import PointTracker

                    desc = data['desc']
                    warped_desc = data['warped_desc']

                    nn_thresh = 1.2
                    print("nn threshold: ", nn_thresh)
                    tracker = PointTracker(max_length=2, nn_thresh=nn_thresh)
                    # matches = tracker.nn_match_two_way(desc, warped_desc, nn_)
                    tracker.update(keypoints.T, desc.T)
                    tracker.update(warped_keypoints.T, warped_desc.T)
                    matches = tracker.get_matches().T
                    mscores = tracker.get_mscores().T

                    # mAP
                    # matches = data['matches']
                    print("matches: ", matches.shape)
                    print("mscores: ", mscores.shape)
                    print("mscore max: ", mscores.max(axis=0))
                    print("mscore min: ", mscores.min(axis=0))

                    return matches, mscores

                def getInliers(matches, H, epi=3, verbose=False):
                    """
                    input:
                        matches: numpy (n, 4(x1, y1, x2, y2))
                        H (ground truth homography): numpy (3, 3)
                    """
                    from evaluations.detector_evaluation import warp_keypoints
                    # warp points 
                    warped_points = warp_keypoints(matches[:, :2], H) # make sure the input fits the (x,y)

                    # compute point distance
                    norm = np.linalg.norm(warped_points - matches[:, 2:4],
                                            ord=None, axis=1)
                    inliers = norm < epi
                    if verbose:
                        print("Total matches: ", inliers.shape[0], ", inliers: ", inliers.sum(),
                                          ", percentage: ", inliers.sum() / inliers.shape[0])

                    return inliers

                def getInliers_cv(matches, H=None, epi=3, verbose=False):
                    import cv2
                    # count inliers: use opencv homography estimation
                    # OpenCV requires at least 4 correspondences for homography
                    # estimation. If not enough matches are available simply
                    # return an empty inlier mask instead of crashing.
                    if matches.shape[0] < 4:
                        if verbose:
                            print("no valid estimation")
                        return np.zeros(matches.shape[0], dtype=bool)

                    # Estimate the homography between the matches using RANSAC
                    # and the provided pixel threshold.
                    H, inliers = cv2.findHomography(
                        matches[:, [0, 1]],
                        matches[:, [2, 3]],
                        cv2.RANSAC,
                        epi,
                    )
                    inliers = inliers.flatten()
                    print("Total matches: ", inliers.shape[0],
                          ", inliers: ", inliers.sum(),
                          ", percentage: ", inliers.sum() / inliers.shape[0])
                    return inliers
            
            
                def computeAP(m_test, m_score):
                    from sklearn.metrics import average_precision_score

                    average_precision = average_precision_score(m_test, m_score)
                    print('Average precision-recall score: {0:0.2f}'.format(
                        average_precision))
                    return average_precision

                def flipArr(arr):
                    return arr.max() - arr
                
                if args.sift:
                    assert result is not None
                    matches, mscores = result['matches'], result['mscores']
                else:
                    matches, mscores = getMatches(data)
                
                real_H = data['homography']
                if inliers_method == 'gt':
                    # use ground truth homography
                    print("use ground truth homography for inliers")
                    inliers = getInliers(
                        matches,
                        real_H,
                        epi=args.inlier_pixel_threshold,
                        verbose=verbose,
                    )
                else:
                    # use opencv estimation as inliers
                    print("use opencv estimation for inliers")
                    inliers = getInliers_cv(
                        matches,
                        real_H,
                        epi=args.inlier_pixel_threshold,
                        verbose=verbose,
                    )
                # Ensure small reprojection errors are counted as inliers.
                err = np.linalg.norm(
                    warp_keypoints(matches[:, :2], real_H) - matches[:, 2:4],
                    axis=1,
                )
                inliers = np.logical_or(inliers, err <= args.inlier_pixel_threshold)
                    
                ## distance to confidence
                if args.sift:
                    m_flip = flipArr(mscores[:])  # for sift
                else:
                    m_flip = flipArr(mscores[:,2])
        
                if inliers.shape[0] > 0 and inliers.sum()>0:
#                     m_flip = flipArr(m_flip)
                    # compute ap
                    ap = computeAP(inliers, m_flip)
                else:
                    ap = 0
                
                mAP.append(ap)


            if args.outputImg:
                # draw warping
                output = result
                # img1 = image/255
                # img2 = warped_image/255
                img1 = image
                img2 = warped_image

                img1 = to3dim(img1)
                img2 = to3dim(img2)
                H = output['homography']
                warped_img1 = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
                # from numpy.linalg import inv
                # warped_img1 = cv2.warpPerspective(img1, inv(H), (img2.shape[1], img2.shape[0]))
                img1 = np.concatenate([img1, img1, img1], axis=2)
                warped_img1 = np.stack([warped_img1, warped_img1, warped_img1], axis=2)
                img2 = np.concatenate([img2, img2, img2], axis=2)
                plot_imgs([img1, img2, warped_img1], titles=['img1', 'img2', 'warped_img1'], dpi=200)
                plt.tight_layout()
                plt.savefig(path_warp + '/' + f_num + '.png')

                ## plot filtered image
                # reuse possibly reconstructed warped_image
                img1, img2 = image, warped_image
                warped_img1 = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
                plot_imgs([img1, img2, warped_img1], titles=['img1', 'img2', 'warped_img1'], dpi=200)
                plt.tight_layout()
                # plt.savefig(path_warp + '/' + f_num + '_fil.png')
                plt.savefig(path_warp + '/' + f_num + '.png')

                # plt.show()

                # draw matches
                result['image1'] = image
                result['image2'] = warped_image
                matches = np.array(result['cv2_matches'])
                ratio = 0.2
                ran_idx = np.random.choice(matches.shape[0], int(matches.shape[0]*ratio))

                img = draw_matches_cv(result, matches[ran_idx], plot_points=True)
                # filename = "correspondence_visualization"
                plot_imgs([img], titles=["Two images feature correspondences"], dpi=200)
                plt.tight_layout()
                plt.savefig(path_match + '/' + f_num + 'cv.png', bbox_inches='tight')
                plt.close('all')
                # pltImshow(img)

                if args.stable_matching and pred_mask is not None:
                    # overlay predicted segmentation and visualize matches only for static/flat classes
                    mask1 = pred_mask
                    mask2 = cv2.warpPerspective(mask1, real_H, (mask1.shape[1], mask1.shape[0]),
                                                flags=cv2.INTER_NEAREST)
                    coords = result['matches'].astype(int)
                    # clip coordinates to image bounds to avoid indexing errors
                    h, w = mask1.shape
                    coords[:, 0] = np.clip(coords[:, 0], 0, w - 1)
                    coords[:, 1] = np.clip(coords[:, 1], 0, h - 1)
                    coords[:, 2] = np.clip(coords[:, 2], 0, w - 1)
                    coords[:, 3] = np.clip(coords[:, 3], 0, h - 1)
                    stable = np.isin(mask1[coords[:, 1], coords[:, 0]], [0, 1]) & \
                             np.isin(mask2[coords[:, 3], coords[:, 2]], [0, 1])
                    if np.any(stable):
                        result_stable = dict(result)
                        # boolean mask `stable` corresponds to matches, not all keypoints;
                        # keep keypoint arrays intact and filter match-related data instead
                        result_stable['cv2_matches'] = np.array(result['cv2_matches'])[stable]
                        result_stable['matches'] = result['matches'][stable]
                        if result['inliers'].size == stable.size:
                            result_stable['inliers'] = result['inliers'][stable]
                        result_stable['image1'] = overlay_mask(image, mask1, alpha=0.5,
                                                               class_names=class_names,
                                                               class_colors=None)
                        result_stable['image2'] = overlay_mask(warped_image, mask2, alpha=0.5,
                                                               class_names=class_names,
                                                               class_colors=None)
                        matches_stable = result_stable['cv2_matches']
                        ratio = 0.1
                        ran_idx = np.random.choice(matches_stable.shape[0], int(matches_stable.shape[0]*ratio))
                        img_stable = draw_matches_cv(result_stable, matches_stable[ran_idx], plot_points=True)
                        plot_imgs([img_stable], titles=['Stable class correspondences'], dpi=200)
                        plt.tight_layout()
                        plt.savefig(path_stable + '/' + f_num + 'cv.png', bbox_inches='tight')
                        plt.close('all')

        if args.plotMatching:
            matches = result['matches'] # np [N x 4]
            if matches.shape[0] > 0:
                from utils.draw import draw_matches
                filename = path_match + '/' + f_num + 'm.png'
                ratio = 0.1
                inliers = result['inliers']

                matches_in = matches[inliers == True]
                matches_out = matches[inliers == False]

                def get_random_m(matches, ratio):
                    ran_idx = np.random.choice(matches.shape[0], int(matches.shape[0]*ratio))               
                    return matches[ran_idx], ran_idx
                image = data['image']
                ## outliers
                matches_temp, _ = get_random_m(matches_out, ratio)
                # print(f"matches_in: {matches_in.shape}, matches_temp: {matches_temp.shape}")
                draw_matches(image, warped_image, matches_temp, lw=0.5, color='r',
                            filename=None, show=False, if_fig=True)
                ## inliers
                matches_temp, _ = get_random_m(matches_in, ratio)
                draw_matches(image, warped_image, matches_temp, lw=1.0, 
                        filename=filename, show=False, if_fig=False)

        step += 1






    if args.repeatibility:
        repeatability_ave = np.array(repeatability).mean()
        localization_err_m = np.array(localization_err).mean()
        print("repeatability: ", repeatability_ave)
        print("localization error over ", len(localization_err), " images : ", localization_err_m)
    if args.homography:
        correctness_ave = np.array(correctness).mean(axis=0)
        # est_H_mean_dist = np.array(est_H_mean_dist)
        print("homography estimation threshold", homography_thresh)
        print("correctness_ave", correctness_ave)
        # print(f"mean est H dist: {est_H_mean_dist.mean()}")
        mscore_m = np.array(mscore).mean(axis=0)
        print("matching score", mscore_m)
        if compute_map:
            mAP_m = np.array(mAP).mean()
            print("mean AP", mAP_m)

        print("end")

    if args.evaluate_segmentation and segmentation_iou:
        miou_mean = float(np.mean(segmentation_iou))
        print("segmentation mIoU", miou_mean)

    # save to files
    with open(save_file, "a") as myfile:
        myfile.write("path: " + path + '\n')
        myfile.write("output Images: " + str(args.outputImg) + '\n')
        if args.repeatibility:
            myfile.write("repeatability threshold: " + str(rep_thd) + '\n')
            myfile.write("repeatability: " + str(repeatability_ave) + '\n')
            myfile.write("localization error: " + str(localization_err_m) + '\n')
        if args.homography:
            myfile.write("Homography estimation: " + '\n')
            myfile.write("Homography threshold: " + str(homography_thresh) + '\n')
            myfile.write("Average correctness: " + str(correctness_ave) + '\n')

            # myfile.write("mean est H dist: " + str(est_H_mean_dist.mean()) + '\n')

            if compute_map:
                myfile.write("nn mean AP: " + str(mAP_m) + '\n')
            myfile.write("matching score: " + str(mscore_m) + '\n')

        if args.evaluate_segmentation and segmentation_iou:
            myfile.write("segmentation mIoU: " + str(miou_mean) + '\n')


        if verbose:
            myfile.write("====== details =====" + '\n')
            for i in range(len(files)):

                myfile.write("file: " + files[i])
                if args.repeatibility:
                    myfile.write("; rep: " + str(repeatability[i]))
                if args.homography:
                    myfile.write("; correct: " + str(correctness[i]))
                    # matching
                    myfile.write("; mscore: " + str(mscore[i]))
                    if compute_map:
                        myfile.write(":, mean AP: " + str(mAP[i]))
                myfile.write('\n')
            myfile.write("======== end ========" + '\n')

    dict_of_lists = {
        'repeatability': repeatability,
        'localization_err': localization_err,
        'correctness': np.array(correctness),
        # store homography thresholds only when evaluated
        'homography_thresh': homography_thresh if args.homography else [],
        'mscore': mscore,
        'mAP': np.array(mAP),
        'segmentation_iou': np.array(segmentation_iou),
        # 'est_H_mean_dist': est_H_mean_dist
    }

    filename = f'{save_file[:-4]}.npz'
    logging.info(f"save file: {filename}")
    np.savez(
        filename,
        **dict_of_lists,
    )
    writer.close()


if __name__ == '__main__':
    import argparse


    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--sift', action='store_true', help='use sift matches')
    parser.add_argument('-o', '--outputImg', action='store_true')
    parser.add_argument('-r', '--repeatibility', action='store_true')
    parser.add_argument('-homo', '--homography', action='store_true')
    parser.add_argument('-plm', '--plotMatching', action='store_true')
    parser.add_argument(
        '--stable-matching',
        action='store_true',
        help='overlay segmentation masks and show matches only on static/flat classes',
    )
    parser.add_argument(
        '--inlier-pixel-threshold',
        type=float,
        default=1.0,
        help='Treat matches with reprojection error <= this many pixels as inliers.\n'
             'Helps avoid marking correct matches as wrong on noisy datasets like Cityscapes.',
    )
    parser.add_argument(
        '--evaluate-segmentation',
        action='store_true',
        help='compute segmentation metrics when segmentation masks are present',
    )
    parser.add_argument(
        '--category-file',
        type=str,
        default=None,
        help='path to panoptic category json with class colors',
    )
    args = parser.parse_args()
    evaluate(args)
