"""Demo script to export matches between each Cityscapes frame and its homography-warped view."""

from pathlib import Path
import yaml
import torch
import numpy as np
import cv2

from Val_model_heatmap import Val_model_heatmap
from evaluation import overlay_mask, draw_matches_cv


def nn_match_two_way(desc1: np.ndarray, desc2: np.ndarray, nn_thresh: float) -> np.ndarray:
    """Two-way nearest-neighbor matching for unit-normalized descriptors."""
    assert desc1.shape[0] == desc2.shape[0]
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
        return np.zeros((3, 0))
    if nn_thresh < 0.0:
        raise ValueError("'nn_thresh' should be non-negative")
    dmat = np.dot(desc1.T, desc2)
    dmat = np.sqrt(2 - 2 * np.clip(dmat, -1, 1))
    idx = np.argmin(dmat, axis=1)
    scores = dmat[np.arange(dmat.shape[0]), idx]
    keep = scores < nn_thresh
    idx2 = np.argmin(dmat, axis=0)
    keep_bi = np.arange(len(idx)) == idx2[idx]
    keep = np.logical_and(keep, keep_bi)
    idx = idx[keep]
    scores = scores[keep]
    m_idx1 = np.arange(desc1.shape[1])[keep]
    m_idx2 = idx
    matches = np.zeros((3, int(keep.sum())))
    matches[0, :] = m_idx1
    matches[1, :] = m_idx2
    matches[2, :] = scores
    return matches


def main():
    # Iterate over every sequence (city) in the training split.
    train_root = Path("datasets/Cityscapes/rightImg8bit/train")
    if not train_root.exists():
        raise RuntimeError(f"Training directory {train_root} does not exist")

    # Load model configuration and weights once.
    with open("configs/superpoint_cityscapes_export.yaml", "r") as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_agent = Val_model_heatmap(config["model"], device=device)
    val_agent.loadModel()

    resize = config["data"]["preprocessing"]["resize"]  # model expects this size.

    # Visualization controls for draw_matches_cv.
    draw_keypoints = True  # draw only match lines if False
    point_color = (0, 255, 0)  # BGR color for keypoints when drawn
    point_radius = 2  # radius for keypoint circles

    def load_frame(p: Path) -> np.ndarray:
        """Read and resize a grayscale frame to the model input size."""
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        return cv2.resize(img, (resize[1], resize[0]), interpolation=cv2.INTER_AREA)

    out_root = Path("sequence_demo_output")
    out_root.mkdir(exist_ok=True)

    # Expected location of homography matrices (one per frame).
    homography_root = Path("datasets/Cityscapes/homographies/train")

    # Loop over all sequence directories such as 'aachen', 'bochum', ...
    for seq_dir in sorted(train_root.iterdir()):
        if not seq_dir.is_dir():
            continue
        image_paths = sorted(seq_dir.glob("*.png"))
        if not image_paths:
            continue

        # Create a mirrored output directory for this sequence.
        seq_out = out_root / seq_dir.name
        seq_out.mkdir(parents=True, exist_ok=True)

        for img_path in image_paths:
            # Pair each frame with its homography-warped counterpart; no temporal loop.
            H_path = homography_root / seq_dir.name / f"{img_path.stem}.npy"
            if not H_path.exists():
                print(f"Missing homography for {img_path}, skipping")
                continue
            H = np.load(str(H_path))

            img0_raw = load_frame(img_path)
            img1_raw = cv2.warpPerspective(img0_raw, H, (img0_raw.shape[1], img0_raw.shape[0]))

            img0 = torch.from_numpy(img0_raw.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
            img1 = torch.from_numpy(img1_raw.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)

            # Compute descriptors independently for the frame and its warp.
            val_agent.run(img0)
            pts0 = val_agent.heatmap_to_pts()[0]
            desc0 = val_agent.desc_to_sparseDesc()[0]
            seg0 = None
            if "segmentation" in val_agent.outs:
                seg0 = val_agent.outs["segmentation"].argmax(dim=1).cpu().numpy()[0]

            val_agent.run(img1)
            pts1 = val_agent.heatmap_to_pts()[0]
            desc1 = val_agent.desc_to_sparseDesc()[0]

            # Perform two-way nearest-neighbor matching without tracking.
            matches = nn_match_two_way(desc0, desc1, val_agent.nn_thresh).T
            kpts1 = pts0[[1, 0], :].T
            kpts2 = pts1[[1, 0], :].T

            if matches.size == 0:
                print(f"No matches for {img_path.stem} in {seq_dir.name}, writing placeholder")
                coords = np.zeros((0, 4), dtype=int)
                inliers = np.zeros(0, dtype=bool)
                cv2_matches = []
                seg1 = (
                    cv2.warpPerspective(
                        seg0, H, (seg0.shape[1], seg0.shape[0]), flags=cv2.INTER_NEAREST
                    )
                    if seg0 is not None
                    else None
                )
            else:
                cv2_matches = [
                    cv2.DMatch(int(m[0]), int(m[1]), float(m[2])) for m in matches
                ]
                inliers = np.ones(matches.shape[0], dtype=bool)
                coords = np.zeros((matches.shape[0], 4), dtype=int)
                coords[:, 0] = pts0[0, matches[:, 0].astype(int)]  # x1
                coords[:, 1] = pts0[1, matches[:, 0].astype(int)]  # y1
                coords[:, 2] = pts1[0, matches[:, 1].astype(int)]  # x2
                coords[:, 3] = pts1[1, matches[:, 1].astype(int)]  # y2

                seg1 = None
                if seg0 is not None:
                    # Warp the segmentation mask with the same homography used on the image.
                    seg1 = cv2.warpPerspective(seg0, H, (seg0.shape[1], seg0.shape[0]), flags=cv2.INTER_NEAREST)
                    h, w = seg0.shape
                    coords[:, 0] = np.clip(coords[:, 0], 0, w - 1)
                    coords[:, 1] = np.clip(coords[:, 1], 0, h - 1)
                    coords[:, 2] = np.clip(coords[:, 2], 0, w - 1)
                    coords[:, 3] = np.clip(coords[:, 3], 0, h - 1)
                    # Keep matches only if both points belong to static/flat classes.
                    stable = np.isin(seg0[coords[:, 1], coords[:, 0]], [0, 1]) & \
                             np.isin(seg1[coords[:, 3], coords[:, 2]], [0, 1])
                    coords = coords[stable]
                    inliers = inliers[stable]
                    cv2_matches = [m for m, keep in zip(cv2_matches, stable) if keep]
                    if coords.size == 0:
                        print(f"All matches filtered for {img_path.stem} in {seq_dir.name}")
                        inliers = np.zeros(0, dtype=bool)
                        cv2_matches = []

            # Overlay masks for visualization.
            img0_vis = overlay_mask(img0_raw, seg0) if seg0 is not None else img0_raw
            img1_vis = overlay_mask(img1_raw, seg1) if seg1 is not None else img1_raw

            data = {
                "image1": img0_vis,
                "image2": img1_vis,
                "keypoints1": kpts1,
                "keypoints2": kpts2,
                "matches": coords,
                "inliers": inliers,
            }

            match_img = draw_matches_cv(
                data,
                cv2_matches,
                draw_keypoints=draw_keypoints,
                point_color=point_color,
                point_radius=point_radius,
            )
            cv2.imwrite(str(seq_out / f"{img_path.stem}_matches.png"), match_img)
            np.savez(seq_out / f"{img_path.stem}_matches.npz", **data)


if __name__ == "__main__":
    main()
