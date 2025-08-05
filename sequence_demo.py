"""Demo script to export heatmaps and stable segmentation matches on a Cityscapes
rightImg8bit sequence without modifying existing modules.
"""

from pathlib import Path
import yaml
import torch
import numpy as np
import cv2

# reuse utilities from existing pipeline
from utils.utils import tensor2array
from Val_model_heatmap import Val_model_heatmap
from models.model_wrap import PointTracker
from evaluation import overlay_mask, draw_matches_cv


def main():
    # Hardcoded path to a directory of rightImg8bit frames; adjust as needed.
    seq_dir = Path("datasets/Cityscapes/rightImg8bit/train/aachen")
    image_paths = sorted(seq_dir.glob("*.png"))
    if len(image_paths) < 2:
        raise RuntimeError("Need at least two frames to evaluate matches")

    # Load model configuration and weights.
    with open("configs/superpoint_cityscapes_export.yaml", "r") as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_agent = Val_model_heatmap(config["model"], device=device)
    val_agent.loadModel()

    # Tracker handles feature matching between consecutive frames.
    tracker = PointTracker(max_length=2, nn_thresh=val_agent.nn_thresh)

    resize = config["data"]["preprocessing"]["resize"]  # model expects this size.

    def load_frame(p):
        """Read and normalize a grayscale frame."""
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (resize[1], resize[0]), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

    out_dir = Path("sequence_demo_output")
    out_dir.mkdir(exist_ok=True)

    # --- First frame ---
    img0 = load_frame(image_paths[0]).to(device)
    heat0 = val_agent.run(img0)  # forward pass to obtain heatmap
    pts0 = val_agent.heatmap_to_pts()[0]
    desc0 = val_agent.desc_to_sparseDesc()[0]
    seg0 = None
    if "segmentation" in val_agent.outs:
        seg0 = val_agent.outs["segmentation"].argmax(dim=1).cpu().numpy()[0]
    tracker.update(pts0, desc0)  # store observations for matching
    heat0_color = tensor2array(torch.from_numpy(heat0[0]), colormap="bone", channel_first=False)
    cv2.imwrite(str(out_dir / "heatmap_0.png"), (heat0_color[:, :, ::-1] * 255).astype(np.uint8))

    # --- Second frame ---
    img1 = load_frame(image_paths[1]).to(device)
    heat1 = val_agent.run(img1)
    pts1 = val_agent.heatmap_to_pts()[0]
    desc1 = val_agent.desc_to_sparseDesc()[0]
    seg1 = None
    if "segmentation" in val_agent.outs:
        seg1 = val_agent.outs["segmentation"].argmax(dim=1).cpu().numpy()[0]
    tracker.update(pts1, desc1)
    heat1_color = tensor2array(torch.from_numpy(heat1[0]), colormap="bone", channel_first=False)
    cv2.imwrite(str(out_dir / "heatmap_1.png"), (heat1_color[:, :, ::-1] * 255).astype(np.uint8))

    # Retrieve descriptor matches between the two frames.
    matches = tracker.get_matches().T  # [L, 3] -> query, train, score
    if matches.size == 0:
        print("No matches were found between the first two frames.")
        return

    # Prepare keypoint coordinates in (y, x) order for draw_matches_cv.
    kpts1 = pts0[[1, 0], :].T
    kpts2 = pts1[[1, 0], :].T
    cv2_matches = [cv2.DMatch(int(m[0]), int(m[1]), float(m[2])) for m in matches]

    # Compute pixel coordinates of the matched keypoints.
    coords = np.zeros((matches.shape[0], 4), dtype=int)
    coords[:, 0] = pts0[0, matches[:, 0].astype(int)]  # x1
    coords[:, 1] = pts0[1, matches[:, 0].astype(int)]  # y1
    coords[:, 2] = pts1[0, matches[:, 1].astype(int)]  # x2
    coords[:, 3] = pts1[1, matches[:, 1].astype(int)]  # y2

    # Convert normalized tensors back to uint8 images for visualization.
    img0_np = (img0.cpu().numpy().squeeze() * 255).astype(np.uint8)
    img1_np = (img1.cpu().numpy().squeeze() * 255).astype(np.uint8)
    if seg0 is not None and seg1 is not None:
        img0_vis = overlay_mask(img0_np, seg0)
        img1_vis = overlay_mask(img1_np, seg1)
        stable = np.isin(seg0[coords[:, 1], coords[:, 0]], [0, 1]) & \
                 np.isin(seg1[coords[:, 3], coords[:, 2]], [0, 1])
    else:
        img0_vis, img1_vis = img0_np, img1_np
        stable = np.ones(matches.shape[0], dtype=bool)

    # Draw all matches on top of the (possibly overlayed) images.
    data = {
        "image1": img0_vis,
        "image2": img1_vis,
        "keypoints1": kpts1,
        "keypoints2": kpts2,
        "matches": coords,
        "inliers": np.ones(matches.shape[0], dtype=bool),
    }
    match_img = draw_matches_cv(data, cv2_matches)
    cv2.imwrite(str(out_dir / "matches.png"), match_img)

    # Visualise only matches that belong to stable segmentation classes (0 and 1).
    stable_cv2 = [m for m, keep in zip(cv2_matches, stable) if keep]
    if stable_cv2:
        stable_img = draw_matches_cv(data, stable_cv2)
        cv2.imwrite(str(out_dir / "stable_matches.png"), stable_img)

    tracker.clear_desc()  # reset tracker state


if __name__ == "__main__":
    main()
