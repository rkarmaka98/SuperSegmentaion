"""Demo script to export feature matches on all Cityscapes train sequences."""

from pathlib import Path
import yaml
import torch
import numpy as np
import cv2

from Val_model_heatmap import Val_model_heatmap
from models.model_wrap import PointTracker
from evaluation import overlay_mask, draw_matches_cv


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

    # Tracker handles feature matching between consecutive frames.
    tracker = PointTracker(max_length=2, nn_thresh=val_agent.nn_thresh)

    resize = config["data"]["preprocessing"]["resize"]  # model expects this size.

    def load_frame(p: Path):
        """Read and normalize a grayscale frame."""
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (resize[1], resize[0]), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

    out_root = Path("sequence_demo_output")
    out_root.mkdir(exist_ok=True)

    # Loop over all sequence directories such as 'aachen', 'bochum', ...
    for seq_dir in sorted(train_root.iterdir()):
        if not seq_dir.is_dir():
            continue
        image_paths = sorted(seq_dir.glob("*.png"))
        if len(image_paths) < 2:
            continue  # need at least two frames for matching

        # Create a mirrored output directory for this sequence.
        seq_out = out_root / seq_dir.name
        seq_out.mkdir(parents=True, exist_ok=True)

        # Process consecutive frame pairs.
        for i in range(len(image_paths) - 1):
            tracker.clear_desc()  # start fresh for each pair

            # --- First frame of the pair ---
            img0 = load_frame(image_paths[i]).to(device)
            val_agent.run(img0)  # forward pass populates internal buffers
            pts0 = val_agent.heatmap_to_pts()[0]
            desc0 = val_agent.desc_to_sparseDesc()[0]
            seg0 = None
            if "segmentation" in val_agent.outs:
                seg0 = val_agent.outs["segmentation"].argmax(dim=1).cpu().numpy()[0]
            tracker.update(pts0, desc0)

            # --- Second frame of the pair ---
            img1 = load_frame(image_paths[i + 1]).to(device)
            val_agent.run(img1)
            pts1 = val_agent.heatmap_to_pts()[0]
            desc1 = val_agent.desc_to_sparseDesc()[0]
            seg1 = None
            if "segmentation" in val_agent.outs:
                seg1 = val_agent.outs["segmentation"].argmax(dim=1).cpu().numpy()[0]
            tracker.update(pts1, desc1)

            # Retrieve descriptor matches between the two frames.
            matches = tracker.get_mscores().T  # [L,3] -> query, train, score
            if matches.size == 0:
                continue

            # Prepare keypoint coordinates in (y, x) order for draw_matches_cv.
            kpts1 = pts0[[1, 0], :].T
            kpts2 = pts1[[1, 0], :].T
            cv2_matches = [
                cv2.DMatch(int(m[0]), int(m[1]), float(m[2])) for m in matches
            ]

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
            else:
                img0_vis, img1_vis = img0_np, img1_np

            # Draw matches and store a single visualization for this pair.
            data = {
                "image1": img0_vis,
                "image2": img1_vis,
                "keypoints1": kpts1,
                "keypoints2": kpts2,
                "matches": coords,
                "inliers": np.ones(matches.shape[0], dtype=bool),
            }
            match_img = draw_matches_cv(data, cv2_matches)
            cv2.imwrite(str(seq_out / f"matches_{i:05d}.png"), match_img)


if __name__ == "__main__":
    main()
