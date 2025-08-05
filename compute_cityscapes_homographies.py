from pathlib import Path
import numpy as np

# Utility functions for camera handling and homography computation
from utils.cityscapes_camera import (
    load_cityscapes_camera,
    simulate_ego_motion,
    compute_homography,
)


def process_split(split_root: Path, camera_root: Path, out_root: Path, resize=(512, 1024)):
    """Compute homography for every frame in a Cityscapes split."""
    for city_dir in sorted(split_root.iterdir()):
        if not city_dir.is_dir():
            continue
        fallback_cam = None
        for img_path in sorted(city_dir.glob("*.png")):
            frame = img_path.stem
            cam_json = camera_root / city_dir.name / f"{frame}_camera.json"
            if not cam_json.exists():
                if fallback_cam is None:
                    candidates = sorted(cam_json.parent.glob("*_camera.json"))
                    if candidates:
                        fallback_cam = candidates[0]  # fallback to first available camera calibration
                    else:
                        print(
                            f"Camera JSON missing for {img_path} and no fallback found, skipping"
                        )
                        continue
                cam_json = fallback_cam  # use fallback calibration when frame-specific file is missing

            K, R_cam, t_cam = load_cityscapes_camera(cam_json)  # load calibration
            original_width, original_height = 2048, 1024  # constants from Cityscapes
            scale_x = resize[1] / original_width
            scale_y = resize[0] / original_height
            K[0, :] *= scale_x  # scale intrinsics for resized images
            K[1, :] *= scale_y

            R_delta, t_delta = simulate_ego_motion()  # simulate a small motion
            R_warped = R_delta @ R_cam
            t_warped = t_cam + t_delta
            H = compute_homography(K, R_cam, t_cam, R_warped, t_warped)  # compute homography

            out_dir = out_root / city_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / f"{frame}.npy", H)  # save homography matrix


def main():
    base = Path("datasets/Cityscapes")
    image_root = base / "rightImg8bit"
    camera_root = base / "camera"
    homography_root = base / "homographies"

    for split_dir in sorted(image_root.iterdir()):
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        process_split(split_dir, camera_root / split, homography_root / split)


if __name__ == "__main__":
    main()
