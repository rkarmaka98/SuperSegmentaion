import argparse
import glob
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.draw import plot_imgs, warp_to_canvas


def main(args):
    """Generate visualizations for each prediction archive."""
    npz_paths = sorted(glob.glob(os.path.join(args.dir, "*.npz")))

    for npz_path in tqdm(npz_paths):
        data = np.load(npz_path)
        H = data["homography"]
        img1 = data["image"][..., np.newaxis]
        img2 = data["warped_image"][..., np.newaxis]

        h, w = img1.shape[:2]
        out_size = tuple(args.canvas_size) if args.canvas_size else (w, h)

        warped = warp_to_canvas(img1, H.astype(np.float32), out_size)

        img1_rgb = np.repeat(img1, 3, axis=2)
        img2_rgb = np.repeat(img2, 3, axis=2)
        warped_rgb = np.repeat(warped, 3, axis=2)

        plot_imgs(
            [img1_rgb, img2_rgb, warped_rgb],
            titles=["img1", "img2", "warp_to_canvas"],
            dpi=200,
        )

        base_name = os.path.splitext(os.path.basename(npz_path))[0]
        out_path = os.path.join(args.dir, f"test_{base_name}.png")
        plt.savefig(out_path)
        print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize homography warping using warp_to_canvas"
    )
    parser.add_argument(
        "dir", help="Directory containing *.npz prediction files"
    )
    parser.add_argument(
        "--canvas-size",
        metavar=("W", "H"),
        type=int,
        nargs=2,
        help="Size of the output canvas (width height)",
    )
    main(parser.parse_args())
