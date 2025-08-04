"""Utility script to turn exported demo frames into a video."""
import argparse
from pathlib import Path
import cv2


def main():
    parser = argparse.ArgumentParser(description="Create a demo video from exported SuperPoint frames")
    parser.add_argument("frames_dir", type=str,
                        help="Directory with PNG frames produced by export.py (--outputImg)")
    parser.add_argument("--output", type=str, default="demo.mp4",
                        help="Name of the generated video file")
    parser.add_argument("--fps", type=int, default=17,
                        help="Frame rate of the output video")
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    images = sorted(frames_dir.glob("*.png"))  # gather exported frames
    if not images:
        raise RuntimeError(f"No PNG images found in {frames_dir}")

    first = cv2.imread(str(images[0]))  # read one frame to get resolution
    height, width = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # use mp4 codec
    video = cv2.VideoWriter(args.output, fourcc, args.fps, (width, height))

    for img in images:
        frame = cv2.imread(str(img))  # load frame
        video.write(frame)            # append frame to video

    video.release()  # finalize writing
    print(f"Saved video to {args.output}")


if __name__ == "__main__":
    main()
