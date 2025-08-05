from pathlib import Path
import cv2
import argparse
import numpy as np  # needed for placeholder frames


def images_to_video(frame_dir: Path, output_path: Path, fps: int) -> None:
    """Combine image frames into a video file."""
    # Gather all PNG frames in sorted order
    images = sorted(frame_dir.glob("*.png"))  # type: list[Path]
    if not images:
        raise FileNotFoundError(f"No PNG images found in {frame_dir}")

    # Read first frame to determine video resolution
    first_frame = cv2.imread(str(images[0]))
    if first_frame is None:
        raise ValueError(f"Failed to read the first frame: {images[0]}")
    height, width, _ = first_frame.shape  # video resolution (h, w)

    # Initialize the video writer using mp4v codec
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    # Write each frame into the video
    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            # Insert a black frame to preserve timing when an image is missing
            print(f"Warning: {img_path} unreadable, inserting blank frame")
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        writer.write(frame)

    writer.release()  # finalize the video file


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a video from frames produced by sequence_demo.py."
        )
    )
    parser.add_argument(
        "--sequence",
        default="aachen",
        help="Name of the sequence folder inside sequence_demo_output",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=17,
        help="Frame rate for the output video",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for the resulting MP4 video",
    )
    args = parser.parse_args()

    out_root = Path("sequence_demo_output")
    frame_dir = out_root / args.sequence
    if args.output is None:
        output_path = out_root / f"{args.sequence}.mp4"
    else:
        output_path = Path(args.output)

    images_to_video(frame_dir, output_path, args.fps)
    print(f"Video saved to {output_path}")


if __name__ == "__main__":
    main()
