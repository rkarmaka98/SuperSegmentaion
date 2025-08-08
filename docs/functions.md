# Key Functions

A non-exhaustive list of entry points and helpers used across the project.

- `train()` in `Train_model_frontend.py`: main training loop handling batches and logging.
- `get_matches()` in `models/model_wrap.py`: performs descriptor matching using `PointTracker`.
- `export_descriptor()` in `export.py`: writes keypoints, descriptors and matches for evaluation.
- `sample_homography()` in `utils/homographies.py`: draws random homographies for data augmentation.
- `compute_miou()` in `evaluation.py`: calculates mean IoU for segmentation.

Refer to inline comments in the respective files for parameter details and usage examples.

