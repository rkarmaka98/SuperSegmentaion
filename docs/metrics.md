# Metrics

SuperSegmentaion evaluates local features and segmentation using a collection of standard metrics.

## Detector & Descriptor
- **Repeatability**: proportion of keypoints that reappear after geometric transforms. Implemented in `evaluations/detector_evaluation.py`.
- **Homography Accuracy**: average corner error after estimating a homography from matches (`evaluation.py`).
- **Descriptor Matching**: mean Average Precision and matching score (`evaluations/descriptor_evaluation.py`).

## Segmentation
- **Mean Intersection over Union (mIoU)**: computed in `evaluation.py` via `compute_miou` for segmentation masks.

## Logging
During evaluation, metrics are written to TensorBoard and CSV files for later analysis.

