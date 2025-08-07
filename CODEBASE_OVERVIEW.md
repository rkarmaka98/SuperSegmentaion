# SuperSegmentaion Codebase Documentation

## 1. Overview
SuperSegmentaion is a research-grade implementation of SuperPoint-style keypoint detection, description, and matching, extended with optional sub-pixel refinement and semantic segmentation. Training scripts, evaluation utilities, dataset loaders, and various helper functions are provided to reproduce and extend the results reported in the accompanying paper and README.

### 1.1 Theoretical Foundations
- **Keypoint Detection & Description** – Builds upon SuperPoint’s self-supervised framework, where synthetic corner detection and homography adaptation supervise a fully convolutional network that predicts interest point heatmaps and 256‑D descriptors [1].
- **Semantic Segmentation** – Employs UNet-style decoders and Atrous Spatial Pyramid Pooling (ASPP) to capture multi-scale context, following ideas popularized by DeepLab and related segmentation literature [2][3].
- **Sub-pixel Refinement** – Uses differentiable soft-argmax to regress offsets within detection cells, similar to techniques used for pose estimation and optical flow refinement [4].
- **Evaluation Metrics** – Detector repeatability, homography estimation accuracy, and descriptor matching scores follow the definitions established in the HPatches benchmark [5].

---

## 2. Directory Structure

| Path | Description |
|------|-------------|
| `models/` | Neural network architectures and matching helpers. |
| `utils/` | General utilities: logging, geometry, losses, data loading, etc. |
| `datasets/` | Dataset definitions and preprocessing tools. |
| Root scripts | Training (`Train_model_*`), validation (`Val_model_*`), export, evaluation, and ancillary tools. |
| `configs/` | YAML configuration files for different training/evaluation setups. |
| `evaluations/` | Metric computations for homography, detector repeatability, descriptor quality, segmentation. |
| Misc. scripts | ONNX export, dataset conversions, demos, video generation, etc. |

---

## 3. Models (`models/`)

### 3.1 `SuperPointNet.py`
- **SuperPointNet**: Encoder–decoder CNN with shared backbone for detection and description. The detection head outputs a 65‑channel tensor (8×8 grid plus dustbin) that is softmax‑normalized and later subjected to non‑maximum suppression, mimicking the pipeline of the original SuperPoint architecture [1]. The descriptor head produces 256‑D vectors that are L2 normalized for cosine similarity. Supports BN‑ReLU ordering via `reBn` flag, max‑pooling/unpooling for feature maps, and optional subpixel head (commented).
  - `forward(x, subpixel=False)`: Returns `{ 'semi': heatmap, 'desc': descriptors }` with L2 normalization.
  - `forward_original(x)`: Baseline forward pass without BatchNorm/skip logic.
  - Helper conv/upconv builders at file end.

### 3.2 `SuperPointNet_gauss2.py`
- **ASPP**: Lightweight Atrous Spatial Pyramid Pooling module for segmentation, enabling dense predictions at multiple receptive field sizes [3].
- **SuperPointNet_gauss2**: UNet-style architecture producing detection heatmap, descriptors, and (optionally refined) segmentation logits. Joint learning encourages shared features for geometric and semantic tasks.
  - `forward(x)`: Runs encoder, heads, and segmentation refinement; outputs dictionary with `semi`, `desc`, and `segmentation`.
  - `process_output(sp_processer)`: Converts raw outputs to keypoints, offsets, descriptors using post-processing utilities.
- `get_matches(deses_SP)`: Uses `PointTracker` for symmetric NN matching.
- `main()`: Self-test and throughput benchmarking.

### 3.3 `SubpixelNet.py`
UNet used for sub-pixel coordinate regression.
- `soft_argmax_2d(patches)`: Differentiable argmax using Kornia, equivalent to computing the expectation over a probability map [4].
- `forward(x, subpixel=False)`: Produces `semi`, `desc`, and optionally subpixel offsets that refine coarse keypoint locations.

### 3.4 `SuperPointNet_pretrained.py`
Minimal SuperPoint network compatible with MagicLeap’s pretrained weights plus `PoseExpNet` for pose/exp mask estimation (separate task).

### 3.5 Utility Modules
- `model_wrap.py`: `SuperPointFrontend_torch` wraps inference-time preprocessing/post-processing. Implements grid-based NMS (`nms_fast`), model loading, multi-GPU support, and exposes keypoints/descriptors.
- `model_utils.py`: auxiliary operations (e.g., spatial soft-argmax, descriptor sampling).
- `classical_detectors_descriptors.py`: SIFT/ORB wrappers for classical baselines.
- `homographies.py`: geometric transformation helpers.
- `unet_parts.py`: building blocks (`double_conv`, `down`, `up`, `outconv`) reused by UNet-like nets.

---

## 4. Utilities (`utils/`)

### 4.1 `utils.py`
Large collection of helper routines:
- Image/array utilities (`tensor2array`, `img_overlap`, `thd_img`).
- Logging/checkpointing (`save_checkpoint`, `load_checkpoint`, `saveLoss`, `append_csv`).
- Geometry and warping (`warp_points`, `inv_warp_image_batch`, `labels2Dto3D`, `labels2Dto3D_flattened`). Homography-based warps assume locally planar scenes and are central to homography adaptation during training [1].
- Homography sampling and scaling (`sample_homography`, `homography_scaling`).
- Misc: loading configs, saving images, compute overlaps, etc.

### 4.2 `loader.py`
Data and model loaders:
- `dataLoader` / `dataLoader_test`: build PyTorch `DataLoader`s for training/validation/testing with configurable worker seeds and collate functions.
- `modelLoader`: imports model classes dynamically.
- `pretrainedLoader`: loads checkpoints (full or weight-only).
- `get_save_path`, `worker_init_fn`, `filter_none_collate`: training utilities.

### 4.3 Losses and Augmentation
- `loss_functions/` & `losses.py`: descriptor losses (dense/sparse), segmentation loss, subpixel loss, etc.
- `photometric.py` / `photometric_augmentation.py`: augmentation pipelines that simulate illumination changes, blurs, and perspective variations to encourage robustness.
- `homographies.py` (under `utils/`): sampling random homographies.
- `correspondence_tools/`: keypoint correspondence utilities.

### 4.4 Miscellaneous
- `cityscapes_camera.py`, `coco_panoptic_to_cs34.py`, `remove_unpaired_coco_images.py`: dataset-specific scripts.
- `print_tool.py`: structured logging/printing helpers.
- `var_dim.py`: tensor dimension manipulation.
- `draw.py`: visualization helpers.

---

## 5. Datasets (`datasets/`)

Each dataset module inherits `BaseDataset` or implements PyTorch dataset interface.

- **`base_dataset.py`**: Abstract TensorFlow-based dataset skeleton with `split_names`, `_init_dataset`, `_get_data`, and generator utilities.
- **Synthetic datasets**: `SyntheticDataset_gaussian.py`, `synthetic_dataset.py`, `synthetic_shapes.py` generate artificial corners for pretraining.
- **Real datasets**:
  - `Coco.py`, `CocoPanoptic.py`, `Cityscapes.py`, `Kitti_inh.py`, `Apollo.py`, `Tum.py`: Handle reading images, labels, optional homography warping, segmentation masks, and data augmentation for real-world datasets such as COCO [6] and Cityscapes [7].
  - `patches_dataset.py`: HPatches patch extraction for benchmarking [5].
  - `data_tools.py`, `utils/` subfolder: reading annotations, resizing masks (`resize_cityscapes_masks.py`).

---

## 6. Training & Validation Scripts

### 6.1 `Train_model_frontend.py`
Generic trainer for joint detector/descriptor (and optional segmentation) learning.
- Handles configuration merging, AMP setup, logging, tensorboard output.
- `dataParallel`, `adamOptim`, `loadModel`, `train`: orchestrate optimization loop with periodic validation and checkpointing.
- `train_val_sample`: core iteration: forward pass, loss computation (dense or sparse descriptors, detector loss via cross‑entropy, segmentation loss via per-pixel cross‑entropy, subpixel regression via smooth L1), optional subpixel training, AMP scaling, and tensorboard logging.
- `getLabels`, `getMasks`, `get_loss`: format ground truth and compute detector losses derived from homography-adapted keypoints [1].

### 6.2 `Train_model_heatmap.py`, `Train_model_subpixel.py`, `Val_model_heatmap.py`, `Val_model_subpixel.py`
Specialized trainers/validators focusing on heatmap detection or subpixel refinement.

---

## 7. Evaluation & Export

### 7.1 `evaluation.py`
- Computes detector repeatability, descriptor matching metrics, and homography accuracy following HPatches protocols [5].
- `draw_matches_cv`: custom OpenCV visualization for matches with optional keypoints.
- `compute_miou`: mean IoU for segmentation masks.
- Command-line interface supports exporting images, matches, segmentation overlays, and logging metrics via tensorboard.

### 7.2 `export.py`
Exports keypoints, descriptors, matches (and optionally segmentation masks) from trained models.
- `export_descriptor`: loads model checkpoints, iterates through test datasets, processes pairs, applies `PointTracker` for two-way matching, saves `.npz` per image pair.

### 7.3 `export_classical.py`
Equivalent exporter for classical detectors/descriptors (SIFT, ORB), enabling comparison.

### 7.4 `summarize.py`, `evaluation` package
Utility scripts to summarize results (`summary.csv`) and provide evaluation backends (`descriptor_evaluation.py`, `detector_evaluation.py`).

---

## 8. Additional Scripts & Tools
- `save_to_onnx.py`, `run_export.sh`, `scheduler.sh`: deployment and job-scheduling helpers.
- `create_video.py`, `sequence_demo.py`: visualization demos.
- `compute_cityscapes_homographies.py`: precompute homographies for Cityscapes sequences.
- `settings.py`: global paths for datasets and outputs.
- `train4.py`: top-level script that parses configs and launches training/evaluation commands.

---

## 9. Results
The README reports benchmark results on HPatches (homography estimation, repeatability, descriptor mAP, matching score) for pretrained and custom models. Typical performance aligns with the original SuperPoint paper [1]; models trained on COCO and KITTI achieve competitive repeatability and matching scores. The repository provides pretrained weights for COCO and KITTI and scripts to reproduce these metrics.

---

## 10. References
1. DeTone, D., Malisiewicz, T., & Rabinovich, A. *SuperPoint: Self-Supervised Interest Point Detection and Description.* CVPR Workshops, 2018.
2. Ronneberger, O., Fischer, P., & Brox, T. *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI, 2015.
3. Chen, L.-C., Papandreou, G., Kokkinos, I., Murphy, K., & Yuille, A. L. *DeepLab: Semantic Image Segmentation with Atrous Convolution.* arXiv:1606.00915, 2017.
4. Sun, X., Xiao, B., Liu, F., & Wang, J. *Integral Human Pose Regression.* ECCV, 2018.
5. Balntas, V., Lenc, K., Vedaldi, A., & Mikolajczyk, K. *HPatches: A Benchmark and Evaluation of Handcrafted and Learned Local Descriptors.* CVPR, 2017.
6. Lin, T.-Y., et al. *Microsoft COCO: Common Objects in Context.* ECCV, 2014.
7. Cordts, M., et al. *The Cityscapes Dataset for Semantic Urban Scene Understanding.* CVPR, 2016.

---

### Note
This documentation summarizes file-level responsibilities and major classes/functions. Due to the repository’s size and research nature, exhaustive line-by-line descriptions or generated API docs are beyond the scope of this document. For deeper exploration, consult individual modules and inline comments within the codebase.

