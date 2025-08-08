# Pipeline

This document describes the data flow from raw images to final metrics.

1. **Configuration Loading**: `train4.py` and `Train_model_*` scripts read YAML files from `configs/` and merge CLI arguments.
2. **Dataset Preparation**: dataset classes in `datasets/` apply augmentations and output tensors.
3. **Forward Pass**: models from `models/` produce keypoint logits, descriptors and optional segmentation masks.
4. **Loss Computation**: losses in `utils/losses.py` combine detection, descriptor, segmentation and subpixel terms.
5. **Optimization**: training scripts handle optimizer steps, mixed precision and checkpoint saving.
6. **Export/Evaluation**: `export.py` and `evaluation.py` generate `.npz` artifacts and compute metrics.

