# Architecture

This document outlines the high-level structure of **SuperSegmentaion** and how the major components interact.

## Core Networks
- **SuperPointNet** (`models/SuperPointNet.py`): convolutional encoder–decoder that predicts keypoint heatmaps and 256-D descriptors.
- **SuperPointNet_gauss2** (`models/SuperPointNet_gauss2.py`): extends `SuperPointNet` with an Atrous Spatial Pyramid Pooling module and segmentation head.
- **SubpixelNet** (`models/SubpixelNet.py`): optional refinement network that regresses sub-pixel offsets.

All models share the same tensor dictionary output with `semi` for keypoint logits, `desc` for descriptors, and optional auxiliary fields.

## Supporting Modules
- **Utils** (`utils/`): geometry transforms, losses, augmentation and logging helpers.
- **Datasets** (`datasets/`): PyTorch datasets for synthetic and real imagery.
- **Training Scripts** (`Train_model_*`): orchestrate loading data, running forward passes, computing losses and updating weights.

## Flow
1. Images are loaded through dataset classes.
2. Forward passes through the network produce dense predictions.
3. Post-processing in `model_wrap.py` converts network outputs into keypoints and descriptors.
4. Evaluation modules consume these features for metric computation or matching tasks.

