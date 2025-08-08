# Architecture

This document presents a detailed architectural overview of **SuperSegmentaion**, linking together the motivation, internal structure, and extensibility of the project. It complements other guides such as the [configuration reference](configs.md) and the [pipeline overview](pipeline.md), and is intended for developers who wish to understand or extend the codebase. References to those documents are provided where configuration or pipeline logic is mentioned. Block diagram placeholders are included to illustrate where visual aids can be inserted in future iterations of this documentation.

## 1. Core Objectives

SuperSegmentaion aims to unify keypoint detection, descriptor learning, and segmentation within a single research environment. The repository was built with the following goals:

- **Consistency**: Every model returns predictions using a standard dictionary-based interface so that downstream utilities can remain agnostic to the specific network architecture.
- **Modularity**: Datasets, models, and utilities live in separate modules with minimal coupling. A change to one component should not require edits to others.
- **Reproducibility**: Configuration files in `configs/` capture hyperparameters, data paths, and training schedules to make experiments repeatable.
- **Extensibility**: Developers can add new models or datasets with minimal boilerplate, and the training scripts will automatically incorporate them once configured.

These goals drive the layout of the repository and inform the interaction between modules. Understanding this philosophy helps when navigating the code or designing new functionality.

## 2. Repository Layout

The top-level repository structure reflects the separation of concerns:

- **`models/`** – network definitions for keypoint detection, descriptor extraction, and optional segmentation heads.
- **`datasets/`** – dataset loaders and augmentation utilities built on top of PyTorch’s `Dataset` class.
- **`utils/`** – common helpers for geometry, losses, logging, and model wrappers.
- **`docs/`** – documentation for architecture, [configuration](configs.md), [pipeline](pipeline.md), [metrics](metrics.md), and [settings](settings.md).
- **`test/`** – a handful of unit tests verifying that core modules integrate correctly.
- **Root scripts** – training scripts (`Train_model_heatmap.py`, `Train_model_subpixel.py`, `Train_model_frontend.py`), evaluation scripts, and conversion utilities.

```
[Block Diagram Placeholder: repository tree]
```

Most scripts import modules using relative paths, which means the repository can be used without installing it as a Python package. This choice reduces friction for experimentation but requires developers to be mindful of the working directory when running commands.

## 3. Models

The `models` directory contains the neural networks that form the heart of SuperSegmentaion. All models inherit from `torch.nn.Module` and expose a forward method that accepts a batch dictionary and returns a dictionary of tensor outputs. This design allows utilities and training scripts to treat models interchangeably.

### 3.1 SuperPointNet

`SuperPointNet.py` implements a convolutional encoder–decoder model inspired by the original SuperPoint paper. The encoder consists of a sequence of convolutional blocks that gradually reduce spatial resolution while increasing channel depth. The decoder uses transpose convolutions and skip connections to restore the original resolution. The final layer outputs two tensors: `semi`, a heatmap of keypoint logits, and `desc`, a dense descriptor map where each pixel has a 256-dimensional descriptor. The model is trained using cross-entropy for keypoint detection and a metric learning loss for descriptor quality.

### 3.2 SuperPointNet_gauss2

`SuperPointNet_gauss2.py` extends the base architecture with an Atrous Spatial Pyramid Pooling module and an additional segmentation head. The ASPP module captures multi-scale context, while the segmentation head produces logits in the `segs` output field. This model is useful when keypoint detection and semantic segmentation should be trained jointly. Because the forward interface matches `SuperPointNet`, existing training and evaluation scripts can switch between models by changing the configuration.

### 3.3 SubpixelNet

`SubpixelNet.py` refines coarse keypoint locations to sub-pixel accuracy. It receives cropped feature patches around each detected keypoint and predicts offsets that maximise localisation precision. During training it is typically paired with one of the SuperPoint variants, but the refinement step is optional at inference time. The network outputs refined coordinates and confidence scores, packaged again as a dictionary.

```
[Block Diagram Placeholder: model interfaces]
```

## 4. Datasets

Dataset classes in `datasets/` provide input images, labels, and optional metadata. Each dataset returns a dictionary with keys required by the models and losses. Because the format is consistent, switching datasets requires only a configuration change.

### 4.1 Data Helpers and Transforms

- **`data_helpers.py`** centralises common routines such as image loading, homography estimation, and batch assembly. These functions ensure that geometric information is handled consistently across datasets.
- **`transforms.py`** defines photometric and geometric augmentations. Augmentations are composed and parameterised via configuration files. Examples include random brightness shifts, rotations, and perspective warps.

### 4.2 Example Dataset

`simple_sequence.py` illustrates the expected dataset interface. It loads a sequence of images, applies the configured transforms, and returns a dictionary containing an `image` tensor and metadata such as homographies or synthetic keypoints. Developers can use this file as a template when implementing loaders for real-world data.

The dataset abstractions are purposely lightweight: they rely on Python dictionaries rather than custom classes for labels. This strategy keeps the learning curve shallow and mirrors the output format of models and losses.

```
[Block Diagram Placeholder: dataset flow]
```

## 5. Utilities

Utilities provide glue code that ensures consistency across training and evaluation. Key modules include:

- **`utils/geometry.py`** – operations on homographies, coordinate frames, and perspective warps.
- **`utils/losses.py`** – custom loss functions for keypoint repeatability, descriptor distance, and segmentation accuracy.
- **`utils/augmentation.py`** – random photometric and geometric augmentations applied during training.
- **`utils/logging.py`** – wrappers around standard logging to standardise progress bars, metric aggregation, and checkpoint output.
- **`utils/model_wrap.py`** – converts raw network outputs into structured predictions. For instance, it performs non-maximum suppression on heatmaps and normalises descriptor vectors.

These utilities are imported across training scripts, evaluation scripts, and even dataset implementations. Keeping them in a single directory reduces duplication and provides a central place to update shared functionality.

```
[Block Diagram Placeholder: utility interactions]
```

## 6. Training and Evaluation Scripts

Training scripts are simple orchestrators that assemble datasets, models, and utilities. Despite their brevity, they encapsulate the standard training loop used by most experiments in the repository.

### 6.1 Common Workflow

1. **Configuration Parsing** – A script reads a configuration file from `configs/` or command-line arguments that override default values. The [configuration reference](configs.md) enumerates available parameters.
2. **Component Construction** – Based on configuration names, the script instantiates dataset and model classes. Augmentations and loss functions are configured here.
3. **Optimiser and Scheduler Setup** – Learning rates, weight decay, and scheduler policies are derived from configuration values.
4. **Training Loop** – For each batch, the script performs forward passes, computes losses, backpropagates gradients, and steps the optimiser. Metrics are recorded via `utils/logging`.
5. **Validation and Checkpointing** – At intervals, the model is evaluated using utilities in `utils/model_wrap.py` and metrics defined in `docs/metrics.md`. Checkpoints are saved so training can resume later.

### 6.2 Evaluation Scripts

Evaluation scripts such as `Val_model_heatmap.py` reuse much of the training logic but focus on metric computation. They load saved checkpoints, run the model on a validation dataset, and output scores like repeatability or segmentation accuracy. Because models return dictionaries, evaluation code can inspect keys to determine whether, for example, segmentation metrics should be computed.

```
[Block Diagram Placeholder: training workflow]
```

## 7. Data Flow and Module Responsibilities

To illustrate how components interact, consider a typical training iteration:

1. **Data Loading** – The `DataLoader` requests a batch from a dataset class. The dataset returns a dictionary containing at least an `image` tensor and any auxiliary labels.
2. **Model Forward Pass** – The batch dictionary is sent to a model. The model reads the `image` key and optional metadata, producing predictions such as `semi`, `desc`, or `segs`.
3. **Post-processing** – `utils/model_wrap.py` processes raw outputs into intermediate representations: keypoint coordinates are extracted, descriptors are normalised, and segmentation logits may be upsampled.
4. **Loss Computation** – Functions in `utils/losses.py` compare predictions with ground truth labels provided by the dataset. For instance, the keypoint loss uses cross-entropy on the `semi` tensor, while descriptor losses operate on `desc` pairs.
5. **Optimisation** – The training script calls `loss.backward()` and steps the optimiser.
6. **Logging and Checkpointing** – Metrics and losses are sent to `utils/logging.py`. At scheduled intervals, checkpoints and configuration files are saved for reproducibility.
7. **Validation** – The evaluation scripts load checkpoints, run inference using the same dataset interface, and compute metrics using utilities from `docs/metrics.md`.

This pipeline demonstrates each module’s responsibility: datasets supply data, models compute predictions, utilities transform and evaluate those predictions, and scripts manage execution.

```
[Block Diagram Placeholder: end-to-end data flow]
```

## 8. Extensibility Guidelines

SuperSegmentaion is intended as a foundation for further research. The following guidelines help maintain consistency when introducing new functionality.

### 8.1 Adding a Model

1. Create a new file under `models/` that defines a class inheriting from `torch.nn.Module`.
2. Ensure the `forward` method accepts a batch dictionary and returns a dictionary. At minimum, include `semi` and `desc` if the model supports keypoint detection.
3. Update training scripts to import and instantiate the new model when the configuration specifies its name.
4. Document new configuration options in `configs/` and, if necessary, expand this architecture guide or other docs with usage instructions.

### 8.2 Adding a Dataset

1. Implement a new class in `datasets/` that follows the interface used by `simple_sequence.py`.
2. Reuse transforms from `transforms.py` or add new ones if required. Ensure that fields like `image` and labels align with model expectations.
3. Register the dataset in configuration files so training scripts can load it via a name field.

### 8.3 Extending Utilities

- **Losses or Geometry** – Add new functions to `utils/losses.py` or `utils/geometry.py`. Provide docstrings and, ideally, unit tests in `test/`.
- **Logging** – Extend `utils/logging.py` for custom metric aggregation or visualisation backends.
- **Model Wrapper** – If a new model introduces unique outputs, modify `utils/model_wrap.py` to handle them while preserving existing behaviour.

### 8.4 Diagram Placeholders

Wherever this document includes placeholders, contributors can insert diagrams hosted in `docs/images/`. Images should have descriptive filenames and alt text. For example:

```
![Sample Pipeline](images/pipeline_placeholder.png)
```

## 9. Interaction with Configuration and Pipeline

Configuration files define the experiment’s behaviour and tie components together. The [pipeline overview](pipeline.md) elaborates on how these pieces interact, but the key points are summarised here:

- **Config Loading** – Training scripts load YAML files specifying dataset paths, model names, optimiser settings, and scheduler policies. Overrides from the command line merge with these defaults.
- **Parameter Propagation** – Once loaded, configuration dictionaries are passed to dataset constructors, model initialisers, and utility functions. For example, augmentation parameters in the config are forwarded to `transforms.py`.
- **Reproducibility** – Configuration files are saved alongside checkpoints, allowing others to rerun experiments with identical settings.

By keeping configuration separate from code, the repository achieves reproducibility without littering scripts with hard-coded constants. Developers should consult the [configuration reference](configs.md) whenever adding new parameters.

## 10. Future Improvements

The current architecture has proven effective for experimentation, but several enhancements could improve usability:

- **Diagram Completion** – Replacing placeholders with actual diagrams will make the flow of data and control easier to grasp.
- **Package Structure** – Converting the repository into an installable package would allow cleaner imports while preserving script-level convenience through entry points.
- **Comprehensive Tests** – Expanding the `test/` directory to cover models, datasets, and utilities would reduce regressions as the codebase evolves.
- **Plugin Registry** – A registry that discovers models and datasets automatically could simplify extensibility, removing the need to modify training scripts when new components are added.

## 11. Key Takeaways

- The repository is organised around models, datasets, utilities, and scripts, each with clear responsibilities.
- Data flows through a batch dictionary, enabling components to remain decoupled yet interoperable.
- Training scripts assemble the pipeline using configuration files, making experiments reproducible and easy to modify.
- Utility modules handle cross-cutting concerns such as geometry, losses, logging, and post-processing.
- Extensibility is central to the design; new models or datasets can be integrated by following established patterns and updating configuration files.

As SuperSegmentaion grows, contributors are encouraged to keep this document up to date, add diagrams to the placeholders, and cross-reference related documentation to maintain a cohesive knowledge base.

