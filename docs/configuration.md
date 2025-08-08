# Configuration

Managing configuration for complex experiments can be as challenging as designing models or collecting data. This document explains how SuperSegmentaion organizes settings, where to store them, and how to safely override defaults for custom runs. It covers the hierarchy from base YAML files through command-line arguments, illustrates common directory layouts, and outlines versioning strategies that prevent configuration drift. The aim is to make configuration behavior predictable so that results can be reproduced and deployment workflows remain consistent across environments.

Throughout this document, references are made to other parts of the repository. The [Pipeline](./pipeline.md) description shows how configuration drives data flow, while the deployment notes in [docs.md](../docs.md#8-additional-scripts--tools) highlight scripts for packaging models or exporting artifacts. Configuration is thus a central piece in the project lifecycle—from local experimentation to large-scale deployment.

## 1. Configuration Hierarchy

The project uses a multi-layer system to balance sensible defaults with experiment-specific overrides. Each layer contributes to the final configuration dictionary used at runtime. The layers are merged in a deterministic order, with later stages overriding earlier ones when keys clash.

### 1.1 Base Configuration Files

The base layer resides in the `configs/` directory. These YAML files define canonical settings for model architecture, training regimes, dataset paths, and evaluation options. They are intentionally minimal, capturing only parameters that rarely change. The goal is to encode a shared vocabulary for the entire team.

A typical base file might look like this:

```yaml
# configs/base_training.yaml
model:
  name: superpoint  # core model to load
  descriptor_dim: 256  # descriptor dimensionality
train:
  epochs: 100  # training epochs
  batch_size: 8  # minibatch size
  optimizer: adam  # optimizer type
  learning_rate: 0.001  # base learning rate
```

YAML comments document the intent of each field. The convention is to group related parameters under top-level namespaces such as `model`, `train`, `data`, `eval`, and `logging`. The training scripts read the YAML into a dictionary and pass sub-trees to specific modules—for example, the `train` block goes to the optimizer setup while the `model` block is passed to the network factory.

### 1.2 Override Files

Experiment-specific changes are captured in separate override files. These files live in subdirectories like `configs/experiments/` or `configs/datasets/` and are merged on top of the base configuration. Using many small override files instead of editing base files helps avoid merge conflicts and keeps version history meaningful.

Example override:

```yaml
# configs/experiments/large_descriptor.yaml
model:
  descriptor_dim: 512  # expand descriptor size

data:
  dataset: cityscapes  # switch to Cityscapes dataset
  augmentations:
    - name: color_jitter  # random color jittering
      params:
        brightness: 0.2  # adjust brightness range
        contrast: 0.2  # adjust contrast range
```

When an experiment uses multiple overrides, they are merged in the order provided to the command line interface. This approach lets you compose configuration fragments: one file can specialize the dataset, another can adjust optimization parameters, and a third can toggle experimental features. Since merge order matters, list overrides from the most generic to the most specific.

The merge algorithm performs a deep update, meaning nested dictionaries are merged recursively. Lists are replaced wholesale rather than concatenated, which keeps semantics predictable.

### 1.3 Command-Line Interface

The final layer comes from command-line arguments. Training and evaluation scripts parse flags such as `--learning_rate` or `--model.name`. These flags override both base and override files, offering ad-hoc adjustments without editing configuration files.

Example command:

```bash
python train4.py \
  --config configs/base_training.yaml \
  --override configs/experiments/large_descriptor.yaml \
  --train.learning_rate=0.0005 \
  --train.batch_size=16
```

Nested keys use dot notation. When the parser encounters `--train.learning_rate`, it traverses the configuration dictionary and sets the value at `train -> learning_rate`. Type inference is based on the existing value: if `learning_rate` was a float, the parser converts the argument accordingly. CLI overrides are powerful but potentially dangerous because they can drift from version-controlled settings, so scripts log the final merged configuration to disk.

## 2. Working with YAML and JSON

YAML is the primary format for configuration files, but the codebase can also ingest JSON. JSON is useful when generating configs programmatically or interfacing with external services that already speak JSON. The internal representation after parsing is identical regardless of the source format.

A JSON variant of the base example above would look like:

```json
{
  "model": {
    "name": "superpoint", // core model to load
    "descriptor_dim": 256 // descriptor dimensionality
  },
  "train": {
    "epochs": 100, // training epochs
    "batch_size": 8, // minibatch size
    "optimizer": "adam", // optimizer type
    "learning_rate": 0.001 // base learning rate
  }
}
```

JSON does not natively support comments; the `//` markers above are for illustration only. If you require comments in JSON, consider a small preprocessor or use YAML for human-authored files.

The configuration loader checks file extensions: `.yaml`/`.yml` files use PyYAML, while `.json` files go through Python’s `json` module. Regardless of format, the loader returns a Python dictionary, and subsequent merging logic is agnostic to how the dictionary was created.

## 3. Directory Layouts

Consistency in directory structure makes it easier to find and share configurations. The repository adopts a simple layout:

```
configs/
├── base_training.yaml
├── experiments/
│   ├── large_descriptor.yaml
│   └── subpixel_refinement.yaml
└── datasets/
    ├── coco.yaml
    └── cityscapes.yaml
```

Each subdirectory groups files by purpose. `experiments/` contains overrides that tweak model behavior or optimization strategies, while `datasets/` provides dataset-specific options such as paths and augmentation pipelines. You are free to add additional folders, for example `schedules/` for learning rate schedules, as long as the hierarchy remains intuitive.

When running experiments, the `--config` flag points to the base file and `--override` accepts one or more override files. Paths are resolved relative to the repository root, so you can invoke commands from any working directory. Archiving configurations for long-term storage under `configs/archive/` ensures that historical results remain reproducible even if the main `configs/` folder evolves.

## 4. Configuration in the Pipeline

Configuration values influence every step of the pipeline described in [Pipeline](./pipeline.md). Dataset augmentations defined under the `data` namespace determine how images are preprocessed in Stage 2 (“Dataset Preparation”). Parameters under `train` govern Stage 5 (“Optimization”), including mixed precision and checkpointing frequency.

During evaluation and export, configuration again plays a role. Options under an `eval` namespace can toggle homography estimation or descriptor matching thresholds, which in turn affect metrics logged in the evaluation stage. Export scripts look for paths defined under `EXPER_PATH` (see [settings.md](./settings.md)) to find trained weights and determine where to store `.npz` artifacts. If evaluation results appear inconsistent, verify that training and evaluation used the same dataset configuration.

## 5. Versioning Strategies

Configuration files are code and deserve the same rigor as Python modules. Here are strategies to manage them effectively.

### 5.1 Semantic Naming

Use descriptive filenames that capture intent rather than incidental details. For example, `large_descriptor.yaml` communicates the primary change, whereas `exp1.yaml` requires opening the file to understand its purpose. Names should be stable; if the content evolves significantly, consider creating a new file rather than rewriting the old one.

### 5.2 Git Version Control

All configuration files live in the Git repository, enabling full history tracking. Commit messages should mention why changes were made, not just what changed. For example:

```
Improve homography augmentation range to handle wide FOV cameras
```

Such messages make it easier to trace experimental outcomes to configuration decisions. When configuration files are shared across multiple branches, avoid force-pushing or rebasing after publication to maintain a clear history.

### 5.3 Immutable Archives

For published experiments or benchmarks, copy the exact configuration files into an immutable archive under `configs/archive/`. Tag releases in Git that correspond to major milestones. The archive ensures that even if mainline configurations evolve, historical results remain reproducible.

### 5.4 Config Packages

When deploying models, bundle the exact configuration alongside the weights. The `export.py` script saves an `.npz` for keypoints and descriptors; augment it by copying the final merged configuration into the export folder. Consumers of the model can then inspect the configuration to understand how the model was trained.

### 5.5 Automated Validation

Consider adding unit tests or schema validation for configuration files. A simple JSON schema or `pykwalify` setup can catch typos such as `learningrate` instead of `learning_rate`. Running `pytest` as part of continuous integration prevents malformed configs from slipping into main. After each edit to configuration files, run the tests to ensure compatibility:

```bash
pytest
```

## 6. Deployment Considerations

Configuration does not end with training. Deployment scripts such as `run_export.sh` and ONNX conversion utilities rely on the same hierarchy to locate models and specify export parameters. For guidance on these utilities, refer to the deployment notes in [docs.md](../docs.md#8-additional-scripts--tools). Keeping configuration consistent between training and deployment ensures that exported models behave as expected.

When packaging models for external consumption, include a trimmed configuration file that contains only inference-relevant parameters. For instance, batch size and optimizer settings can be omitted, while model architecture and normalization options should be preserved. Use semantic versioning in deployment configuration filenames (e.g., `descriptor_v1.2_infer.yaml`) so that downstream teams know which model version they are using.

Environment-specific settings such as dataset paths or experiment directories are centralized in `settings.py` and documented in [settings.md](./settings.md). Adjust these paths before running deployment scripts to avoid file-not-found errors. Keeping environment variables in one place simplifies containerization and makes the project friendlier to cloud execution.

## 7. Common Pitfalls

1. **Silent Overrides**: Forgetting the merge order can lead to unexpected behavior when multiple overrides set the same key. Always check logs to confirm which value was applied.
2. **Path Dependencies**: Hard-coded paths in overrides can break on other machines. Use placeholders or environment variables from `settings.py` to keep configs portable.
3. **CLI Drift**: Over-reliance on command-line flags creates configurations that are hard to replicate. Whenever possible, encode changes in override files and keep CLI usage minimal.
4. **Validation Gaps**: Skipping tests after configuration changes risks runtime errors down the pipeline. Always run `pytest` or the relevant validation suite.

## 8. Conclusion

A disciplined approach to configuration management accelerates experimentation, reduces bugs, and ensures that results are comparable across time and collaborators. By leveraging base files, targeted overrides, and command-line tweaks in a controlled hierarchy, the SuperSegmentaion project balances flexibility with reproducibility. Consistent directory layouts and versioning strategies further streamline collaboration. When configurations are treated as first-class citizens—versioned, validated, and archived—the entire pipeline from data ingestion to deployment becomes more robust.

## Key Takeaways

- Start with minimal, well-commented base configurations and layer overrides for experiment-specific changes.
- Use command-line arguments sparingly; prefer version-controlled override files for reproducibility.
- Maintain a clear directory structure for configs and archive immutable copies for published results.
- Link configuration parameters directly to pipeline stages and deployment scripts to understand their downstream effects.
- Validate configuration changes with automated tests (`pytest`) to catch errors early.
- Include the final merged configuration in model exports to support consistent deployment.

