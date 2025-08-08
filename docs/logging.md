# Logging

The **SuperSegmentaion** project records a rich collection of information during both training and evaluation.  Logging serves several purposes: it allows experiments to be reproduced, it provides immediate feedback on the health of a training run, and it creates artifacts that can be mined later for deeper analysis.  This document explains what information is logged, how the logging infrastructure is organised, and how to interpret common outputs.  Examples and placeholders for diagrams are provided so that you can adapt them to your own experiments.

> **Scope**: the practices described here apply to all training scripts (`Train_model_*` and `train4.py`), evaluation utilities, and the helper modules in `utils/`.  They assume the default configuration shipped with the repository, but the ideas can be transferred to custom setups.

## 1. Directory Layout and Log Targets

All experimental artefacts are rooted at the path defined in [`settings.py`](../settings.py).  The constant `EXPER_PATH` defaults to `logs`, which means that running a training script will produce a tree similar to

```
logs/
  experiment_name/
    config.yml
    checkpoints/
      superPointNet_0_checkpoint.pth.tar
      ...
    csv/
      summary.csv
    tensorboard/
      events.out.tfevents-...
    images/
      sample_001.png
```

The exact sub–directory names depend on the script, but the pattern is consistent: checkpoints go into a `checkpoints` folder, plain–text or CSV summaries are grouped under `csv`, and TensorBoard event files live inside `tensorboard`.  Each experiment receives its own timestamped directory thanks to the `getWriterPath` utility in [`utils/utils.py`](../utils/utils.py).  This prevents collisions between runs and allows long–term storage of multiple attempts.

* **Checkpoints** hold the raw model weights and optimiser state.
* **CSV summaries** capture scalar metrics over time for quick spreadsheet analysis.
* **TensorBoard** event files are used to display scalars, images, and additional custom visualisations.
* **Images and auxiliary files** are saved on demand during training or evaluation, for example when visualising keypoints or matches.

```
![Directory structure placeholder](images/logging_directory_placeholder.png)
```

The figure above is a placeholder illustrating the overall log directory layout.  Replace it with an actual diagram if you present this documentation to others.

## 2. Metrics and Artifacts Logged

Logging is not restricted to loss values.  SuperSegmentaion captures a broad spectrum of signals that collectively describe the behaviour of the model.  The following categories are the most important.

### 2.1 Scalar Metrics

Scalar metrics are the most common form of logging.  They include:

* **Training Losses** – Each training script records the total loss as well as its individual components (e.g. detector loss, descriptor loss, or subpixel refinement loss).  These are stored as TensorBoard scalars and optionally appended to CSV files using the `append_csv` helper.
* **Validation Metrics** – During evaluation phases, precision, recall, mean average precision (mAP), and other dataset–specific metrics are logged.
* **Learning Rate** – When schedulers are used, the current learning rate is recorded so that changes in model performance can be correlated with optimisation steps.

The raw values are gathered in Python dictionaries and fed to the TensorBoard `SummaryWriter` through calls like `writer.add_scalar('train/loss', loss, step)`.  The `append_csv` function in [`utils/utils.py`](../utils/utils.py) mirrors these writes to a human–readable CSV file, enabling analysis without TensorBoard.

### 2.2 Images

Images are central to visual tasks, so training scripts routinely log them.  Typical examples include:

* **Input and Augmented Images** – Raw input frames and their warped or augmented versions help debug data pipelines.
* **Predicted Heatmaps** – The network’s detector head produces heatmaps that are visualised and logged using `writer.add_image` after conversion to numpy arrays with `tensor2array`.
* **Matches and Keypoints** – Functions in `utils/draw.py` render keypoints on top of images or draw matching lines between image pairs.  The resulting images are saved to disk and also logged to TensorBoard.

### 2.3 Histograms and Distributions

When the need arises to inspect the distribution of weights or feature responses, `writer.add_histogram` is employed.  Although not used in every script, the capability is built in and easy to enable.

### 2.4 Checkpoints

Model checkpoints are saved via the `save_checkpoint` function in [`utils/utils.py`](../utils/utils.py).  The helper takes a directory, a network state dictionary, the current epoch, and an optional file name.  It automatically prefixes file names with the network type (`superPointNet`) and attaches the epoch number.  Loading is symmetrical: `load_checkpoint` retrieves the stored weights so that experiments can be resumed.

Checkpoints are crucial artefacts for reproducibility.  They allow you to roll back to previous states, transfer weights to new experiments, and perform ablation studies.  Because they are stored as standard PyTorch `state_dict` pickles, they can be inspected with external tools if required.

### 2.5 CSV and Plain–Text Logs

Not all logging needs to be interactive.  Some scripts favour lightweight text logs.  The `saveLoss` function writes entries such as

```
train iter: 500, loss: 0.183, {'lr': 0.0002}
```

into a text file, while `append_csv` collects rows of data for later spreadsheet analysis.  These functions are deliberately simple, encouraging quick inspection without the overhead of a GUI.

## 3. TensorBoard Integration

TensorBoard is the primary tool for visualising logs.  All training and evaluation scripts construct a `SummaryWriter` at start-up.  Consider the snippet from `train4.py`:

```python
from torch.utils.tensorboard import SummaryWriter
from utils.utils import getWriterPath

writer = SummaryWriter(getWriterPath(task=args.command,
                                     exper_name=args.exper_name,
                                     date=True))
```

The `getWriterPath` utility generates a directory of the form `runs/train_joint/experiment_2024-01-01_12-00-00`.  TensorBoard monitors this directory and updates graphs in real time.  Common usages include:

* `add_scalar(tag, scalar_value, global_step)` – Record training and validation losses, precision, recall, learning rate, or any numeric measurement.
* `add_image(tag, img_tensor, global_step, dataformats='CHW')` – Log images, heatmaps, or visualisations of keypoints and matches.
* `add_figure(tag, figure, global_step)` – Log Matplotlib figures for custom plots.
* `add_histogram(tag, values, global_step)` – Examine the distribution of activations or gradients.

### 3.1 Launching TensorBoard

After a script starts producing event files, run:

```
tensorboard --logdir runs
```

TensorBoard will open a web interface at `http://localhost:6006`.  The **Scalars** tab shows time series such as losses and accuracy, while the **Images** tab contains any images logged during training.  Histograms appear under their own tab when recorded.

```
![TensorBoard overview placeholder](images/tensorboard_overview_placeholder.png)
```

This placeholder represents a TensorBoard dashboard.  Replace it with an actual screenshot highlighting your most relevant plots.

### 3.2 Interpreting TensorBoard Outputs

Scalars typically reveal learning dynamics.  For instance, a decreasing training loss accompanied by a stagnating validation metric may suggest overfitting.  Logging learning rate alongside loss can reveal whether sudden spikes correspond to optimisation schedule steps.  Visualising images can expose mis–aligned keypoints or faulty augmentations.

Histograms can be used to detect gradient explosion or vanishing by observing the range of weight updates.  When a distribution collapses to zero, it may indicate dead neurons or a flawed preprocessing pipeline.

## 4. Custom Logging Tools

Beyond TensorBoard, SuperSegmentaion employs a set of lightweight utilities to improve readability and flexibility.

### 4.1 Colored Console Output

The module [`utils/custom_logging.py`](../utils/custom_logging.py) configures the standard Python `logging` module with the `coloredlogs` package.  Messages at `INFO` level appear in colour, making it easy to distinguish information, warnings, and errors in a scrolling terminal.  Because the configuration happens at import time, scripts simply `import utils.custom_logging` to activate the behaviour.

Two helper functions, `toRed` and `toCyan`, wrap text in colour codes and can be used for ad–hoc highlighting, for example:

```python
from utils.custom_logging import toRed
logging.info(toRed('Dataset path missing'))
```

### 4.2 Structured Printing

[`utils/print_tool.py`](../utils/print_tool.py) provides convenience functions for printing configuration dictionaries or object attributes in a structured manner.  The function `print_config` produces a block like:

```
==========  important config:  ==========
learning_rate :  0.001
batch_size   :  16
========== 32
```

This output can be redirected to a log file if needed by providing a `file` argument.

### 4.3 Textual and CSV Logging

`saveLoss` and `append_csv` were introduced earlier; here we expand on their typical use.  Many training loops maintain an in–memory dictionary of metrics that is periodically flushed to disk.  An example pattern is:

```python
metrics = {'loss': loss.item(), 'precision': precision, 'recall': recall}
append_csv(os.path.join(csv_dir, 'train.csv'), [epoch, step, metrics['loss'], metrics['precision'], metrics['recall']])
```

These functions do not enforce a schema; they simply append rows.  This flexibility is valuable during early experimentation when the set of metrics may evolve quickly.

### 4.4 Lightweight Image Dumps

The helper `saveImg` allows scripts to write raw numpy images to disk without relying on external libraries.  It is often used together with the visualisation helpers in `utils/draw.py` to create overlays of keypoints or matches.

## 5. Sample Log Excerpts and Interpretation

This section provides representative log snippets and suggestions on how to interpret them.  The exact numbers will differ in your experiments, but the structure is similar.

### 5.1 Training Output

```
[01/15/2024 12:00:05 INFO] train_joint: starting epoch 1/5
[01/15/2024 12:00:07 INFO] step 50/500 - loss: 1.234 (avg 1.567) - precision: 0.67 - recall: 0.59 - lr: 0.001000
[01/15/2024 12:00:09 INFO] step 100/500 - loss: 0.987 (avg 1.345) - precision: 0.71 - recall: 0.63 - lr: 0.001000
[01/15/2024 12:00:11 INFO] validation - mAP: 0.456 - mean desc err: 0.082
```

* **Epoch announcement** – Confirms the beginning of an epoch and indicates how many remain.
* **Step metrics** – Each line lists the current training step, the instantaneous loss, a running average, and any additional metrics.  A stable decrease in the running average is a good sign; sudden spikes may reveal bugs or learning–rate issues.
* **Validation results** – Periodic evaluation summarises the model’s ability to generalise.  Comparing validation metrics to training metrics highlights overfitting or underfitting.

### 5.2 Evaluation Output

```
[02/03/2024 09:15:31 INFO] Evaluating experiment baseline_2024-02-03
[02/03/2024 09:15:33 INFO] Loaded checkpoint superPointNet__checkpoint.pth.tar
[02/03/2024 09:15:45 INFO] Sequence urban_010 - precision: 0.82 - recall: 0.79 - mean repeatability: 0.68
[02/03/2024 09:15:52 INFO] Summary - precision: 0.80 +/- 0.02 - recall: 0.77 +/- 0.03 - F1: 0.785
```

* **Checkpoint loading** – Confirms that weights were successfully restored.
* **Per–sequence metrics** – Useful for spotting problematic subsets of data.  Outliers may warrant manual inspection of the corresponding images.
* **Summary statistics** – Aggregate numbers show overall performance.  The `+/-` values denote standard deviation across sequences.

### 5.3 TensorBoard Scalars

The following is a conceptual excerpt of how TensorBoard displays scalar logs; actual visuals will appear in the web interface.

```
step | train/loss | val/loss | val/mAP | lr
-----|------------|----------|---------|------
   0 |      2.031 |     NaN  |   NaN   | 0.0010
 100 |      1.512 |   1.820  | 0.134   | 0.0010
 200 |      1.221 |   1.509  | 0.201   | 0.0010
 300 |      1.040 |   1.372  | 0.238   | 0.0010
```

* Training loss decreases steadily, while validation loss lags behind—this is typical at early stages when the model is still learning generic features.
* The mAP metric climbs as descriptors become more discriminative.
* The learning rate remains constant; you would expect jumps if a scheduler were active.

```
![Loss curve placeholder](images/loss_curve_placeholder.png)
```

Insert an actual loss curve here to illustrate the trends summarised above.

### 5.4 Image and Match Visualisations

TensorBoard’s **Images** tab might contain entries like `train/sample_001` showing side–by–side comparisons of raw images and predicted keypoints.  When using the `draw_matches` function, another image appears with green lines linking matched features between image pairs.  These visualisations make it simple to spot systematic errors such as misplaced keypoints or mismatched orientation.

```
![Keypoint overlay placeholder](images/keypoint_overlay_placeholder.png)
```

## 7. Maintenance and Cleanup

Logs accumulate quickly. Rotate checkpoints to keep only the most relevant models, prune outdated TensorBoard runs, and archive CSV summaries so disk usage stays manageable.

## 9. Summary

Logging in SuperSegmentaion captures scalars, images, histograms, checkpoints, and text summaries.  TensorBoard offers an interactive view of these artefacts, while custom utilities provide colourful terminal output and quick text logs.  The system is flexible: you can add metrics, integrate new visualisations, or adapt the directory structure with minimal effort.  Proper use of these tools ensures that experiments are transparent, reproducible, and easy to debug.

## Key Takeaways

* All logs reside under the directory specified by `EXPER_PATH` (`logs/` by default).
* TensorBoard’s `SummaryWriter` records scalars, images, and more; launch TensorBoard with `tensorboard --logdir runs` to inspect them.
* Helper functions such as `save_checkpoint`, `append_csv`, and `saveLoss` create persistent artefacts for later analysis.
* `utils/custom_logging` and `utils/print_tool` enhance console readability with colours and structured output.
* Regularly interpret logs—loss curves, validation metrics, and visualisations—to catch issues early and guide experimentation.
