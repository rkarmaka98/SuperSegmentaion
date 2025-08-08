# Settings and Deployment

## Introduction
SuperSegmentaion combines classical geometry with modern deep learning to provide high‑quality keypoint detection and descriptor models. Deploying these models in real products requires careful attention to both the software environment and the optimization strategies used to fit models onto diverse hardware. This guide covers recommended operating systems, dependencies, and hardware; outlines differences between development and production workflows; and dives into optimization techniques—mixed precision, quantization, and pruning—that help models run efficiently. Sample commands are included throughout to ground explanations in practical usage. The document concludes with key takeaways summarizing best practices.

## Environment Setup

A reproducible environment is critical for training and deploying SuperSegmentaion models. The project targets Linux, but the instructions generalize to macOS and Windows via WSL2. The following components form the baseline environment.

### Operating system
- **Linux:** Ubuntu 20.04 or 22.04 LTS are recommended because the PyTorch ecosystem releases precompiled CUDA packages for these versions.
- **Windows:** Install WSL2 with an Ubuntu image to obtain a Linux‑like terminal. GPU passthrough to WSL2 requires recent Windows 11 builds.
- **macOS:** While training with GPU is not supported, the CPU‑only workflow allows experimentation and deployment on Apple Silicon using the Metal backend.

### Python and package management
- **Python version:** 3.9 or higher. The repository was tested with Python 3.10.
- **Virtual environment:** Using `venv` or `conda` keeps project dependencies isolated.
- **Installation:**
  ```bash
  # create environment
  python -m venv .venv
  source .venv/bin/activate
  # install dependencies
  pip install -r requirements.txt
  ```
  The file `requirements_torch.txt` lists a minimal PyTorch stack and can be used when customizing CUDA versions.

### Core dependencies
- **PyTorch:** Provides tensor computation and autograd. The project uses the 2.x series for improved kernel fusion.
- **OpenCV:** Enables image manipulation and visualization. Ensure that the Python bindings include GUI support if you plan to run demos.
- **NumPy and SciPy:** Offer linear algebra routines used in homography estimation and evaluations.
- **Logging utilities:** The repository integrates with `tensorboard` for visualization and `tqdm` for progress bars.

### Hardware requirements
- **GPU:** A CUDA‑capable NVIDIA card with at least 6 GB of memory is recommended for training. Cards like the RTX 3060 or higher provide a good balance of cost and performance.
- **CPU:** Modern x86 processors with eight or more cores shorten preprocessing tasks.
- **Memory:** 16 GB RAM is sufficient for most experiments, but large‑scale training benefits from 32 GB.
- **Storage:** Datasets such as COCO and Cityscapes demand tens of gigabytes; use SSDs to avoid training bottlenecks.

### Optional: Docker
Containerization enhances reproducibility and simplifies distribution to production servers. A minimal Dockerfile might install system libraries, copy the repository, and set the entry point to a training or export script. Example:
```Dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3-pip git
WORKDIR /app
COPY . /app
RUN pip3 install -r requirements.txt
ENTRYPOINT ["python3", "Train_model_heatmap.py"]
```
Use `docker run --gpus all` to expose GPUs to the container.

## Development vs. Production Workflows

SuperSegmentaion supports rapid experimentation while encouraging practices that simplify deployment. Although both workflows share the same code base, their goals differ.

### Development workflow
During development, the priority is flexibility and visibility. Typical characteristics include:
- **Editable code:** Developers work in a virtual environment on a cloned repository. Git branches track experimental ideas, and debugging instrumentation—such as extensive logging and assertions—is often left enabled.
- **Dataset subsets:** To iterate quickly, developers may run experiments on a small portion of datasets.
- **Interactive tools:** Jupyter notebooks or TensorBoard dashboards help diagnose training issues.

A basic training command for development might look like:
```bash
python Train_model_heatmap.py \
    --config configs/magicpoint_kitti_train.yaml \
    --batch_size 8 \
    --experiment dev_run
```
This command references a YAML configuration. The repository’s `docs/configs.md` explains each field and how to override settings on the command line.

### Production workflow
Production deployments aim for stability, reproducibility, and efficient inference. Key practices include:
- **Frozen dependencies:** Use a lockfile or Docker image to pin exact package versions. Avoid system Python.
- **Automated pipelines:** Training and evaluation run in CI pipelines or scheduled jobs. Scripts such as `scheduler.sh` demonstrate batch job submission.
- **Artifact versioning:** Trained checkpoints, exported ONNX graphs, and evaluation metrics are stored in a dedicated `EXPER_PATH` defined in `settings.py`. Tracking artifacts with tools like MLflow or DVC further strengthens reproducibility.
- **Reduced logging:** Only essential metrics are emitted to minimize overhead.
- **Continuous testing:** The `test` directory houses unit tests that guard against regressions. Run them in your CI environment.

Deployment to embedded devices or cloud servers often uses the export utilities bundled with the repository. For example, converting a trained model to ONNX for inference in a C++ service:
```bash
python save_to_onnx.py \
    --checkpoint $EXPER_PATH/superpoint/best.pth \
    --output superpoint.onnx
```
The produced file can be loaded by TensorRT or OpenVINO runtimes for accelerated inference.

## Optimization Techniques

Optimization ensures that models meet real‑time requirements and fit within device memory budgets. SuperSegmentaion models are relatively lightweight but still benefit from three major techniques: mixed precision, quantization, and pruning.

### Mixed precision
Mixed precision uses 16‑bit floating‑point numbers for most operations while keeping a 32‑bit master copy of weights to preserve accuracy. On modern GPUs, especially those with Tensor Cores, this yields substantial speedups.

#### Training with automatic mixed precision (AMP)
PyTorch offers automatic mixed precision through the `torch.cuda.amp` module. Enabling it in the training loop is straightforward:
```python
scaler = torch.cuda.amp.GradScaler()  # manages scaling to prevent underflow
for data in loader:
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        outputs = model(data)             # forward pass in FP16
        loss = criterion(outputs, data)
    scaler.scale(loss).backward()        # scales gradients
    scaler.step(optimizer)               # unscales and steps
    scaler.update()                      # adjusts scaling factor
```
This pattern can be enabled in SuperSegmentaion scripts by wrapping the forward pass and optimizer steps. The reduction in memory footprint also enables larger batch sizes.

#### Inference with mixed precision
For inference, cast the model and inputs to half precision:
```python
model.half()                              # convert weights to FP16
input = input_tensor.half()
with torch.no_grad():
    pred = model(input)
```
Some layers (e.g., batch normalization) may require staying in FP32. Adding explicit `.float()` conversions around sensitive operations prevents numerical issues.

### Quantization
Quantization converts floating‑point weights to lower‑precision integers, often 8‑bit. This reduces model size and can accelerate inference on CPUs and edge accelerators.

#### Dynamic quantization
Dynamic quantization works well for models dominated by linear layers. PyTorch provides a high‑level API:
```python
from torch.quantization import quantize_dynamic
quantized_model = quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```
The quantized model can be exported to ONNX and executed in runtimes that support integer arithmetic. This approach leaves activations in FP32, which minimizes accuracy loss but still shrinks the model footprint.

#### Static (post‑training) quantization
For convolutional networks, static quantization requires calibration with representative data:
```python
model.eval()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
prepared = torch.quantization.prepare(model)
# run calibration
for imgs, _ in calib_loader:
    prepared(imgs)
quantized_model = torch.quantization.convert(prepared)
```
Using a small calibration dataset, often a few hundred images, yields better activation ranges. The resulting model can be significantly faster on CPUs that support vectorized integer instructions.

#### Quantization‑aware training
When accuracy after static quantization is unacceptable, quantization‑aware training (QAT) injects fake quantization nodes during training. SuperSegmentaion scripts can be adapted by calling `torch.quantization.prepare_qat` before training begins. Because QAT increases training complexity, it is generally reserved for production deployments with strict latency targets.

### Pruning
Pruning removes redundant connections to reduce model size and computation. While SuperSegmentaion models are already small, pruning helps when deploying to microcontrollers or when bandwidth is limited.

#### Unstructured pruning
Unstructured pruning zeroes individual weights based on magnitude. PyTorch’s `nn.utils.prune` module provides utilities:
```python
import torch.nn.utils.prune as prune
prune.l1_unstructured(module.conv1, name='weight', amount=0.2)  # prune 20%
```
After pruning, call `prune.remove(module.conv1, 'weight')` to make the sparsity permanent. This technique works well with sparse tensor libraries but may not accelerate inference on standard hardware unless the underlying runtime exploits sparsity.

#### Structured pruning
Structured pruning removes entire channels or filters, leading to dense models that standard libraries accelerate easily:
```python
prune.ln_structured(module.conv1, name='weight', amount=0.3, n=2, dim=0)
```
After pruning, run a brief fine‑tuning phase to recover accuracy. Structured pruning changes layer dimensions, so the corresponding configuration files and pre‑trained weights must be updated.

#### Automated pruning policies
Automated methods such as magnitude‑based iterative pruning or reinforcement learning can discover optimal sparsity patterns. While not built into SuperSegmentaion, these approaches integrate by wrapping the training loop. Keep records of the pruning masks along with the model checkpoint for reproducibility.

### Combining techniques
Mixed precision, quantization, and pruning are complementary. A typical deployment pipeline might train with mixed precision for speed, prune the resulting model, fine‑tune, and finally quantize for inference. Always measure accuracy and latency after each step to ensure the model meets requirements.

## Sample Commands and Configurations

This section aggregates frequently used commands and references to configuration files so that new users can get started quickly.

### Training
```
# Train MagicPoint on KITTI using the provided YAML configuration
python Train_model_heatmap.py \
    --config configs/magicpoint_kitti_train.yaml \
    --batch_size 16 \
    --experiment kitti_baseline
```
The configuration file defines optimizer settings, data paths, and augmentation parameters. Refer to `docs/configs.md` for a comprehensive description of each key.

### Evaluation
```
# Evaluate the SuperPoint model on a validation set
python Val_model_heatmap.py \
    --config configs/superpoint_cityscapes_finetune.yaml \
    --checkpoint $EXPER_PATH/superpoint/best.pth
```
Evaluation metrics and plots are written to the directory specified by `EXPER_PATH` in `settings.py`.

### Export to ONNX
```
python save_to_onnx.py \
    --checkpoint $EXPER_PATH/superpoint/best.pth \
    --output superpoint.onnx \
    --opset 17
```
Set `--opset` according to the target runtime’s requirements. ONNX exports can be quantized using external toolchains such as `onnxruntime-tools` or `openvino`.

### Docker build and run
```
# Build the Docker image
sudo docker build -t supersegmentation:latest .

# Run training inside a container with GPU access
sudo docker run --gpus all -v $PWD:/workspace supersegmentation:latest \
    python Train_model_heatmap.py --config configs/magicpoint_kitti_train.yaml
```
Using a volume mount (`-v`) ensures that checkpoints written in the container persist on the host system.

## Production Considerations

Beyond raw performance, production systems must handle monitoring, updates, and security.

### Monitoring and logging
Instrument inference services with lightweight logging that records latency and error metrics. Tools such as Prometheus and Grafana visualize trends, enabling proactive scaling or rollback.

### Continuous integration and delivery
Automate testing and deployment using CI/CD pipelines. A typical workflow triggers unit tests on each pull request, builds Docker images for tagged releases, and pushes them to a registry. Deployment scripts then pull the tagged image to staging or production servers.

### Security
Keep dependencies up to date with `pip list --outdated`. When distributing Docker images, minimize the attack surface by using `distroless` or `scratch` base images and removing build tools after compilation. Use environment variables for secrets rather than hard‑coding them in configuration files.

## Key Takeaways

- **Reproducible environments** rely on pinning dependencies and using virtual environments or Docker containers.
- **Development and production workflows differ** in their focus on experimentation versus stability. Automate tests and artifact tracking to bridge the gap.
- **Mixed precision** accelerates training and inference by leveraging Tensor Cores and reducing memory consumption, with minimal code changes.
- **Quantization** shrinks model size and improves CPU inference speed; static quantization benefits convolutional networks, while dynamic quantization is simple for linear layers.
- **Pruning** removes redundant parameters and can further reduce latency when combined with structured approaches and fine‑tuning.
- **Sample commands** provided throughout this guide demonstrate how to train, evaluate, export, and containerize models.

By following these guidelines, practitioners can configure robust environments and deploy SuperSegmentaion models efficiently across a range of hardware platforms.
