# PyTorch SuperPoint with Segmentation

This repository provides a PyTorch implementation of
[SuperPoint: Self-Supervised Interest Point Detection and Description](https://arxiv.org/abs/1712.07629)
along with utilities for semantic segmentation and the DeepFEPE project.
The code builds upon the TensorFlow version from
https://github.com/rpautrat/SuperPoint.

## Installation

- Python ≥ 3.10
- Install dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements_torch.txt  # PyTorch and related packages
```

Dataset and log paths are configured in `settings.py`.

## Project structure

```
SuperSegmentaion/
├── configs/      # training and export configuration files
├── datasets/     # dataset loaders and tools
├── docs/         # in-depth documentation
├── models/       # network definitions
├── utils/        # helper scripts
└── ...
```

## Dataset preparation

Datasets should reside under the directory specified by `DATA_PATH` in
`settings.py`. A typical layout is:

```
datasets/ ($DATA_PATH)
├── COCO
│   ├── train2017/
│   ├── val2017/
│   ├── panoptic_train2017/
│   └── panoptic_val2017/
├── HPatches/
├── Cityscapes
│   ├── leftImg8bit/{train,val,test}/
│   ├── gtFine/
│   └── gtFine_1024/{train,val,test}/
└── synthetic_shapes/  # generated automatically
```

- **COCO 2017** – Download the images and panoptic annotations. Use
  `utils/coco_panoptic_to_cs34.py` to generate Cityscapes-34 labels when
  training segmentation.
- **HPatches** – Download the sequence dataset from the
  [official site](http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz).
- **Cityscapes** – Extract the dataset under `$DATA_PATH/Cityscapes` and run
  `datasets/resize_cityscapes_masks.py` to create 512×1024 masks.

Utility scripts in `utils/` provide additional conversions and dataset clean-up.

## Pipeline

1. **Train MagicPoint on Synthetic Shapes**

   ```bash
   python train4.py train_base configs/magicpoint_shapes_pair.yaml magicpoint_synth --eval
   ```

2. **Export homography-adapted detections on COCO or KITTI**

   ```bash
   python export.py export_detector_homoAdapt configs/magicpoint_coco_export.yaml \
       magicpoint_synth_homoAdapt_coco
   ```

3. **Train SuperPoint on COCO or KITTI**

   ```bash
   python train4.py train_joint configs/superpoint_coco_train_heatmap.yaml superpoint_coco --eval --debug
   python train4.py train_joint configs/superpoint_kitti_train_heatmap.yaml superpoint_kitti --eval --debug
   ```

4. **Export and evaluate on HPatches**

   ```bash
   python export.py export_descriptor configs/magicpoint_repeatability_heatmap.yaml superpoint_hpatches_test
   python evaluation.py logs/superpoint_hpatches_test/predictions --repeatibility --outputImg --homography --plotMatching
   ```

   Segmentation metrics can be produced with:

   ```bash
   python evaluation.py <output_path> --evaluate-segmentation
   ```

5. **(Optional) Benchmark classical descriptors such as SIFT**

   ```bash
   python export_classical.py export_descriptor configs/classical_descriptors.yaml sift_test --correspondence
   python evaluation.py logs/sift_test/predictions --sift --repeatibility --homography
   ```

## Pretrained models

- COCO: `logs/superpoint_coco_heat2_0/checkpoints/superPointNet_170000_checkpoint.pth.tar`
- KITTI: `logs/superpoint_kitti_heat2_0/checkpoints/superPointNet_50000_checkpoint.pth.tar`
- MagicLeap: `pretrained/superpoint_v1.pth`

## Documentation

In-depth guides on the model architecture, configuration options, and
evaluation pipeline are available in the [docs](docs/) directory and the
project overview [docs.md](docs.md).
Key references include `architecture.md`, `configuration.md`, `pipeline.md`,
and `benchmarks.md`.

## Results and benchmarks

Evaluation metrics are collated in `summary.csv`. An example Cityscapes export
reports the following scores:

| Task              | Homography@3 | Homography@5 | Repeatability | MLE  | NN mAP | Matching Score | Segmentation IOU |
|-------------------|--------------|--------------|---------------|------|--------|----------------|------------------|
| cityscapes_export | 0.9997       | 1.0          | 0.8334        | 1.75 | 0.9988 | 0.99998        | 0.8427           |

See [docs/benchmarks.md](docs/benchmarks.md) for detailed benchmarking
workflows and sample result tables.

## Citation

Please cite the original SuperPoint paper and our DeepFEPE paper:

```
@inproceedings{detone2018superpoint,
  title     = {SuperPoint: Self-supervised interest point detection and description},
  author    = {DeTone, Daniel and Malisiewicz, Tomasz and Rabinovich, Andrew},
  booktitle = {CVPR Workshops},
  pages     = {224--236},
  year      = {2018}
}

@misc{2020_jau_zhu_deepFEPE,
  Author = {You-Yi Jau and Rui Zhu and Hao Su and Manmohan Chandraker},
  Title  = {Deep Keypoint-Based Camera Pose Estimation with Geometric Constraints},
  Year   = {2020},
  Eprint = {arXiv:2007.15122},
}
```

## Credits

This implementation was developed by
[You-Yi Jau](https://github.com/eric-yyjau) and
[Rui Zhu](https://github.com/Jerrypiglet) and is based on work by
[Rémi Pautrat](https://github.com/rpautrat),
[Paul-Edouard Sarlin](https://github.com/Skydes) and
[MagicLeap Research](https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork).

