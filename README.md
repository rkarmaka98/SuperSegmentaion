# pytorch-superpoint

This is a PyTorch implementation of  "SuperPoint: Self-Supervised Interest Point Detection and Description." Daniel DeTone, Tomasz Malisiewicz, Andrew Rabinovich. [ArXiv 2018](https://arxiv.org/abs/1712.07629).
This code is partially based on the tensorflow implementation
https://github.com/rpautrat/SuperPoint.

Please be generous to star this repo if it helps your research.
This repo is a bi-product of our paper [deepFEPE(IROS 2020)](https://github.com/eric-yyjau/pytorch-deepFEPE.git).

## Differences between our implementation and original paper
- *Descriptor loss*: We tested descriptor loss using different methods, including dense method (as paper but slightly different) and sparse method. We notice sparse loss can converge more efficiently with similar performance. The default setting here is sparse method.

## Results on HPatches
| Task                                      | Homography estimation |      |      | Detector metric |      | Descriptor metric |                |
|-------------------------------------------|-----------------------|------|------|-----------------|------|-------------------|----------------|
|                                           | Epsilon = 1           | 3    | 5    | Repeatability   | MLE  | NN mAP            | Matching Score |
| Pretrained model                        | 0.44                  | 0.77 | 0.83 | 0.606           | 1.14 | 0.81              | 0.55           |
| Sift (subpixel accuracy)                  | 0.63                  | 0.76 | 0.79 | 0.51            | 1.16 | 0.70               | 0.27            |
| superpoint_coco_heat2_0_170k_hpatches_sub | 0.46                  | 0.75 | 0.81 | 0.63            | 1.07 | 0.78              | 0.42           |
| superpoint_kitti_heat2_0_50k_hpatches_sub | 0.44                  | 0.71 | 0.77 | 0.56            | 0.95 | 0.78              | 0.41           |

- Pretrained model is from [SuperPointPretrainedNetwork](https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork).
- The evaluation is done under our evaluation scripts.
- COCO/ KITTI pretrained model is included in this repo.


## Installation
### Requirements
- python == 3.6
- pytorch >= 1.1 (tested in 1.3.1)
- torchvision >= 0.3.0 (tested in 0.4.2)
- cuda (tested in cuda10)

```
conda create --name py36-sp python=3.6
conda activate py36-sp
pip install -r requirements.txt
pip install -r requirements_torch.txt # install pytorch
```

### Path setting
- paths for datasets ($DATA_DIR), logs are set in `setting.py`

### Dataset
Datasets should be downloaded into $DATA_DIR. The Synthetic Shapes dataset will also be generated there. The folder structure should look like:

```
datasets/ ($DATA_DIR)
|-- COCO
|   |-- train2017
|   |   |-- file1.jpg
|   |   `-- ...
|   `-- val2017
|       |-- file1.jpg
|       `-- ...
|   |-- panoptic_train2017
|   |-- panoptic_val2017
|   |-- panoptic_val2017.json
|   |-- panoptic_train2017.json
`-- HPatches
|   |-- i_ajuntament
|   `-- ...
`-- synthetic_shapes  # will be automatically created
```
- MS-COCO 2017
    - [MS-COCO 2017 link](http://cocodataset.org/#download)
    - For panoptic training download the panoptic annotations and place the corresponding folders `panoptic_train2017/` and `panoptic_val2017/` with their JSON files under `COCO/annotations`.
    - Use `utils/coco_panoptic_to_cs34.py` to convert panoptic PNGs to Cityscapes-34 labels when training semantic segmentation with 34 classes. The script writes grayscale masks which can be placed under `panoptic_cs34_train2017/` and `panoptic_cs34_val2017/`.
      Example:

      ```bash
      python utils/coco_panoptic_to_cs34.py \
          --src panoptic_val2017 \
          --ann annotations/panoptic_val2017.json \
          --categories annotations/panoptic_coco_categories.json \
          --dst panoptic_cs34_val2017
      ```
    - Set `load_panoptic: true` and `use_cs34_masks: true` in your configuration to train with these grayscale masks instead of the standard panoptic labels. When both options are enabled, the JSON annotations are not required.
    - For evaluation, generate a category file matching the Cityscapes‑34 IDs:

      ```bash
      python utils/create_cs34_categories_json.py \
          --dst annotations/panoptic_cs34_categories.json
      ```

      Pass `--category-file annotations/panoptic_cs34_categories.json` to
      `evaluation.py` so class names and colors are correct when visualizing
      predictions.
- HPatches
    - [HPatches link](http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz)

### Cityscapes
- [Cityscapes link](https://www.cityscapes-dataset.com/downloads/)
- After extracting, place the dataset under `$DATA_DIR/Cityscapes` so that the
  directory structure is:

```
datasets/ ($DATA_DIR)
`-- Cityscapes
    |-- leftImg8bit
    |   |-- train
    |   |-- val
    |   `-- test
    `-- gtFine
        |-- train
        |-- val
        `-- test
```
- If the dataset is stored elsewhere, update `DATA_PATH` in `settings.py`.

Example commands using the Cityscapes configs:

```bash
# training
python train4.py train_joint configs/superpoint_cityscapes_finetune.yaml superpoint_cityscapes --eval

# export predictions (including segmentation masks)
python export.py export_descriptor configs/superpoint_cityscapes_export.yaml cityscapes_export --export-segmentation

# evaluate exported segmentation
python evaluation.py logs/cityscapes_export/predictions --evaluate-segmentation
```

To visualize the 4-category mapping, pass `--category-file utils/cs4_categories.json`
to `evaluation.py`.

Both configs load a checkpoint via their `pretrained` option. Update this path
to point at the model you wish to fine-tune or evaluate.

When `warped_pair.enable` is set to true in `superpoint_cityscapes_export.yaml`,
the Cityscapes loader also returns `warped_image` and its corresponding
`homography` so descriptor metrics can be exported like on COCO or HPatches.

Predicted segmentation masks can also be exported by adding
`--export-segmentation` to other export commands and evaluated using
`evaluation.py --evaluate-segmentation`.

The segmentation head uses a lightweight ASPP decoder which provides
better context while keeping the model small enough for mid-range GPUs.


## run the code
- Notes:
    - Start from any steps (1-4) by downloading some intermediate results
    - Training usually takes 8-10 hours on one 'NVIDIA 2080Ti'.
    - Currently Support training on 'COCO' dataset (original paper), 'KITTI' dataset.
- Tensorboard:
    - log files is saved under 'runs/<\export_task>/...'
    
`tensorboard --logdir=./runs/ [--host | static_ip_address] [--port | 6008]`

### 1) Training MagicPoint on Synthetic Shapes
```
python train4.py train_base configs/magicpoint_shapes_pair.yaml magicpoint_synth --eval
```
you don't need to download synthetic data. You will generate it when first running it.
Synthetic data is exported in `./datasets`. You can change the setting in `settings.py`.

### 2) Exporting detections on MS-COCO / kitti
This is the step of homography adaptation(HA) to export pseudo ground truth for joint training.
- make sure the pretrained model in config file is correct
- make sure COCO dataset is in '$DATA_DIR' (defined in setting.py)
<!-- - you can export hpatches or coco dataset by editing the 'task' in config file -->
- config file:
```
export_folder: <'train' | 'val'>  # set export for training or validation
```
#### General command:
```
python export.py <export task>  <config file>  <export folder> [--outputImg | output images for visualization (space inefficient)]
```
#### export coco - do on training set 
```
python export.py export_detector_homoAdapt configs/magicpoint_coco_export.yaml magicpoint_synth_homoAdapt_coco
```
#### export coco - do on validation set 
- Edit 'export_folder' to 'val' in 'magicpoint_coco_export.yaml'
```
python export.py export_detector_homoAdapt configs/magicpoint_coco_export.yaml magicpoint_synth_homoAdapt_coco
```


### 3) Training Superpoint on MS-COCO/ KITTI
You need pseudo ground truth labels to traing detectors. Labels can be exported from step 2) or downloaded from [link](https://drive.google.com/drive/folders/1nnn0UbNMFF45nov90PJNnubDyinm2f26?usp=sharing). Then, as usual, you need to set config file before training.
- config file
  - root: specify your labels root
  - root_split_txt: where you put the train.txt/ val.txt split files (no need for COCO, needed for KITTI)
  - labels: the exported labels from homography adaptation
  - pretrained: specify the pretrained model (you can train from scratch)
- 'eval': turn on the evaluation during training 

#### General command
```
python train4.py <train task> <config file> <export folder> --eval
```

#### COCO
```
python train4.py train_joint configs/superpoint_coco_train_heatmap.yaml superpoint_coco --eval --debug
```
To train using COCO panoptic labels, set `dataset: 'coco_panoptic'` and `load_panoptic: true` in the configuration. The COCO folder must contain `panoptic_train2017/`, `panoptic_val2017/` and their corresponding JSON files under `annotations/`.
#### kitti
```
python train4.py train_joint configs/superpoint_kitti_train_heatmap.yaml superpoint_kitti --eval --debug
```

- set your batch size (originally 1)
- refer to: 'train_tutorial.md'

### 4) Export/ Evaluate the metrics on HPatches
- Use pretrained model or specify your model in config file
- ```./run_export.sh``` will run export then evaluation.

#### Export
- download HPatches dataset (link above). Put in the $DATA_DIR.
```python export.py <export task> <config file> <export folder>```
- Export keypoints, descriptors, matching
```
python export.py export_descriptor  configs/magicpoint_repeatability_heatmap.yaml superpoint_hpatches_test
```
To also export predicted segmentation masks, append `--export-segmentation`:
```
python export.py export_descriptor <config> <folder> --export-segmentation
```
#### evaluate
```python evaluation.py <path to npz files> [-r, --repeatibility | -o, --outputImg | -homo, --homography ]```
- Evaluate homography estimation/ repeatability/ matching scores ...
```
python evaluation.py logs/superpoint_hpatches_test/predictions --repeatibility --outputImg --homography --plotMatching
```
Segmentation quality can be evaluated with:
```
python evaluation.py <output_path> --evaluate-segmentation
```
Passing `--outputImg` saves colored predictions and ground truth masks under
`<output_path>/segmentation`.

### 5) Export/ Evaluate repeatability on SIFT
- Refer to another project: [Feature-preserving image denoising with multiresolution filters](https://github.com/eric-yyjau/image_denoising_matching)
```shell
# export detection, description, matching
python export_classical.py export_descriptor configs/classical_descriptors.yaml sift_test --correspondence

# evaluate (use 'sift' flag)
python evaluation.py logs/sift_test/predictions --sift --repeatibility --homography 
```


- specify the pretrained model

## Pretrained models
### Current best model
- *COCO dataset*
```logs/superpoint_coco_heat2_0/checkpoints/superPointNet_170000_checkpoint.pth.tar```
- *KITTI dataset*
```logs/superpoint_kitti_heat2_0/checkpoints/superPointNet_50000_checkpoint.pth.tar```
### model from magicleap
```pretrained/superpoint_v1.pth```

## Jupyter notebook 
```shell
# show images saved in the folders
jupyter notebook
notebooks/visualize_hpatches.ipynb 
```

## Updates (year.month.day)
- 2020.08.05: 
  - Update pytorch nms from (https://github.com/eric-yyjau/pytorch-superpoint/pull/19)
  - Update and test KITTI dataloader and labels on google drive (should be able to fit the KITTI raw format)
  - Update and test SIFT evaluate at step 5.

## Known problems
- ~~test step 5: evaluate on SIFT~~
- Export COCO dataset in low resolution (240x320) instead of high resolution (480x640).
- Due to step 1 was done long time ago. We are still testing it again along with step 2-4. Please refer to our pretrained model or exported labels. Or let us know how the whole pipeline works.
- Warnings from tensorboard.

## Work in progress
- Release notebooks with unit testing.
- Dataset: ApolloScape/ TUM.

## Citations
Please cite the original paper.
```
@inproceedings{detone2018superpoint,
  title={Superpoint: Self-supervised interest point detection and description},
  author={DeTone, Daniel and Malisiewicz, Tomasz and Rabinovich, Andrew},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  pages={224--236},
  year={2018}
}
```

Please also cite our DeepFEPE paper.
```
@misc{2020_jau_zhu_deepFEPE,
Author = {You-Yi Jau and Rui Zhu and Hao Su and Manmohan Chandraker},
Title = {Deep Keypoint-Based Camera Pose Estimation with Geometric Constraints},
Year = {2020},
Eprint = {arXiv:2007.15122},
}
```

# Credits
This implementation is developed by [You-Yi Jau](https://github.com/eric-yyjau) and [Rui Zhu](https://github.com/Jerrypiglet). Please contact You-Yi for any problems. 
Again the work is based on Tensorflow implementation by [Rémi Pautrat](https://github.com/rpautrat) and [Paul-Edouard Sarlin](https://github.com/Skydes) and official [SuperPointPretrainedNetwork](https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork).
Thanks to Daniel DeTone for help during the implementation.

## Posts
[What have I learned from the implementation of deep learning paper?](https://medium.com/@eric.yyjau/what-have-i-learned-from-the-implementation-of-deep-learning-paper-365ee3253a89)
