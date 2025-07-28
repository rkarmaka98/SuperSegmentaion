"""
This script exports detection/ description using pretrained model.

Author: You-Yi Jau, Rui Zhu
Date: 2019/12/12
"""

## basic
import argparse
import time
import csv
import yaml
import os
import logging
from pathlib import Path

import numpy as np
from imageio import imread
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

## torch
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

## other functions
from utils.utils import (
    tensor2array,
    save_checkpoint,
    load_checkpoint,
    save_path_formatter,
)
from utils.utils import getWriterPath, flattenDetection
from utils.loader import dataLoader, modelLoader, pretrainedLoader
from utils.utils import inv_warp_image_batch
from models.model_wrap import SuperPointFrontend_torch, PointTracker

## parameters
from settings import EXPER_PATH

#### util functions


def combine_heatmap(heatmap, inv_homographies, mask_2D, device="cpu"):
    ## multiply heatmap with mask_2D
    heatmap = heatmap * mask_2D

    heatmap = inv_warp_image_batch(
        heatmap, inv_homographies[0, :, :, :], device=device, mode="bilinear"
    )

    ##### check
    mask_2D = inv_warp_image_batch(
        mask_2D, inv_homographies[0, :, :, :], device=device, mode="bilinear"
    )
    heatmap = torch.sum(heatmap, dim=0)
    mask_2D = torch.sum(mask_2D, dim=0)
    return heatmap / mask_2D
    pass


#### end util functions

def export_descriptor(config, output_dir, args):
    """
    # input 2 images, output keypoints and correspondence
    save prediction:
        pred:
            'image': np(320,240)
            'prob' (keypoints): np (N1, 2)
            'desc': np (N2, 256)
            'warped_image': np(320,240)
            'warped_prob' (keypoints): np (N2, 2)
            'warped_desc': np (N2, 256)
            'homography': np (3,3)
            'matches': np [N3, 4]
    """
    from utils.loader import get_save_path
    from utils.var_dim import squeezeToNumpy

    # basic settings
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("train on device: %s", device)
    with open(os.path.join(output_dir, "config.yml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    writer = SummaryWriter(getWriterPath(task=args.command, date=True))
    save_path = get_save_path(output_dir)
    save_output = save_path / "../predictions"
    os.makedirs(save_output, exist_ok=True)

    ## parameters
    outputMatches = True
    subpixel = config["model"]["subpixel"]["enable"]
    patch_size = config["model"]["subpixel"]["patch_size"]

    # data loading
    from utils.loader import dataLoader_test as dataLoader
    task = config["data"]["dataset"]
    data = dataLoader(config, dataset=task)
    test_set, test_loader = data["test_set"], data["test_loader"]
    from utils.print_tool import datasize
    datasize(test_loader, config, tag="test")

    # model loading
    from utils.loader import get_module
    Val_model_heatmap = get_module("", config["front_end_model"])
    ## load pretrained
    val_agent = Val_model_heatmap(config["model"], device=device)
    val_agent.loadModel()

    ## tracker
    tracker = PointTracker(max_length=2, nn_thresh=val_agent.nn_thresh)

    ###### check!!!
    count = 0
    for i, sample in tqdm(enumerate(test_loader)):
        # HPatches samples already contain warped pairs while some datasets like
        # Cityscapes might omit them. In the latter case use the original image
        # and an identity homography.
        img_0 = sample["image"]
        img_1 = sample.get("warped_image", img_0)
        has_warp = "warped_image" in sample

        # first image, no matches
        # img = img_0
        def get_pts_desc_from_agent(val_agent, img, device="cpu"):
            """
            pts: list [numpy (3, N)]
            desc: list [numpy (256, N)]
            """
            heatmap_batch = val_agent.run(
                img.to(device)
            )  # heatmap: numpy [batch, 1, H, W]
            # heatmap to pts
            pts = val_agent.heatmap_to_pts()
            # print("pts: ", pts)
            if subpixel:
                pts = val_agent.soft_argmax_points(pts, patch_size=patch_size)
            # heatmap, pts to desc
            desc_sparse = val_agent.desc_to_sparseDesc()
            # print("pts[0]: ", pts[0].shape, ", desc_sparse[0]: ", desc_sparse[0].shape)
            # print("pts[0]: ", pts[0].shape)
            outs = {"pts": pts[0], "desc": desc_sparse[0]}
            return outs

        def transpose_np_dict(outs):
            for entry in list(outs):
                outs[entry] = outs[entry].transpose()

        outs = get_pts_desc_from_agent(val_agent, img_0, device=device)
        pts, desc = outs["pts"], outs["desc"]  # pts: np [3, N]

        # optional segmentation prediction
        if args.export_segmentation and "segmentation" in val_agent.outs:
            seg_pred = val_agent.outs["segmentation"].argmax(dim=1)
            pred_mask = seg_pred.cpu().numpy().squeeze()
        else:
            pred_mask = None

        if outputMatches and has_warp:
            tracker.update(pts, desc)

        # save keypoints
        pred = {"image": squeezeToNumpy(img_0)}
        pred.update({"prob": pts.transpose(), "desc": desc.transpose()})
        if pred_mask is not None:
            pred["pred_mask"] = pred_mask
            if "segmentation_mask" in sample:
                pred["gt_mask"] = squeezeToNumpy(sample["segmentation_mask"])
            elif "mask" in sample:
                pred["gt_mask"] = squeezeToNumpy(sample["mask"])

        # second image, output matches
        outs = get_pts_desc_from_agent(val_agent, img_1, device=device)
        pts, desc = outs["pts"], outs["desc"]

        if outputMatches and has_warp:
            tracker.update(pts, desc)

        pred.update({"warped_image": squeezeToNumpy(img_1)})
        # If the sample provides a warped pair, also store the corresponding
        # keypoints, descriptors and homography
        if has_warp:
            pred.update(
                {
                    "warped_prob": pts.transpose(),
                    "warped_desc": desc.transpose(),
                }
            )
            if "homography" in sample:
                pred["homography"] = squeezeToNumpy(sample["homography"])

        if "segmentation_mask" in sample:
            mask_np = squeezeToNumpy(sample["segmentation_mask"])
            pred.update({"segmentation_mask": mask_np, "gt_mask": mask_np})
        elif "mask" in sample:
            mask_np = squeezeToNumpy(sample["mask"])
            pred.update({"segmentation_mask": mask_np, "gt_mask": mask_np})


        if outputMatches and has_warp:
            matches = tracker.get_matches()
            print("matches: ", matches.transpose().shape)
            pred.update({"matches": matches.transpose()})
        print("pts: ", pts.shape, ", desc: ", desc.shape)

        # clean last descriptor
        tracker.clear_desc()

        filename = str(count)
        path = Path(save_output, "{}.npz".format(filename))
        np.savez_compressed(path, **pred)
        # print("save: ", path)
        count += 1
    print("output pairs: ", count)
    # close tensorboard writer to avoid duplicate plugin errors
    writer.close()


@torch.no_grad()
def export_detector_homoAdapt_gpu(config, output_dir, args):
    """
    input 1 images, output pseudo ground truth by homography adaptation.
    Save labels:
        pred:
            'prob' (keypoints): np (N1, 3)
    """
    from utils.utils import pltImshow
    from utils.utils import saveImg
    from utils.draw import draw_keypoints
    from evaluation import overlay_mask, compute_miou

    # basic setting
    task = config["data"]["dataset"]
    export_task = config["data"]["export_folder"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logging.info("train on device: %s", device)
    with open(os.path.join(output_dir, "config.yml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    writer = SummaryWriter(
        getWriterPath(task=args.command, exper_name=args.exper_name, date=True)
    )

    ## parameters
    nms_dist = config["model"]["nms"]  # 4
    top_k = config["model"]["top_k"]
    homoAdapt_iter = config["data"]["homography_adaptation"]["num"]
    conf_thresh = config["model"]["detection_threshold"]
    nn_thresh = 0.7
    outputMatches = True
    count = 0
    max_length = 5
    output_images = args.outputImg
    check_exist = True

    ## save data
    save_path = Path(output_dir)
    save_output = save_path
    save_output = save_output / "predictions" / export_task
    save_path = save_path / "checkpoints"
    logging.info("=> will save everything to {}".format(save_path))
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_output, exist_ok=True)

    # data loading
    from utils.loader import dataLoader_test as dataLoader

    data = dataLoader(config, dataset=task, export_task=export_task)
    test_set, test_loader = data["test_set"], data["test_loader"]

    # model loading
    ## load pretrained
    try:
        path = config["pretrained"]
        print("==> Loading pre-trained network.")
        print("path: ", path)
        # This class runs the SuperPoint network and processes its outputs.

        fe = SuperPointFrontend_torch(
            config=config,
            weights_path=path,
            nms_dist=nms_dist,
            conf_thresh=conf_thresh,
            nn_thresh=nn_thresh,
            cuda=False,
            device=device,
        )
        print("==> Successfully loaded pre-trained network.")

        fe.net_parallel()
        print(path)
        # save to files
        save_file = save_output / "export.txt"
        with open(save_file, "a") as myfile:
            myfile.write("load model: " + path + "\n")
    except Exception:
        print(f"load model: {path} failed! ")
        raise

    def load_as_float(path):
        return imread(path).astype(np.float32) / 255

    tracker = PointTracker(max_length, nn_thresh=fe.nn_thresh)
    with open(save_file, "a") as myfile:
        myfile.write("homography adaptation: " + str(homoAdapt_iter) + "\n")

    ## loop through all images
    for i, sample in tqdm(enumerate(test_loader)):
        
        if sample is None:
            print(f"[WARN] Skipping None batch at index {i}")
            continue

        if "image" not in sample or "valid_mask" not in sample:
            print(f"[WARN] Skipping invalid sample at index {i}")
            continue
        img, mask_2D = sample["image"], sample["valid_mask"]
        img = img.transpose(0, 1)
        img_2D = sample["image_2D"].numpy().squeeze()
        mask_2D = mask_2D.transpose(0, 1)

        inv_homographies, homographies = (
            sample["homographies"],
            sample["inv_homographies"],
        )
        img, mask_2D, homographies, inv_homographies = (
            img.to(device),
            mask_2D.to(device),
            homographies.to(device),
            inv_homographies.to(device),
        )
        # sample = test_set[i]
        name = sample["name"][0]
        logging.info(f"name: {name}")
        if check_exist:
            p = Path(save_output, "{}.npz".format(name))
            if p.exists():
                logging.info("file %s exists. skip the sample.", name)
                continue

        # pass through network
        with torch.no_grad():
            outs_all = fe.net(img)
        semi = outs_all["semi"]
        heatmap = flattenDetection(semi, tensor=True)
        if args.export_segmentation and "segmentation" in outs_all:
            seg_pred = outs_all["segmentation"].argmax(dim=1)
            pred_mask = seg_pred.cpu().numpy().squeeze()
        else:
            pred_mask = None
        outputs = combine_heatmap(heatmap, inv_homographies, mask_2D, device=device)
        pts = fe.getPtsFromHeatmap(outputs.detach().cpu().squeeze())  # (x,y, prob)

        # subpixel prediction
        if config["model"]["subpixel"]["enable"]:
            fe.heatmap = outputs  # tensor [batch, 1, H, W]
            print("outputs: ", outputs.shape)
            print("pts: ", pts.shape)
            pts = fe.soft_argmax_points([pts])
            pts = pts[0]

        ## top K points
        pts = pts.transpose()
        print("total points: ", pts.shape)
        print("pts: ", pts[:5])
        if top_k:
            if pts.shape[0] > top_k:
                pts = pts[:top_k, :]
                print("topK filter: ", pts.shape)

        ## save keypoints
        pred = {}
        pred.update({"pts": pts})
        if pred_mask is not None:
            pred["pred_mask"] = pred_mask
            if "segmentation_mask" in sample:
                pred["gt_mask"] = np.squeeze(sample["segmentation_mask"])
            elif "mask" in sample:
                pred["gt_mask"] = np.squeeze(sample["mask"])
        if "segmentation_mask" in sample:
            pred.update({"segmentation_mask": np.squeeze(sample["segmentation_mask"])})
        elif "mask" in sample:
            pred.update({"segmentation_mask": np.squeeze(sample["mask"])})

        # ----- TensorBoard visualization -----
        # log image with predicted keypoints overlay
        img_pts = draw_keypoints(img_2D * 255, pts.transpose())
        writer.add_image(
            "keypoints", torch.from_numpy(img_pts).permute(2, 0, 1) / 255.0, count
        )

        # log segmentation overlay and compute mIoU when available
        if pred_mask is not None:
            overlay = overlay_mask(
                img_2D, pred_mask,
                num_classes=config["data"].get("num_segmentation_classes")
            )
            writer.add_image(
                "seg_overlay", torch.from_numpy(overlay).permute(2, 0, 1) / 255.0, count
            )
            if "gt_mask" in pred:
                miou = compute_miou(pred_mask, pred["gt_mask"],
                                   num_classes=config["data"].get("num_segmentation_classes"))
                writer.add_scalar("mIoU", miou, count)
        # number of detected points per image
        writer.add_scalar("num_points", pts.shape[0], count)

        ## - make directories
        filename = str(name)
        # only KITTI-like datasets provide a scene_name for grouping outputs
        if task in ("Kitti", "Kitti_inh"):
            scene_name = sample["scene_name"][0]
            os.makedirs(Path(save_output, scene_name), exist_ok=True)

        path = Path(save_output, "{}.npz".format(filename))
        np.savez_compressed(path, **pred)

        ## output images for visualization labels
        if output_images:
            img_pts = draw_keypoints(img_2D * 255, pts.transpose())
            f = save_output / (str(count) + ".png")
            if task == "Coco" or "Kitti":
                f = save_output / (name + ".png")
            saveImg(img_pts, str(f))
        count += 1

    print("output pseudo ground truth: ", count)
    save_file = save_output / "export.txt"
    with open(save_file, "a") as myfile:
        myfile.write("Homography adaptation: " + str(homoAdapt_iter) + "\n")
        myfile.write("output pairs: " + str(count) + "\n")
    # close tensorboard writer
    writer.close()
    pass


if __name__ == "__main__":
    # global var
    torch.set_default_tensor_type(torch.FloatTensor)
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # add parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # export command
    p_train = subparsers.add_parser("export_descriptor")
    p_train.add_argument("config", type=str)
    p_train.add_argument("exper_name", type=str)
    p_train.add_argument("--correspondence", action="store_true")
    p_train.add_argument("--eval", action="store_true")
    p_train.add_argument(
        "--debug", action="store_true", default=False, help="turn on debuging mode"
    )
    p_train.add_argument(
        "--export-segmentation",
        action="store_true",
        default=False,
        help="save predicted segmentation masks",
    )
    p_train.set_defaults(func=export_descriptor)

    # using homography adaptation to export detection psuedo ground truth
    p_train = subparsers.add_parser("export_detector_homoAdapt")
    p_train.add_argument("config", type=str)
    p_train.add_argument("exper_name", type=str)
    p_train.add_argument("--eval", action="store_true")
    p_train.add_argument(
        "--outputImg", action="store_true", help="output image for visualization"
    )
    p_train.add_argument(
        "--debug", action="store_true", default=False, help="turn on debuging mode"
    )
    p_train.add_argument(
        "--export-segmentation",
        action="store_true",
        default=False,
        help="save predicted segmentation masks",
    )
    # p_train.set_defaults(func=export_detector_homoAdapt)
    p_train.set_defaults(func=export_detector_homoAdapt_gpu)

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    print("check config!! ", config)

    output_dir = os.path.join(EXPER_PATH, args.exper_name)
    os.makedirs(output_dir, exist_ok=True)

    # with capture_outputs(os.path.join(output_dir, 'log')):
    logging.info("Running command {}".format(args.command.upper()))
    args.func(config, output_dir, args)
