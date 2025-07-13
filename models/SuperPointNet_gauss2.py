"""latest version of SuperpointNet. Use it!

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
from models.unet_parts import *
import numpy as np

# from models.SubpixelNet import SubpixelNet

class ASPP(nn.Module):
    """Atrous spatial pyramid pooling used in the segmentation head.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels produced by each convolutional branch.
    dilations : Sequence[int]
        Dilation rates for the atrous convolutions. A global average pooling
        branch is always added in addition to these rates.
    """

    def __init__(self, in_ch, out_ch, dilations=(1, 6, 12, 18)):
        super().__init__()
        # parallel atrous convolutions with different dilation rates
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=d, dilation=d, bias=False),
                # GroupNorm is robust to small batch sizes
                nn.GroupNorm(32, out_ch),
                nn.ReLU(inplace=True),
            )
            for d in dilations
        ])

        # global image-level features
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(32, out_ch),
            nn.ReLU(inplace=True),
        )

        conv_in = out_ch * (len(dilations) + 1)
        # merge all branches and project to ``out_ch`` channels
        self.project = nn.Sequential(
            nn.Conv2d(conv_in, out_ch, 1, bias=False),
            nn.GroupNorm(32, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        res = [blk(x) for blk in self.blocks]
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=size, mode="bilinear", align_corners=False)
        res.append(gp)
        x = torch.cat(res, dim=1)
        return self.project(x)

class SuperPointNet_gauss2(torch.nn.Module):
    """Pytorch definition of the SuperPoint feature and segmentation network."""

    def __init__(self, subpixel_channel=1, num_classes=1, input_channels=1):
        """Initialize network layers.

        Parameters
        ----------
        subpixel_channel : int, optional
            Number of channels for the optional subpixel head (unused here).
        num_classes : int, optional
            Number of classes predicted by the segmentation head.
        input_channels : int, optional
            Channel count of the input images. Defaults to ``1`` for grayscale
            inputs.
        """

        super(SuperPointNet_gauss2, self).__init__()
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        det_h = 65
        self.inc = inconv(1, c1)
        self.down1 = down(c1, c2)
        self.down2 = down(c2, c3)
        self.down3 = down(c3, c4)
        # self.down4 = down(c4, 512)
        # self.up1 = up(c4+c3, c2)
        # self.up2 = up(c2+c2, c1)
        # self.up3 = up(c1+c1, c1)
        # self.outc = outconv(c1, subpixel_channel)
        self.relu = torch.nn.ReLU(inplace=True)
        # self.outc = outconv(64, n_classes)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnPa = nn.BatchNorm2d(c5)
        self.convPb = torch.nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0)
        self.bnPb = nn.BatchNorm2d(det_h)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnDa = nn.BatchNorm2d(c5)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
        self.bnDb = nn.BatchNorm2d(d1)
        # Segmentation Head with ASPP for richer context
        self.seg_aspp = ASPP(c4, 128)
        self.seg_head = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

        self.output = None



    def forward(self, x):
        """Compute keypoints, descriptors and segmentation logits.

        Parameters
        ----------
        x : ``Tensor``
            Input image tensor of shape ``(N, C, H, W)`` where ``C`` equals
            ``input_channels``.

        Returns
        -------
        dict
            Dictionary containing ``semi`` (detector logits), ``desc`` (L2
            normalized descriptors) and ``segmentation`` logits.
        """

        # Shared encoder producing a feature map at 1/8 resolution
        x1 = self.inc(x)      # -> [B, 64, H, W]
        x2 = self.down1(x1)   # -> [B, 64, H/2, W/2]
        x3 = self.down2(x2)   # -> [B, 128, H/4, W/4]
        x4 = self.down3(x3)   # -> [B, 128, H/8, W/8]

        # Detector head predicts keypoint scores
        cPa = self.relu(self.bnPa(self.convPa(x4)))  # -> [B, 256, H/8, W/8]
        semi = self.bnPb(self.convPb(cPa))           # -> [B, 65, H/8, W/8]

        # Descriptor head outputs raw descriptors
        cDa = self.relu(self.bnDa(self.convDa(x4)))  # -> [B, 256, H/8, W/8]
        desc = self.bnDb(self.convDb(cDa))           # -> [B, 256, H/8, W/8]

        # Segmentation head produces per-pixel class logits
        seg = self.seg_aspp(x4)                      # -> [B, 128, H/8, W/8]
        seg_logits = self.seg_head(seg)              # -> [B, num_classes, H/8, W/8]

        dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
        output = {'semi': semi, 'desc': desc, 'segmentation': seg_logits}
        self.output = output

        return output

    def process_output(self, sp_processer):
        """Post-process network output to obtain keypoint coordinates.

        Parameters
        ----------
        sp_processer : ``SuperPointNet_process``
            Helper object providing non-maximum suppression and descriptor
            sampling utilities.

        Returns
        -------
        dict
            ``self.output`` dictionary augmented with ``pts_int`` (integer
            coordinates), ``pts_offset`` (subpixel offsets) and ``pts_desc``
            (corresponding descriptors).
        """

        from utils.utils import flattenDetection

        output = self.output
        semi = output['semi']
        desc = output['desc']

        # convert detection logits to heatmap: [B, 1, H, W]
        heatmap = flattenDetection(semi)
        # apply NMS to get sparse keypoint map
        heatmap_nms_batch = sp_processer.heatmap_to_nms(heatmap, tensor=True)

        # compute subpixel offsets from heatmap
        outs = sp_processer.pred_soft_argmax(heatmap_nms_batch, heatmap)
        residual = outs['pred']              # [B, N, 2]

        # sample descriptors at keypoint locations
        outs = sp_processer.batch_extract_features(desc, heatmap_nms_batch, residual)  # adds 'pts_desc'

        # output.update({'heatmap': heatmap, 'heatmap_nms': heatmap_nms, 'descriptors': descriptors})
        output.update(outs)
        self.output = output
        return output


def get_matches(deses_SP):
    """Return mutual nearest neighbour matches between two descriptor sets.

    Parameters
    ----------
    deses_SP : list[Tensor]
        List containing two descriptor tensors of shape ``(N, D)``.

    Returns
    -------
    ndarray
        Boolean mask indicating mutual matches between the two sets.
    """

    from models.model_wrap import PointTracker

    tracker = PointTracker(max_length=2, nn_thresh=1.2)
    f = lambda x: x.cpu().detach().numpy()
    matching_mask = tracker.nn_match_two_way(f(deses_SP[0]).T, f(deses_SP[1]).T, nn_thresh=1.2)
    return matching_mask

    # print("matching_mask: ", matching_mask.shape)
    # f_mask = lambda pts, maks: pts[]
    # pts_m = []
    # pts_m_res = []
    # for i in range(2):
    #     idx = xs_SP[i][matching_mask[i, :].astype(int), :]
    #     res = reses_SP[i][matching_mask[i, :].astype(int), :]
    #     print("idx: ", idx.shape)
    #     print("res: ", idx.shape)
    #     pts_m.append(idx)
    #     pts_m_res.append(res)
    #     pass

    # pts_m = torch.cat((pts_m[0], pts_m[1]), dim=1)
    # matches_test = toNumpy(pts_m)
    # print("pts_m: ", pts_m.shape)

    # pts_m_res = torch.cat((pts_m_res[0], pts_m_res[1]), dim=1)
    # # pts_m_res = toNumpy(pts_m_res)
    # print("pts_m_res: ", pts_m_res.shape)
    # # print("pts_m_res: ", pts_m_res)
        
    # pts_idx_res = torch.cat((pts_m, pts_m_res), dim=1)
    # print("pts_idx_res: ", pts_idx_res.shape)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SuperPointNet_gauss2()
    model = model.to(device)


    # check keras-like model summary using torchsummary
    from torchsummary import summary
    summary(model, input_size=(1, 240, 320))

    ## test
    image = torch.zeros((2,1,120, 160))
    outs = model(image.to(device))
    print("outs: ", list(outs))

    from utils.print_tool import print_dict_attr
    print_dict_attr(outs, 'shape')

    from models.model_utils import SuperPointNet_process 
    params = {
        'out_num_points': 500,
        'patch_size': 5,
        'device': device,
        'nms_dist': 4,
        'conf_thresh': 0.015
    }

    sp_processer = SuperPointNet_process(**params)
    outs = model.process_output(sp_processer)
    print("outs: ", list(outs))
    print_dict_attr(outs, 'shape')

    # timer
    import time
    from tqdm import tqdm
    iter_max = 50

    start = time.time()
    print("Start timer!")
    for i in tqdm(range(iter_max)):
        outs = model(image.to(device))
    end = time.time()
    print("forward only: ", iter_max/(end - start), " iter/s")

    start = time.time()
    print("Start timer!")
    xs_SP, deses_SP, reses_SP = [], [], []
    for i in tqdm(range(iter_max)):
        outs = model(image.to(device))
        outs = model.process_output(sp_processer)
        xs_SP.append(outs['pts_int'].squeeze())
        deses_SP.append(outs['pts_desc'].squeeze())
        reses_SP.append(outs['pts_offset'].squeeze())
    end = time.time()
    print("forward + process output: ", iter_max/(end - start), " iter/s")

    start = time.time()
    print("Start timer!")
    for i in tqdm(range(len(xs_SP))):
        get_matches([deses_SP[i][0], deses_SP[i][1]])
    end = time.time()
    print("nn matches: ", iter_max/(end - start), " iters/s")


if __name__ == '__main__':
    main()



