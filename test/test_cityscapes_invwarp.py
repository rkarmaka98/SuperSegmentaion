import numpy as np
import torch
import cv2
from datasets.Cityscapes import Cityscapes
from utils.utils import inv_warp_image


def _create_fake_cityscapes(root):
    """Create a minimal Cityscapes-like structure with one image."""
    img_dir = root / "leftImg8bit" / "train" / "sample"
    img_dir.mkdir(parents=True, exist_ok=True)
    pattern = np.arange(64, dtype=np.uint8).reshape(8, 8)
    img = np.stack([
        np.tile(pattern, (8, 1)),
        np.tile(pattern, (8, 1)),
        np.tile(pattern, (8, 1)),
    ], axis=2)
    cv2.imwrite(str(img_dir / "000000_leftImg8bit.png"), img)


def test_inv_warp_identity(tmp_path):
    fake_root = tmp_path / "Cityscapes"
    _create_fake_cityscapes(fake_root)
    ds = Cityscapes(
        task="train",
        root=str(fake_root),
        load_segmentation=False,
        preprocessing={"resize": [32, 32]},
    )
    sample = ds[0]
    img = sample["image"][0]
    K = sample["K"]
    K_inv = sample["K_inv"]

    R = Cityscapes._euler_to_matrix(0.0, 0.0, 0.0)
    H = torch.from_numpy(K.numpy() @ R @ K_inv.numpy())
    inv_H = torch.inverse(H)

    warped = inv_warp_image(img, inv_H)
    assert torch.allclose(warped, img, atol=1e-6)
