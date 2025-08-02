import torch
import torch.nn as nn
from pathlib import Path
from models.SuperPointNet_gauss2 import SuperPointNet_gauss2

core = SuperPointNet_gauss2(seg_refine=True).eval()

# Optional: load weights (skip this if you only want the naked graph)
CKPT_PATH = Path("logs/superpoint_cityscapes/checkpoints/superPointNet_50000_checkpoint.pth.tar")  # change or set to None
if CKPT_PATH and CKPT_PATH.is_file():
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    core.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    print(f"Weights loaded from {CKPT_PATH}")

# ────────────────────────────────────────────────────────────────────
# 2. Thin wrapper → tuple output (ONNX exporter requirement)
# ────────────────────────────────────────────────────────────────────
class SuperPointONNX(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, x):
        out = self.net(x)
        return out["semi"], out["desc"], out["segmentation"]

model = SuperPointONNX(core).eval()

# ────────────────────────────────────────────────────────────────────
# 3. Export
# ────────────────────────────────────────────────────────────────────
dummy  = torch.randn(1, 1, 240, 320)  # (batch, channel, H, W)
onnx_file = "superpoint_gauss2.onnx"

torch.onnx.export(
    model,
    dummy,
    onnx_file,
    input_names=["image"],
    output_names=["semi", "desc", "seg"],
    dynamic_axes={"image": {0: "batch"},
                  "semi":  {0: "batch"},
                  "desc":  {0: "batch"},
                  "seg":   {0: "batch"}},
    opset_version=17,
    do_constant_folding=True,
)

print(f"ONNX model saved to {Path(onnx_file).resolve()}")