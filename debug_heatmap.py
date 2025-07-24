import torch, yaml, cv2
from pathlib import Path
from Val_model_heatmap import Val_model_heatmap  

cfg = yaml.safe_load(open("configs/magicpoint_cityscape_export.yaml"))
agent = Val_model_heatmap(cfg['model'], device='cuda:0')
agent.loadModel()

# img = cv2.imread("datasets/Cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png",
#                  cv2.IMREAD_GRAYSCALE) / 255.
# img = cv2.resize(img, (cfg['data']['preprocessing']['resize'][1],
#                        cfg['data']['preprocessing']['resize'][0]))
# img = torch.tensor(img, dtype=torch.float32,device='cuda:0').unsqueeze(0).unsqueeze(0)

# heatmap = agent.run(img)[0]          # Val_model_heatmap.run returns np(B,1,H,W) :contentReference[oaicite:0]{index=0}
# print("max heat:", heatmap.max())     # should be > 0
img = torch.rand(1, 1, 240, 320)     # shape [B, C, H, W] (values in [0,1])
img=torch.tensor(img,dtype=torch.float32,device='cuda:0')
with torch.no_grad():
    outs  = agent(img)
    semi  = outs['semi']             # the heat-map logits
print(semi.shape, semi.min().item(), semi.max().item())
