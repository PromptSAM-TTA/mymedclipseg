import torch
from utils.main_utils import load_cfg_from_cfg_file
from trainers import build_medclipseg_unimedclip

cfg = load_cfg_from_cfg_file("configs/DUAL100_336.yaml")
model = build_medclipseg_unimedclip(cfg)
device = torch.device("cuda")
model = model.to(device)
model.eval()

image = torch.randn(2, 3, 336, 336).to(device)
text = ["tooth 11", "tooth 12"]
with torch.no_grad():
    out = model(image, text, num_samples=1)
print("output shape:", tuple(out.shape))
assert out.shape[-2:] == (336, 336), f"expected 336x336, got {out.shape}"
print("SMOKE TEST PASSED: 336x336 input works")
