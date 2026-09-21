import sys
sys.path.insert(0, "/work/METEOR")
import torch
from bevlane.model import DepthSegIPMNetV52, enable_depth_slim

model = DepthSegIPMNetV52(n_seg=21).cuda()
enable_depth_slim(model, widths=(128, 128, 96, 64))
model.eval()

shapes = []
def make_hook(name):
    def hook(m, inp, outp):
        in_s = [list(x.shape) if isinstance(x, torch.Tensor) else str(type(x).__name__) for x in inp]
        if isinstance(outp, torch.Tensor):
            out_s = list(outp.shape)
        elif isinstance(outp, (tuple, list)):
            out_s = [list(x.shape) if isinstance(x, torch.Tensor) else str(type(x).__name__) for x in outp]
        else:
            out_s = str(type(outp).__name__)
        shapes.append((name, type(m).__name__, in_s, out_s))
    return hook

for n, m in model.named_modules():
    if n in [
        "stem", "layer1", "layer2", "layer3", "layer4",
        "lat1", "lat2", "lat3", "lat4", "fuse",
        "depth_head", "seg_head", "ctx",
        "lidar_stem", "sdmap_stem", "tl_stem",
        "dec", "det_stem", "hm_head", "reg_head",
        "det2d_stem", "hm2d_head", "reg2d_head",
        "occ_stem", "occ_head",
        "ego_stem", "ego_mlp", "ego_attn",
        "traj_stem", "traj_head",
        "stat_head", "risk_head", "unk_dense", "pl_head",
        "refiner"
    ]:
        m.register_forward_hook(make_hook(n))

B, N = 1, 8
imgs = torch.zeros(B, N, 3, 432, 768, device="cuda")
K = torch.eye(3, device="cuda").unsqueeze(0).repeat(B*N, 1, 1).view(B, N, 3, 3)
Tc = torch.eye(4, device="cuda").unsqueeze(0).repeat(B*N, 1, 1).view(B, N, 4, 4)
lidar = torch.zeros(B, N, 108, 192, device="cuda")
lidar_bev = torch.zeros(B, 4, 400, 250, device="cuda")

with torch.no_grad():
    out = model(imgs, K, Tc, lidar=lidar, lidar_bev=lidar_bev)

print(f"{'Module Name':<15} | {'Module Type':<18} | {'Input Shape':<35} | {'Output Shape'}")
print("-" * 110)
for name, mtype, in_s, out_s in shapes:
    print(f"{name:<15} | {mtype:<18} | {str(in_s):<35} | {str(out_s)}")
