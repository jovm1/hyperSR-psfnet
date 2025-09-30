from pathlib import Path
import torch
from torchvision.utils import save_image

def save_kernel_grid(k: torch.Tensor, out_path: Path, n: int = 8):
    """Save first n kernels as images for quick inspection."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    k_disp = k[:n].to(dtype=torch.float32)  # (n, K, K)

    # Normalize each kernel to [0,1] for display
    mn = k_disp.amin(dim=(1, 2), keepdim=True)
    k_disp = k_disp - mn
    mx = k_disp.amax(dim=(1, 2), keepdim=True)
    k_disp = k_disp / (mx + 1e-8)

    # make into 1-channel images
    k_img = k_disp.unsqueeze(1).contiguous()  # (n,1,K,K)
    save_image(k_img, fp=str(out_path), nrow=min(n, 8))