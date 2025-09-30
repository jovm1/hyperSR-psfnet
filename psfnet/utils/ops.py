import torch
import torch.nn.functional as F

# def apply_psf_rgb(img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
#     """
#     Convolve an RGB image (B,3,H,W) with a *single* 2D kernel (k,k) per sample.
#     We apply the same kernel to each channel independently.
#     """
#     B, C, H, W = img.shape
#     assert C == 3, "Expected RGB input"
#     k = kernel.shape[-1]
#     pad = k // 2

#     # (B,1,k,k) -> repeat to (B,C,k,k) -> flatten to (B*C,1,k,k)
#     kB = kernel.reshape(B, 1, k, k).repeat(1, C, 1, 1)
#     kBC = kB.reshape(B * C, 1, k, k).contiguous()

#     # (B, C, H, W) -> (1, B*C, H, W)
#     x = img.reshape(1, B * C, H, W)
#     x = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode='reflect')

#     y = torch.nn.functional.conv2d(x, weight=kBC, bias=None, stride=1, padding=0, groups=B * C)
#     y = y.reshape(B, C, H, W)
#     return y

def apply_psf_rgb(x, k):
    # x: [B,3,H,W], k: [B,1,ks,ks] or [B,ks,ks]
    if k.dim() == 3: k = k.unsqueeze(1)
    k = k / (k.sum(dim=(2,3), keepdim=True) + 1e-12)
    k = k.repeat(1, 3, 1, 1)
    padding = k.shape[-1] // 2
    return F.conv2d(x, k, padding=padding, groups=3)