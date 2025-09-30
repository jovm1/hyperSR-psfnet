import torch
import torch.nn.functional as F

from psfnet.utils.ops import apply_psf_rgb


def center_of_mass_loss(k: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    """Penalize kernel mass drifting away from the center (stability prior)."""
    B, K, _ = k.shape
    device = k.device
    coords = torch.linspace(-(K - 1) / 2, (K - 1) / 2, K, device=device)
    X, Y = torch.meshgrid(coords, coords, indexing='xy')  # (K,K)
    X = X.unsqueeze(0).expand(B, -1, -1)
    Y = Y.unsqueeze(0).expand(B, -1, -1)
    mass = k.sum(dim=(1, 2), keepdim=True) + 1e-8
    x_cm = (k * X).sum(dim=(1, 2), keepdim=True) / mass
    y_cm = (k * Y).sum(dim=(1, 2), keepdim=True) / mass
    dist2 = x_cm.pow(2) + y_cm.pow(2)  # squared distance from center (0,0)
    return weight * dist2.mean()


def reconstruction_loss(sharp: torch.Tensor, blurred: torch.Tensor, k_pred: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    # sharp/blurred: (B,3,H,W), k_pred: (B,K,K)
    y_hat = apply_psf_rgb(sharp, k_pred)
    return weight * F.mse_loss(y_hat, blurred)