import torch
import torch.nn as nn

class KernelNet(nn.Module):
    def __init__(self, in_ch: int = 6, hidden: int = 64, kernel_size: int = 15):
        super().__init__()
        self.k = kernel_size
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(hidden, kernel_size * kernel_size)
        self.softplus = nn.Softplus()  # ensure positivity before normalization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,6,H,W)
        h = self.features(x)
        h = self.pool(h).flatten(1)
        k_logits = self.head(h)
        k_pos = self.softplus(k_logits) + 1e-8
        # normalize to sum=1 per sample
        k_norm = k_pos / k_pos.sum(dim=1, keepdim=True)
        # reshape to (B,K,K)
        B = x.size(0)
        k = k_norm.view(B, self.k, self.k)
        return k


