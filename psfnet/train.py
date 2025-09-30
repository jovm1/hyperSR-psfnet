from pathlib import Path
from torch import get_device
import torch
from psfnet.data.datasets import ImageFolderKernelDataset
from psfnet.losses import center_of_mass_loss, reconstruction_loss
from psfnet.models.psfnet import KernelNet
from psfnet.utils.seed import set_seed
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

from psfnet.utils.viz import save_kernel_grid
import os, time, math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data.datasets import ImageFolderKernelDataset          # <- your dataset
from .models.psfnet import KernelNet                         # <- your model
from .losses import reconstruction_loss, center_of_mass_loss # <- your losses
from .metrics import kernel_mse, com_error_px                # <- (optional) basic metrics
from .utils.viz import save_kernel_grid                      # <- to save PNGs
from .utils.seed import set_seed

def make_loaders(cfg):
    train_ds = ImageFolderKernelDataset(
        data_dir=cfg["data"]["dir"],
        patch_size=cfg["data"]["patch_size"],
        kernel_size=cfg["data"]["kernel_size"],
        # add your blur/noise params here if your dataset uses them
    )
    # (Optional) you can build a val_ds similarly; or let the dataset split internally.
    val_ds = None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 4),
        pin_memory=True if get_device().type in ("cuda",) else False,
        drop_last=True,
    )
    val_loader = None if val_ds is None else DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 4),
        pin_memory=True if get_device().type in ("cuda",) else False,
        drop_last=False,
    )
    return train_loader, val_loader

def one_epoch(model, loader, optim, device, lambdas):
    model.train()
    avg_loss = 0.0
    n = 0
    for sharp, blurred, kgt, _ in loader:       # match your dataset’s return
        sharp = sharp.to(device)
        blurred = blurred.to(device)
        kgt = kgt.to(device)

        khat = model(sharp, blurred)

        loss = 0.0
        # recon loss
        loss_rec = reconstruction_loss(sharp, blurred, khat)
        loss = loss + lambdas["lambda_recon"] * loss_rec
        # center-of-mass loss (no GT needed—pull-to-center)
        loss_com = center_of_mass_loss(khat)
        loss = loss + lambdas["lambda_center"] * loss_com

        optim.zero_grad()
        loss.backward()
        optim.step()

        bs = sharp.size(0)
        avg_loss += loss.item() * bs
        n += bs
    return avg_loss / max(1, n)

@torch.no_grad()
def validate(model, loader, device):
    if loader is None:
        return {}
    model.eval()
    tot_kmse, tot_com, n = 0.0, 0.0, 0
    for sharp, blurred, kgt, _ in loader:
        sharp = sharp.to(device)
        blurred = blurred.to(device)
        kgt = kgt.to(device)
        khat = model(sharp, blurred)
        tot_kmse += kernel_mse(khat, kgt).item() * sharp.size(0)
        tot_com  += com_error_px(khat, kgt).item() * sharp.size(0)
        n += sharp.size(0)
    return {
        "val/kernel_mse": tot_kmse / max(1, n),
        "val/com_px":     tot_com  / max(1, n),
    }

def save_some_visuals(model, batch, out_dir, epoch, device):
    model.eval()
    sharp, blurred, kgt, _ = batch
    sharp = sharp.to(device)[:1]     # first sample only
    blurred = blurred.to(device)[:1]
    kgt = kgt.to(device)[:1]
    khat = model(sharp, blurred)[0,0].detach().cpu().numpy()
    kgt_ = kgt[0,0].detach().cpu().numpy()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    save_kernel_grid(khat, kgt_, os.path.join(out_dir, f"epoch_{epoch:03d}.png"))

def run_train(cfg):
    """
    Main training entrypoint used by CLI. 
    Expects a dict loaded from YAML: cfg['data'], cfg['train'], cfg['loss'], cfg['log'].
    """
    # 1) Repro + device
    set_seed(cfg["train"].get("seed", 42))
    device = get_device()

    # 2) Output dir
    stamp = time.strftime("%Y-%m-%d-psf")
    out_dir = os.path.join(cfg["log"]["out_dir"], stamp)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 3) Data
    train_loader, val_loader = make_loaders(cfg)

    # 4) Model & Optim
    ks = cfg["data"]["kernel_size"]
    model = KernelNet(kernel_size=ks).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    # 5) Train loop
    best_val = math.inf
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        loss_tr = one_epoch(
            model, train_loader, optim, device, cfg["loss"]
        )

        # (optional) quick val
        val_stats = validate(model, val_loader, device)
        kmse = val_stats.get("val/kernel_mse", float("nan"))

        # 6) Save visuals every N epochs
        if (epoch % cfg["log"].get("save_every", 1)) == 0:
            # take the first batch again (cheap: grab a new iterator)
            b = next(iter(train_loader))
            save_some_visuals(model, b, os.path.join(out_dir, "kernels"), epoch, device)

        # 7) Checkpoint (best by val kernel MSE if available, else by train loss)
        score = kmse if not math.isnan(kmse) else loss_tr
        if score < best_val:
            best_val = score
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "cfg": cfg},
                os.path.join(out_dir, "best.pt"),
            )

        # 8) Simple console log (replace with CSV/JSON if you prefer)
        if math.isnan(kmse):
            print(f"Epoch {epoch:03d}  train_loss={loss_tr:.6f}")
        else:
            print(f"Epoch {epoch:03d}  train_loss={loss_tr:.6f}  val_kMSE={kmse:.6f}")

    print(f"Done. Best score={best_val:.6f}. Artifacts in: {out_dir}")