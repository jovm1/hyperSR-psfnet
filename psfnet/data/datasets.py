#RGBPairDataset, HSIRGBPairDataset
from ast import Tuple
import random
from tkinter import Image
from click import Path
import torch.nn.functional as F
from matplotlib import transforms
from netCDF4 import Dataset
import torch

from psfnet.data.kernels import sample_random_kernel
from psfnet.utils.ops import apply_psf_rgb


class ImageFolderKernelDataset(Dataset):
    def __init__(self,
                 img_dir: str | Path,
                 kernel_size: int = 15,
                 patch_size: int = 128,
                 min_side: int = 256,
                 extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')):
        self.img_paths = [p for p in Path(img_dir).glob('**/*') if p.suffix.lower() in extensions]
        if not self.img_paths:
            raise FileNotFoundError(f"No images found in {img_dir}")
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.min_side = min_side
        self.to_tensor = transforms.ToTensor()
        self.aug_geo = transforms.RandomHorizontalFlip(p=0.5)
        self.aug_geo_v = transforms.RandomVerticalFlip(p=0.5)

    def __len__(self):
        return len(self.img_paths)

    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert('RGB')
        # Resize the short side to at least min_side to allow crops
        w, h = img.size
        scale = max(1.0, self.min_side / min(w, h))
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        if scale > 1.0:
            img = img.resize((new_w, new_h), resample=Image.BICUBIC)
        t = self.to_tensor(img)  # [3,H,W] in [0,1]
        return t

    def __getitem__(self, idx: int):
        path = self.img_paths[idx % len(self.img_paths)]
        sharp = self._load_image(path)
        # Random geom augments *before* cropping
        sharp = self.aug_geo(sharp)
        sharp = self.aug_geo_v(sharp)

        _, H, W = sharp.shape
        ph = self.patch_size
        if H < ph or W < ph:
            # pad reflect
            pad_h = max(0, ph - H)
            pad_w = max(0, ph - W)
            sharp = F.pad(sharp.unsqueeze(0), (0, pad_w, 0, pad_h), mode='reflect').squeeze(0)
            _, H, W = sharp.shape
        # random crop
        top = random.randint(0, H - ph)
        left = random.randint(0, W - ph)
        sharp_patch = sharp[:, top:top+ph, left:left+ph]

        # sample kernel and blur
        k_np = sample_random_kernel(self.kernel_size)
        k = torch.from_numpy(k_np)
        k = k / (k.sum() + 1e-8)
        blurred_patch = apply_psf_rgb(sharp_patch.unsqueeze(0), k.unsqueeze(0)).squeeze(0)

        # stack inputs: [sharp, blurred] -> 6 channels
        x = torch.cat([sharp_patch, blurred_patch], dim=0)
        return x, k