import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import os


class CGNDataset(Dataset):
    """Point-cloud + grasp-label dataset produced by ``data/generate_data.py``.

    Searches *data_dir* and all immediate subdirectories for ``.npz`` files
    so both flat (``data/out/*.npz``) and per-category layouts
    (``data/out/Mug/*.npz``, ``data/out/Bowl/*.npz``, …) work.
    """

    LABEL_KEYS = ("confidence", "approach_dirs", "base_dirs", "widths")

    def __init__(self, data_dir: str, num_points: int = 20000):
        self.num_points = num_points
        self.files = sorted(
            glob.glob(os.path.join(data_dir, "*.npz"))
            + glob.glob(os.path.join(data_dir, "*", "*.npz"))
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data = np.load(self.files[idx])
        points = data["points"]
        labels = {k: data[k] for k in self.LABEL_KEYS}

        n = len(points)
        if n > self.num_points:
            choice = np.random.choice(n, self.num_points, replace=False)
            points = points[choice]
            labels = {k: v[choice] for k, v in labels.items()}

        out = {"points": torch.from_numpy(points).float()}
        for k, v in labels.items():
            out[k] = torch.from_numpy(v).float()
        return out
