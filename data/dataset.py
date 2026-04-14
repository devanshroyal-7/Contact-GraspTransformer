from __future__ import annotations

import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import os
from typing import Optional


def random_rotation_matrix() -> np.ndarray:
    """Sample a uniform random SO(3) rotation matrix."""
    q = np.random.randn(4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float32)


class CGNDataset(Dataset):
    """Point-cloud + grasp-label dataset produced by ``data/generate_data.py``.

    Searches *data_dir* and all immediate subdirectories for ``.npz`` files
    so both flat (``data/out/*.npz``) and per-category layouts
    (``data/out/Mug/*.npz``, ``data/out/Bowl/*.npz``, ...) work.
    """

    LABEL_KEYS = ("confidence", "approach_dirs", "base_dirs", "widths")

    def __init__(self, data_dir: str, num_points: int = 4096,
                 split: str = "train", val_fraction: float = 0.2,
                 augment: Optional[bool] = None, seed: int = 42):
        self.num_points = num_points
        self.augment = augment if augment is not None else (split == "train")

        all_files = sorted(
            glob.glob(os.path.join(data_dir, "*.npz"))
            + glob.glob(os.path.join(data_dir, "*", "*.npz"))
        )

        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(all_files))
        n_val = max(1, int(len(all_files) * val_fraction))

        if split == "val":
            self.files = [all_files[i] for i in indices[:n_val]]
        else:
            self.files = [all_files[i] for i in indices[n_val:]]

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

        if self.augment:
            R = random_rotation_matrix()
            points = points @ R.T
            labels["approach_dirs"] = labels["approach_dirs"] @ R.T
            labels["base_dirs"] = labels["base_dirs"] @ R.T
            points = points + np.random.randn(*points.shape).astype(np.float32) * 0.001

        out = {"points": torch.from_numpy(points).float()}
        for k, v in labels.items():
            out[k] = torch.from_numpy(v).float()
        return out
