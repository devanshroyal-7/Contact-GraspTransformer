from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from models.types import SampleBatch


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


def _load_budget(budget_path: str) -> tuple[dict, list[str] | None]:
    with open(budget_path) as f:
        budget = json.load(f)
    cats = budget.get("categories")
    return budget, cats


def resolve_train_cap(budget_path: str,
                      override: Optional[int] = None,
                      preset: Optional[str] = None) -> int:
    """Return the ``train_objects_per_category`` cap to use.

    Precedence: explicit override > named preset arg > ``active_preset`` in JSON.
    """
    budget, _ = _load_budget(budget_path)
    if override is not None:
        return int(override)
    name = preset or budget.get("active_preset")
    if name is None or name not in budget.get("presets", {}):
        raise ValueError(
            f"training budget file {budget_path} has no valid active_preset "
            f"and no override was given (requested preset={preset!r})"
        )
    return int(budget["presets"][name]["train_objects_per_category"])


@dataclass(frozen=True)
class DatasetConfig:
    """Paths and budget policy resolved once; passed into ``CGNDataset``."""

    data_dir: str
    manifest_path: str
    budget_path: str
    num_points: int = 4096
    val_fraction: float = 0.2
    seed: int = 42
    train_objects_per_category: Optional[int] = None
    budget_preset: Optional[str] = None
    categories: Optional[tuple[str, ...]] = None

    @classmethod
    def from_paths(
        cls,
        data_dir: str,
        manifest_path: str,
        budget_path: str,
        *,
        num_points: int = 4096,
        val_fraction: float = 0.2,
        seed: int = 42,
        train_objects_per_category: Optional[int] = None,
        budget_preset: Optional[str] = None,
        categories: Optional[Iterable[str]] = None,
    ) -> "DatasetConfig":
        cats = tuple(categories) if categories is not None else None
        return cls(
            data_dir=data_dir,
            manifest_path=manifest_path,
            budget_path=budget_path,
            num_points=num_points,
            val_fraction=val_fraction,
            seed=seed,
            train_objects_per_category=train_objects_per_category,
            budget_preset=budget_preset,
            categories=cats,
        )


class CGNDataset(Dataset):
    """Point-cloud + grasp-label dataset driven by ``manifest.json``.

    File discovery follows the layout produced by ``data/generate_data.py``:

        <data_dir>/<split>/<category>/<mesh_hash>/NNN.npz

    The manifest (``manifest.json``) assigns every mesh a ``split``
    ("train" or "test") and, for train meshes, a ``rank`` in 1..N_TRAIN.
    The training budget JSON caps how many ranks per category are used
    for the "train" / "val" splits; "test" always uses all test meshes.

    Within the "train" split, a fraction of the *views* is held out as
    "val" (same objects, different renders) so training loss can be
    monitored.  "test" uses completely unseen meshes.
    """

    LABEL_KEYS = ("confidence", "approach_dirs", "base_dirs", "widths")

    def __init__(
        self,
        config: DatasetConfig,
        *,
        split: str = "train",
        augment: Optional[bool] = None,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")

        self.config = config
        self.num_points = config.num_points
        self.split = split
        self.augment = augment if augment is not None else (split == "train")

        with open(config.manifest_path) as f:
            manifest: list[dict] = json.load(f)

        _, cfg_cats = _load_budget(config.budget_path)
        cat_filter = set(config.categories) if config.categories is not None else (
            set(cfg_cats) if cfg_cats else None
        )

        if split == "test":
            objs = [m for m in manifest if m.get("split") == "test"]
        else:
            k = resolve_train_cap(
                config.budget_path,
                override=config.train_objects_per_category,
                preset=config.budget_preset,
            )
            self.train_cap = k
            objs = [m for m in manifest
                    if m.get("split") == "train" and int(m.get("rank", 0)) <= k]

        if cat_filter is not None:
            objs = [m for m in objs if m["category"] in cat_filter]

        disk_split = "test" if split == "test" else "train"
        all_files: list[str] = []
        for m in objs:
            mesh_hash = m.get("mesh_hash") or os.path.splitext(
                os.path.basename(m["mesh_path"]))[0]
            pattern = os.path.join(config.data_dir, disk_split, m["category"],
                                   mesh_hash, "*.npz")
            all_files.extend(sorted(glob.glob(pattern)))

        if not all_files:
            expected = os.path.join(
                config.data_dir, "<split>", "<category>", "<mesh_hash>", "*.npz"
            )
            raise FileNotFoundError(
                f"No .npz samples found for split={split!r} under "
                f"{config.data_dir!r}. Expected nested layout: {expected}. "
                f"Regenerate with data/generate_data.py (legacy flat "
                f"<data_dir>/<category>/NNN.npz is no longer supported)."
            )

        self.objects = objs

        if split == "test":
            self.files = all_files
            return

        # View-level train/val split within the selected training meshes.
        rng = np.random.RandomState(config.seed)
        indices = rng.permutation(len(all_files))
        n_val = max(1, int(len(all_files) * config.val_fraction))
        if split == "val":
            self.files = [all_files[i] for i in indices[:n_val]]
        else:
            # split == "train" (test returned earlier)
            self.files = [all_files[i] for i in indices[n_val:]]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> SampleBatch:
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

        return SampleBatch(
            points=torch.from_numpy(points).float(),
            confidence=torch.from_numpy(labels["confidence"]).float(),
            approach_dirs=torch.from_numpy(labels["approach_dirs"]).float(),
            base_dirs=torch.from_numpy(labels["base_dirs"]).float(),
            widths=torch.from_numpy(labels["widths"]).float(),
        )
