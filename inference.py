"""ContactGraspNet inference: point cloud -> grasp poses.

This script wraps a trained model so you can give it a point cloud of
*arbitrary size* (any N x 3 numpy array, in any frame) and get back a
ranked list of 6-DoF grasps in a form that any physics simulator or ROS
stack can consume directly.

Output format per grasp
-----------------------
Everything is returned in the **same frame as the input point cloud**
(the script handles mean-subtraction internally and undoes it before
returning). The canonical output is a batch of 4x4 SE(3) homogeneous
transforms following the standard Franka-Panda "panda_hand" convention:

    z-axis   = gripper approach direction
    x-axis   = baseline (the axis fingers open/close along)
    y-axis   = z x x  (right-handed)
    origin   = wrist keypoint (between the fingers, at the hand base)

This is the same convention used by the Contact-GraspNet paper, Isaac
Sim / Isaac Lab, MoveIt, and most grasp datasets. If your simulator
uses a different convention (e.g. z pointing out of a different face)
you only need to post-multiply by a constant offset transform.

The predictor returns a dict with numpy arrays:

    {
        "poses":       (K, 4, 4) float32  - SE(3) transforms
        "positions":   (K, 3)    float32  - wrist xyz (== poses[:, :3, 3])
        "quaternions": (K, 4)    float32  - xyzw (ROS / scipy convention)
        "widths":      (K,)      float32  - target gripper opening [m]
        "scores":      (K,)      float32  - confidence in [0, 1]
        "contacts":    (K, 3)    float32  - contact points on object
    }

Integration notes
-----------------
* PyBullet / Isaac Sim: feed `poses` straight in, or unpack
  `positions[i]` and `quaternions[i]` (xyzw).
* ROS 1/2 (MoveIt, moveit_py, etc.): build a
  `geometry_msgs/PoseStamped` from `positions[i]` / `quaternions[i]`
  (xyzw). Set `header.frame_id` to the point cloud's TF frame
  (e.g. `"camera_depth_optical_frame"`).
* Franka Panda: `poses[i]` can be used as the goal for the
  `panda_hand` (not `panda_link8`) frame directly.

Usage
-----
    # Programmatic (``train.py`` checkpoints: ``model_state_dict`` + ``config``)
    from inference import GraspPredictor
    predictor = GraspPredictor("checkpoints/ptv3/<run_folder>/best.pt")
    grasps = predictor.predict(points_np, top_k=50, score_thresh=0.5)

    # CLI (writes out/<run>.h5 (ACRONYM layout) + out/<run>.json with mesh info)
    python inference_cli.py --ckpt checkpoints/ptv3/<run_folder>/best.pt \\
        --points data/out/train/Camera/<hash>/001.npz --top-k 100
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional, TypedDict, Tuple

import numpy as np
import torch

from models.model import ContactGraspNet
from models.cgn_heads import wrist_from_grasp
from checkpoint_io import (
    infer_cpe_mode_from_state_dict,
    state_dict_and_config_from_checkpoint,
    torch_load_checkpoint,
)


# ───────────────────────────── config ────────────────────────────────────────

DEFAULT_NUM_POINTS = 4096   # must match the value used during training


# ────────────────────────── point cloud I/O ──────────────────────────────────

def load_point_cloud(path: str) -> np.ndarray:
    """Load a point cloud from .npy / .npz / .ply / .pcd / .xyz / .txt.

    Returns an (N, 3) float32 array.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        arr = np.load(path)
    elif ext == ".npz":
        data = np.load(path)
        key = "points" if "points" in data.files else data.files[0]
        arr = data[key]
    elif ext in (".ply", ".pcd"):
        try:
            import open3d as o3d
        except ImportError as e:
            raise ImportError(
                f"Loading {ext} files requires open3d. `pip install open3d`."
            ) from e
        pcd = o3d.io.read_point_cloud(path)
        arr = np.asarray(pcd.points)
    elif ext in (".xyz", ".txt"):
        arr = np.loadtxt(path)[:, :3]
    else:
        raise ValueError(f"Unsupported point cloud extension: {ext}")

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Expected (N, >=3) point cloud, got {arr.shape}")
    return arr[:, :3]


def sample_points(points: np.ndarray,
                  num_points: int,
                  rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Resample an arbitrary point cloud to exactly ``num_points`` rows.

    * Too many points  -> random subset without replacement.
    * Too few points   -> sample with replacement (simple duplicate padding).
    * Exactly right    -> random permutation of all rows (via sampling without
      replacement), returned as a new array (not an in-place shuffle).
    """
    rng = rng or np.random.default_rng()
    n = points.shape[0]
    if n == 0:
        raise ValueError("Input point cloud is empty.")
    if n >= num_points:
        idx = rng.choice(n, num_points, replace=False)
    else:
        # upsample by duplicating points; harmless for the transformer / PN2
        idx = rng.choice(n, num_points, replace=True)
    return points[idx]


# ────────────────────────── math helpers ─────────────────────────────────────

def rotation_to_quaternion_xyzw(R: np.ndarray) -> np.ndarray:
    """Batched rotation matrix -> unit quaternion (x, y, z, w).

    Vectorised variant of the standard "shepperd" / sign-safe method.
    Input:  (..., 3, 3)  Output: (..., 4)
    """
    m = np.asarray(R, dtype=np.float64)
    t = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]

    q = np.empty(m.shape[:-2] + (4,), dtype=np.float64)

    # Case 1: t > 0
    mask0 = t > 0
    s = np.sqrt(t[mask0] + 1.0) * 2
    q[mask0, 3] = 0.25 * s
    q[mask0, 0] = (m[mask0, 2, 1] - m[mask0, 1, 2]) / s
    q[mask0, 1] = (m[mask0, 0, 2] - m[mask0, 2, 0]) / s
    q[mask0, 2] = (m[mask0, 1, 0] - m[mask0, 0, 1]) / s

    # Case 2: diag(0) is largest
    mask1 = (~mask0) & (m[..., 0, 0] >= m[..., 1, 1]) & (m[..., 0, 0] >= m[..., 2, 2])
    s = np.sqrt(1.0 + m[mask1, 0, 0] - m[mask1, 1, 1] - m[mask1, 2, 2]) * 2
    q[mask1, 3] = (m[mask1, 2, 1] - m[mask1, 1, 2]) / s
    q[mask1, 0] = 0.25 * s
    q[mask1, 1] = (m[mask1, 0, 1] + m[mask1, 1, 0]) / s
    q[mask1, 2] = (m[mask1, 0, 2] + m[mask1, 2, 0]) / s

    # Case 3: diag(1) is largest
    mask2 = (~mask0) & (~mask1) & (m[..., 1, 1] >= m[..., 2, 2])
    s = np.sqrt(1.0 + m[mask2, 1, 1] - m[mask2, 0, 0] - m[mask2, 2, 2]) * 2
    q[mask2, 3] = (m[mask2, 0, 2] - m[mask2, 2, 0]) / s
    q[mask2, 0] = (m[mask2, 0, 1] + m[mask2, 1, 0]) / s
    q[mask2, 1] = 0.25 * s
    q[mask2, 2] = (m[mask2, 1, 2] + m[mask2, 2, 1]) / s

    # Case 4: diag(2) is largest
    mask3 = ~(mask0 | mask1 | mask2)
    s = np.sqrt(1.0 + m[mask3, 2, 2] - m[mask3, 0, 0] - m[mask3, 1, 1]) * 2
    q[mask3, 3] = (m[mask3, 1, 0] - m[mask3, 0, 1]) / s
    q[mask3, 0] = (m[mask3, 0, 2] + m[mask3, 2, 0]) / s
    q[mask3, 1] = (m[mask3, 1, 2] + m[mask3, 2, 1]) / s
    q[mask3, 2] = 0.25 * s

    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return q.astype(np.float32)


def build_poses(contacts: np.ndarray,
                approach: np.ndarray,
                baseline: np.ndarray,
                widths: np.ndarray) -> np.ndarray:
    """Assemble (K, 4, 4) grasp poses in the Panda hand convention.

    Axes: x = baseline, z = approach, y = z x x (right-handed).
    Origin = wrist keypoint (same formula as training-time loss).
    """
    x = baseline / (np.linalg.norm(baseline, axis=-1, keepdims=True) + 1e-8)
    z = approach / (np.linalg.norm(approach, axis=-1, keepdims=True) + 1e-8)
    # Re-orthogonalise x against z (the network's Gram-Schmidt already does
    # this, but a second pass is cheap insurance against fp drift).
    x = x - (x * z).sum(axis=-1, keepdims=True) * z
    x /= (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
    y = np.cross(z, x)

    K = contacts.shape[0]
    R = np.stack([x, y, z], axis=-1)  # (K, 3, 3), columns are axes

    t = wrist_from_grasp(contacts, z, x, widths)

    poses = np.zeros((K, 4, 4), dtype=np.float32)
    poses[:, :3, :3] = R
    poses[:, :3, 3] = t
    poses[:, 3, 3] = 1.0
    return poses


def nms(positions: np.ndarray, scores: np.ndarray,
        radius: float) -> np.ndarray:
    """Greedy non-maximum suppression by 3D position."""
    order = np.argsort(-scores)
    keep = []
    suppressed = np.zeros(len(scores), dtype=bool)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        d2 = ((positions - positions[i]) ** 2).sum(axis=-1)
        suppressed |= d2 < radius * radius
    return np.array(keep, dtype=np.int64)


def _reindex_grasps(
    idx: np.ndarray,
    poses: np.ndarray,
    positions: np.ndarray,
    quats: np.ndarray,
    widths: np.ndarray,
    scores: np.ndarray,
    contacts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply the same row index to every grasp array."""
    return (
        poses[idx],
        positions[idx],
        quats[idx],
        widths[idx],
        scores[idx],
        contacts[idx],
    )


# ─────────────────────────── predictor ───────────────────────────────────────

class GraspArrays(TypedDict):
    """Numpy arrays exported by ``GraspResult.as_dict``."""

    poses: np.ndarray
    positions: np.ndarray
    quaternions: np.ndarray
    widths: np.ndarray
    scores: np.ndarray
    contacts: np.ndarray


class MeshInfo(TypedDict, total=False):
    """Optional mesh metadata resolved from a training npz path / manifest."""

    category: Optional[str]
    mesh_hash: Optional[str]
    split: Optional[str]
    view: Optional[str]
    mesh_path: Optional[str]
    grasp_file: Optional[str]
    scale: Optional[float]


@dataclass
class GraspResult:
    poses: np.ndarray        # (K, 4, 4)
    positions: np.ndarray    # (K, 3)
    quaternions: np.ndarray  # (K, 4) xyzw
    widths: np.ndarray       # (K,)
    scores: np.ndarray       # (K,)
    contacts: np.ndarray     # (K, 3)

    def as_dict(self) -> GraspArrays:
        return {
            "poses": self.poses,
            "positions": self.positions,
            "quaternions": self.quaternions,
            "widths": self.widths,
            "scores": self.scores,
            "contacts": self.contacts,
        }


def prepare_model_input(points: np.ndarray,
                         num_points: int,
                         device: torch.device,
                         seed: Optional[int] = None) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {points.shape}")

    rng = np.random.default_rng(seed)
    sampled = sample_points(points.astype(np.float32), num_points, rng)

    # Mean-subtract: the model was trained on mean-centred clouds.
    centroid = sampled.mean(axis=0, keepdims=True).astype(np.float32)
    centred = sampled - centroid
    xyz = torch.from_numpy(centred).unsqueeze(0).to(device)  # (1, N, 3)
    return centroid, centred, xyz


class GraspPredictor:
    """Thin wrapper around ContactGraspNet for real-time / ROS use.

    Load once, call ``.predict(points)`` per frame.

    Checkpoints from ``train.py`` (``model_state_dict`` + ``config``) are
    recognised; backbone / ``cpe_mode`` / ``num_points`` are taken from
    ``config`` when present so the graph matches the weights.
    """

    def __init__(self,
                 ckpt_path: str,
                 backbone: str = "ptv3",
                 num_points: int = DEFAULT_NUM_POINTS,
                 device: Optional[str] = None,
                 cpe_mode: Optional[str] = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))

        raw = torch_load_checkpoint(ckpt_path, map_location=self.device)
        state, ckpt_cfg = state_dict_and_config_from_checkpoint(raw)

        eff_backbone = str(ckpt_cfg.get("backbone", backbone))
        self.num_points = int(ckpt_cfg.get("num_points", num_points))

        eff_cpe = None
        if eff_backbone == "ptv3":
            # Explicit constructor arg wins over checkpoint config; both win
            # over weight-shape heuristics. Prefer config['cpe_mode'] when
            # present — infer_cpe_mode_from_state_dict is only a fallback for
            # old checkpoints that lack an embedded train config.
            eff_cpe = cpe_mode or ckpt_cfg.get("cpe_mode")
            if eff_cpe is None:
                eff_cpe = infer_cpe_mode_from_state_dict(state) or "knn"
            eff_cpe = str(eff_cpe)

        self.model = ContactGraspNet(
            backbone=eff_backbone,
            cpe_mode=eff_cpe,
        ).to(self.device)

        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self,
                points: np.ndarray,
                score_thresh: float = 0.5,
                top_k: Optional[int] = None,
                nms_radius: float = 0.0,
                seed: Optional[int] = None) -> GraspResult:
        """Run inference on a single point cloud.

        Parameters
        ----------
        points : (N, 3) array in the sensor / world frame. N may be any
            size; the method resamples to the model's expected count.
        score_thresh : drop grasps with confidence < this value.
        top_k : if set, keep only the top-K grasps by confidence (after
            threshold + NMS).
        nms_radius : if > 0, apply greedy NMS on grasp positions with
            this radius (metres). Good default: 0.02-0.03.
        seed : optional int for reproducible sampling.

        Returns
        -------
        GraspResult with numpy arrays in the *input* frame.
        """
        centroid, centred, xyz = prepare_model_input(
            points,
            num_points=self.num_points,
            device=self.device,
            seed=seed,
        )
        preds = self.model(xyz)

        scores = preds["confidence"][0].cpu().numpy()                   # (N,)
        approach = preds["approach_dirs"][0].cpu().numpy()              # (N, 3)
        baseline = preds["base_dirs"][0].cpu().numpy()                  # (N, 3)
        widths = preds["widths"][0].cpu().numpy()                       # (N,)

        # Back to the original (input) frame.
        contacts = centred + centroid  # == sampled
        contacts = contacts.astype(np.float32)

        keep = scores >= score_thresh
        if not keep.any():
            return _empty_result()

        contacts = contacts[keep]
        approach = approach[keep]
        baseline = baseline[keep]
        widths = widths[keep]
        scores = scores[keep]

        poses = build_poses(contacts, approach, baseline, widths)
        positions = poses[:, :3, 3].copy()
        quats = rotation_to_quaternion_xyzw(poses[:, :3, :3])

        if nms_radius > 0.0 and len(positions) > 1:
            keep_idx = nms(positions, scores, nms_radius)
            poses, positions, quats, widths, scores, contacts = _reindex_grasps(
                keep_idx, poses, positions, quats, widths, scores, contacts
            )

        order = np.argsort(-scores)
        if top_k is not None:
            order = order[:top_k]
        poses, positions, quats, widths, scores, contacts = _reindex_grasps(
            order, poses, positions, quats, widths, scores, contacts
        )

        return GraspResult(
            poses=poses.astype(np.float32),
            positions=positions.astype(np.float32),
            quaternions=quats.astype(np.float32),
            widths=widths.astype(np.float32),
            scores=scores.astype(np.float32),
            contacts=contacts.astype(np.float32),
        )


def _empty_result() -> GraspResult:
    return GraspResult(
        poses=np.zeros((0, 4, 4), dtype=np.float32),
        positions=np.zeros((0, 3), dtype=np.float32),
        quaternions=np.zeros((0, 4), dtype=np.float32),
        widths=np.zeros((0,), dtype=np.float32),
        scores=np.zeros((0,), dtype=np.float32),
        contacts=np.zeros((0, 3), dtype=np.float32),
    )


# ───────────────────── ACRONYM-style output helpers ─────────────────────────

_TRAINING_NPZ_RE = re.compile(
    r"data/out/(?P<split>train|val|test)/(?P<category>[^/]+)/"
    r"(?P<mesh_hash>[0-9a-fA-F]+)/(?P<view>\d+)\.npz$"
)


def _find_manifest_entry(manifest: list, category: Optional[str],
                         mesh_hash: Optional[str]) -> Optional[dict]:
    """Return the first manifest entry matching ``category`` and ``mesh_hash``."""
    for entry in manifest:
        if category is not None and entry.get("category") != category:
            continue
        if mesh_hash is not None and entry.get("mesh_hash") != mesh_hash:
            continue
        return entry
    return None


def _parse_training_npz_path(points_path: str) -> Optional[dict]:
    """Pull ``category`` / ``mesh_hash`` / ``split`` from a training sample path."""
    norm = points_path.replace("\\", "/")
    m = _TRAINING_NPZ_RE.search(norm)
    if not m:
        return None
    return {
        "split": m.group("split"),
        "category": m.group("category"),
        "mesh_hash": m.group("mesh_hash"),
        "view": m.group("view"),
    }


def resolve_mesh_info(points_path: str,
                      manifest_path: Optional[str],
                      category: Optional[str] = None,
                      mesh_hash: Optional[str] = None) -> MeshInfo:
    """Best-effort lookup of ``category`` / ``mesh_hash`` / ``scale`` / mesh paths.

    Strategy: parse the ``.npz`` path against the training layout
    ``data/out/<split>/<Category>/<mesh_hash>/<view>.npz`` and cross-reference
    with ``manifest.json``. Explicit ``--category`` / ``--mesh-hash`` flags
    always win over path-inference.
    """
    info: MeshInfo = {
        "category": category,
        "mesh_hash": mesh_hash,
        "split": None,
        "view": None,
        "mesh_path": None,
        "grasp_file": None,
        "scale": None,
    }
    parsed = _parse_training_npz_path(points_path) or {}
    info["split"] = parsed.get("split")
    info["view"] = parsed.get("view")
    info["category"] = info["category"] or parsed.get("category")
    info["mesh_hash"] = info["mesh_hash"] or parsed.get("mesh_hash")

    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        entry = _find_manifest_entry(manifest, info["category"], info["mesh_hash"])
        if entry is not None:
            info["mesh_path"] = entry.get("mesh_path")
            info["grasp_file"] = entry.get("grasp_file")
            info["scale"] = entry.get("scale")
            info["category"] = info["category"] or entry.get("category")
            info["mesh_hash"] = info["mesh_hash"] or entry.get("mesh_hash")
    return info


def write_grasps_h5(path: str, grasps: "GraspResult",
                    mesh_info: Optional[MeshInfo] = None) -> None:
    """Write an ACRONYM-compatible ``.h5`` with predicted grasps.

    Matches the datasets used elsewhere in this repo:
      * ``grasps/transforms``               (K, 4, 4) float32
      * ``grasps/qualities/flex/object_in_gripper`` (K,) uint8 (= score>=0.5)
      * ``grasps/widths``                   (K,) float32
    Additional (non-ACRONYM) datasets for convenience:
      * ``grasps/scores`` / ``grasps/positions`` / ``grasps/quaternions``
      * ``grasps/contacts``
      * ``object/file`` / ``object/scale`` / ``object/category`` attrs when known
    """
    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "Writing .h5 grasp files requires h5py. `pip install h5py`."
        ) from e

    mesh_info = mesh_info or {}
    K = int(grasps.poses.shape[0])

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        g = f.create_group("grasps")
        g.create_dataset("transforms", data=grasps.poses.astype(np.float32))
        g.create_dataset("widths", data=grasps.widths.astype(np.float32))
        g.create_dataset("scores", data=grasps.scores.astype(np.float32))
        g.create_dataset("positions", data=grasps.positions.astype(np.float32))
        g.create_dataset("quaternions", data=grasps.quaternions.astype(np.float32))
        g.create_dataset("contacts", data=grasps.contacts.astype(np.float32))

        qual = g.create_group("qualities").create_group("flex")
        in_gripper = (grasps.scores >= 0.5).astype(np.uint8) if K else \
            np.zeros((0,), dtype=np.uint8)
        qual.create_dataset("object_in_gripper", data=in_gripper)

        obj = f.create_group("object")
        if mesh_info.get("mesh_path") is not None:
            obj.attrs["file"] = str(mesh_info["mesh_path"])
        if mesh_info.get("scale") is not None:
            obj.attrs["scale"] = float(mesh_info["scale"])
        if mesh_info.get("category") is not None:
            obj.attrs["category"] = str(mesh_info["category"])
        if mesh_info.get("mesh_hash") is not None:
            obj.attrs["mesh_hash"] = str(mesh_info["mesh_hash"])

