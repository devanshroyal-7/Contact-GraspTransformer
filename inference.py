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
  `geometry_msgs/PoseStamped`. Positions and quaternions are in the
  point cloud's frame, so set `header.frame_id` to whatever TF frame
  the point cloud came in (e.g. `"camera_depth_optical_frame"`). A
  ready-to-use helper is `grasp_to_ros_pose_dict` at the bottom of
  this file - it returns a plain dict so you don't need rospy at
  import time.
* Franka Panda: `poses[i]` can be used as the goal for the
  `panda_hand` (not `panda_link8`) frame directly.

Usage
-----
    # Programmatic (``train.py`` checkpoints: ``model_state_dict`` + ``config``)
    from inference import GraspPredictor
    predictor = GraspPredictor("checkpoints/best.pt")
    grasps = predictor.predict(points_np, top_k=50, score_thresh=0.5)

    # CLI (writes out/<run>.h5 (ACRONYM layout) + out/<run>.json with mesh info)
    python inference.py --ckpt checkpoints/best.pt \\
        --points data/out/train/Camera/<hash>/001.npz --top-k 100
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from models.model import ContactGraspNet
from models.cgn_heads import PANDA_BASELINE_DIST


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
            raise RuntimeError(
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
    * Exactly right    -> shuffled in place (to avoid any ordering bias).
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

def _rotation_to_quaternion_xyzw(R: np.ndarray) -> np.ndarray:
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


def _build_poses(contacts: np.ndarray,
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

    t = (contacts
         + 0.5 * widths[:, None] * x
         + PANDA_BASELINE_DIST * z)

    poses = np.zeros((K, 4, 4), dtype=np.float32)
    poses[:, :3, :3] = R
    poses[:, :3, 3] = t
    poses[:, 3, 3] = 1.0
    return poses


def _nms(positions: np.ndarray, scores: np.ndarray,
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


# ─────────────────────────── predictor ───────────────────────────────────────

@dataclass
class GraspResult:
    poses: np.ndarray        # (K, 4, 4)
    positions: np.ndarray    # (K, 3)
    quaternions: np.ndarray  # (K, 4) xyzw
    widths: np.ndarray       # (K,)
    scores: np.ndarray       # (K,)
    contacts: np.ndarray     # (K, 3)

    def as_dict(self) -> dict:
        return {
            "poses": self.poses,
            "positions": self.positions,
            "quaternions": self.quaternions,
            "widths": self.widths,
            "scores": self.scores,
            "contacts": self.contacts,
        }


def _torch_load_checkpoint(path: str, map_location):
    """Load a ``.pt`` file; allow full training dicts (PyTorch 2.6+ safe)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _infer_cpe_mode_from_state_dict(state: dict) -> Optional[str]:
    """Best-effort CPE mode detection when the checkpoint has no ``config``.

    Checks one representative parameter shape:
      * ``knn``      -> ``backbone.enc_blocks.0.0.cpe.weight`` (bare parameter)
      * ``conv1d`` / ``sparse3d`` -> ``backbone.enc_blocks.0.0.cpe.conv.weight``

    Returns ``None`` if the keys are not found (e.g. ``pn2`` backbone).
    """
    if not isinstance(state, dict):
        return None
    if "backbone.enc_blocks.0.0.cpe.weight" in state:
        return "knn"
    if "backbone.enc_blocks.0.0.cpe.conv.weight" in state:
        w = state["backbone.enc_blocks.0.0.cpe.conv.weight"]
        # sparse3d's SubMConv3d weight is 5-D (kD,kH,kW, Cin, Cout),
        # Conv1DCPE's Conv1d weight is 3-D (Cout, Cin/groups, k).
        if hasattr(w, "ndim"):
            return "sparse3d" if w.ndim >= 4 else "conv1d"
    return None


def _state_dict_and_config_from_checkpoint(raw) -> Tuple[dict, dict]:
    """Split ``torch.load`` output into a ``state_dict`` and optional train config.

    Supports ``train.py`` checkpoints (``model_state_dict`` + ``config``),
    ``{"model": ...}``, Lightning-style ``state_dict``, or a bare state_dict.
    """
    ckpt_cfg: dict = {}
    if not isinstance(raw, dict):
        return raw, ckpt_cfg

    if "model_state_dict" in raw:
        return raw["model_state_dict"], dict(raw.get("config") or {})
    if "model" in raw:
        return raw["model"], dict(raw.get("config") or {})
    if "state_dict" in raw:
        return raw["state_dict"], dict(raw.get("config") or {})

    if any(k in raw for k in ("epoch", "optimizer_state_dict", "scheduler_state_dict")):
        raise ValueError(
            "Checkpoint dict is missing 'model_state_dict' / 'model' / 'state_dict'."
        )
    return raw, ckpt_cfg


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

        raw = _torch_load_checkpoint(ckpt_path, map_location=self.device)
        state, ckpt_cfg = _state_dict_and_config_from_checkpoint(raw)

        eff_backbone = str(ckpt_cfg.get("backbone", backbone))
        self.num_points = int(ckpt_cfg.get("num_points", num_points))

        backbone_kwargs = None
        if eff_backbone == "ptv3":
            # Explicit constructor arg wins over checkpoint config; both win
            # over the "knn" fallback. Old checkpoints with empty config let
            # you recover by passing cpe_mode explicitly.
            eff_cpe = cpe_mode or ckpt_cfg.get("cpe_mode")
            if eff_cpe is None:
                eff_cpe = _infer_cpe_mode_from_state_dict(state) or "knn"
            backbone_kwargs = {"cpe_mode": str(eff_cpe)}

        self.model = ContactGraspNet(
            backbone_type=eff_backbone,
            backbone_kwargs=backbone_kwargs,
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
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must be (N, 3), got {points.shape}")
        rng = np.random.default_rng(seed)

        sampled = sample_points(points.astype(np.float32), self.num_points, rng)

        # Mean-subtract: the model was trained on mean-centred clouds.
        centroid = sampled.mean(axis=0, keepdims=True).astype(np.float32)
        centred = sampled - centroid

        xyz = torch.from_numpy(centred).unsqueeze(0).to(self.device)  # (1, N, 3)
        preds = self.model(xyz)

        scores = preds["confidence"][0].cpu().numpy()                   # (N,)
        approach = preds["approach_dirs"][0].cpu().numpy()              # (N, 3)
        baseline = preds["base_dirs"][0].cpu().numpy()                  # (N, 3)
        widths = preds["widths"][0].cpu().numpy()                       # (N,)

        # Back to the original (input) frame.
        contacts = centred + centroid  # == sampled
        contacts = contacts.astype(np.float32)

        # Filter by confidence.
        keep = scores >= score_thresh
        if not keep.any():
            return _empty_result()

        contacts = contacts[keep]
        approach = approach[keep]
        baseline = baseline[keep]
        widths = widths[keep]
        scores = scores[keep]

        poses = _build_poses(contacts, approach, baseline, widths)
        positions = poses[:, :3, 3].copy()
        quats = _rotation_to_quaternion_xyzw(poses[:, :3, :3])

        # Optional NMS on grasp position.
        if nms_radius > 0.0 and len(positions) > 1:
            keep_idx = _nms(positions, scores, nms_radius)
            poses = poses[keep_idx]
            positions = positions[keep_idx]
            quats = quats[keep_idx]
            widths = widths[keep_idx]
            scores = scores[keep_idx]
            contacts = contacts[keep_idx]

        # Top-K by score.
        if top_k is not None and len(scores) > top_k:
            order = np.argsort(-scores)[:top_k]
            poses = poses[order]
            positions = positions[order]
            quats = quats[order]
            widths = widths[order]
            scores = scores[order]
            contacts = contacts[order]
        else:
            order = np.argsort(-scores)
            poses = poses[order]
            positions = positions[order]
            quats = quats[order]
            widths = widths[order]
            scores = scores[order]
            contacts = contacts[order]

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
                      mesh_hash: Optional[str] = None) -> dict:
    """Best-effort lookup of ``category`` / ``mesh_hash`` / ``scale`` / mesh paths.

    Strategy: parse the ``.npz`` path against the training layout
    ``data/out/<split>/<Category>/<mesh_hash>/<view>.npz`` and cross-reference
    with ``manifest.json``. Explicit ``--category`` / ``--mesh-hash`` flags
    always win over path-inference.
    """
    info: dict = {
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


def _default_run_stem(mesh_info: dict, points_path: str) -> str:
    """ACRONYM-style ``<Category>_<mesh_hash>_<scale>`` when possible."""
    cat = mesh_info.get("category")
    h = mesh_info.get("mesh_hash")
    scale = mesh_info.get("scale")
    if cat and h and scale is not None:
        return f"{cat}_{h}_{scale}"
    base = os.path.splitext(os.path.basename(points_path))[0]
    return f"{base or 'grasps'}_pred"


def write_grasps_h5(path: str, grasps: "GraspResult",
                    mesh_info: Optional[dict] = None) -> None:
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
        raise RuntimeError(
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


# ─────────────────────── ROS / simulator helpers ─────────────────────────────

def grasp_to_ros_pose_dict(position: np.ndarray,
                           quaternion_xyzw: np.ndarray) -> dict:
    """Convert a single grasp to a dict matching geometry_msgs/Pose.

    The dict maps 1:1 to ROS fields so you can build the actual message
    without importing rospy here:

        from geometry_msgs.msg import PoseStamped
        p = PoseStamped()
        p.header.frame_id = "camera_depth_optical_frame"
        p.header.stamp = rospy.Time.now()
        d = grasp_to_ros_pose_dict(pos, quat_xyzw)
        p.pose.position.x = d["position"]["x"]
        ...
    """
    return {
        "position": {
            "x": float(position[0]),
            "y": float(position[1]),
            "z": float(position[2]),
        },
        "orientation": {
            "x": float(quaternion_xyzw[0]),
            "y": float(quaternion_xyzw[1]),
            "z": float(quaternion_xyzw[2]),
            "w": float(quaternion_xyzw[3]),
        },
    }


# ───────────────────────────── CLI ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ContactGraspNet inference")
    parser.add_argument("--ckpt", required=True, help="Path to trained .pt file")
    parser.add_argument("--points", required=True,
                        help="Input point cloud (.npy/.npz/.ply/.pcd/.xyz)")
    parser.add_argument("--out-dir", default="out",
                        help="Directory to write <run>.h5 + <run>.json into")
    parser.add_argument("--run-name", default=None,
                        help="Output stem; defaults to "
                             "<Category>_<mesh_hash>_<scale> when resolvable, "
                             "else <points-basename>_pred")
    parser.add_argument("--manifest", default="data/acronym/manifest.json",
                        help="ACRONYM manifest for mesh/scale lookup")
    parser.add_argument("--acronym-root", default="data/acronym",
                        help="Root used to resolve mesh_path / grasp_file")
    parser.add_argument("--category", default=None,
                        help="Override category (otherwise inferred from path)")
    parser.add_argument("--mesh-hash", default=None,
                        help="Override mesh hash (otherwise inferred from path)")
    parser.add_argument("--also-npz", action="store_true",
                        help="Additionally write a sidecar .npz of the grasps")
    parser.add_argument(
        "--backbone",
        default="ptv3",
        choices=["pn2", "ptv3"],
        help="Fallback backbone if the checkpoint has no embedded train config",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=DEFAULT_NUM_POINTS,
        help="Fallback N when the checkpoint has no embedded num_points",
    )
    parser.add_argument(
        "--cpe-mode",
        default=None,
        choices=["knn", "conv1d", "sparse3d"],
        help="Override PTv3 cpe_mode (auto-detected from weights when possible)",
    )
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--nms-radius", type=float, default=0.02,
                        help="NMS radius in metres (0 disables)")
    parser.add_argument("--device", default=None,
                        help="'cuda', 'cpu' (default: cuda if available)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    predictor = GraspPredictor(
        ckpt_path=args.ckpt,
        backbone=args.backbone,
        num_points=args.num_points,
        device=args.device,
        cpe_mode=args.cpe_mode,
    )

    points = load_point_cloud(args.points)
    print(f"Loaded {points.shape[0]} points from {args.points}")

    grasps = predictor.predict(
        points,
        score_thresh=args.score_thresh,
        top_k=args.top_k,
        nms_radius=args.nms_radius,
        seed=args.seed,
    )
    print(f"Predicted {len(grasps.scores)} grasps "
          f"(score>={args.score_thresh}, top_k={args.top_k}, "
          f"nms_r={args.nms_radius})")

    mesh_info = resolve_mesh_info(
        args.points,
        manifest_path=args.manifest,
        category=args.category,
        mesh_hash=args.mesh_hash,
    )
    stem = args.run_name or _default_run_stem(mesh_info, args.points)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    h5_path = os.path.join(out_dir, f"{stem}.h5")
    json_path = os.path.join(out_dir, f"{stem}.json")

    write_grasps_h5(h5_path, grasps, mesh_info=mesh_info)
    print(f"Saved -> {h5_path}")

    meta = {
        "run_name": stem,
        "h5": os.path.relpath(h5_path),
        "points": os.path.abspath(args.points),
        "ckpt": os.path.abspath(args.ckpt),
        "frame": "input_point_cloud",
        "num_grasps": int(grasps.scores.shape[0]),
        "score_thresh": float(args.score_thresh),
        "top_k": int(args.top_k) if args.top_k is not None else None,
        "nms_radius": float(args.nms_radius),
        "mesh": {
            "category": mesh_info.get("category"),
            "mesh_hash": mesh_info.get("mesh_hash"),
            "scale": mesh_info.get("scale"),
            "mesh_path": mesh_info.get("mesh_path"),
            "grasp_file": mesh_info.get("grasp_file"),
            "acronym_root": os.path.abspath(args.acronym_root)
            if os.path.exists(args.acronym_root) else None,
            "mesh_path_abs": (
                os.path.abspath(os.path.join(args.acronym_root, mesh_info["mesh_path"]))
                if mesh_info.get("mesh_path") and os.path.exists(args.acronym_root)
                else None
            ),
        },
    }
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    print(f"Saved -> {json_path}")

    if args.also_npz:
        npz_path = os.path.join(out_dir, f"{stem}.npz")
        np.savez(npz_path, **grasps.as_dict())
        print(f"Saved -> {npz_path}")


if __name__ == "__main__":
    main()
