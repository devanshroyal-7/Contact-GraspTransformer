"""Visualize PTv3 voxels during real ContactGraspNet inference.

This is the inference-time counterpart to ``voxel_viz.py``. It loads a trained
checkpoint, runs the same preprocessing and forward pass used by
``GraspPredictor.predict``, and records the voxel grids produced by each PTv3
``VoxelPoolDown`` layer.

Example:
    python inference_voxel_viz.py --ckpt checkpoints/best.pt \
        --points data/out/train/Mug/2997f21fa426e18a6ab1a25d0e8f3590/000.npz
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from inference import GraspPredictor, load_point_cloud, sample_points
from models.backbone_ptv3 import PTv3Wrapper
from voxel_viz import draw_geometries, quantize_like_ptv3, voxels_to_gradient_mesh


@dataclass
class VoxelFrame:
    name: str
    grid_coord: np.ndarray
    voxel_size: float
    origin: np.ndarray
    xyz: Optional[np.ndarray] = None
    feature_norm: Optional[np.ndarray] = None


def _resolve_device(device_name: Optional[str]) -> torch.device:
    if device_name is None or device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return torch.device(device_name)


def _find_ptv3_backbone(predictor: GraspPredictor) -> PTv3Wrapper:
    backbone = predictor.model.backbone
    if not isinstance(backbone, PTv3Wrapper):
        raise TypeError(
            "Voxel inference visualization requires a PTv3 checkpoint/backbone. "
            f"Loaded backbone is {type(backbone).__name__}."
        )
    return backbone


def _tensor_to_numpy(tensor: torch.Tensor, batch_index: int = 0) -> np.ndarray:
    return tensor[batch_index].detach().cpu().numpy()


def _valid_rows(array: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return array[valid.astype(bool)]


def _make_point_cloud(points: np.ndarray):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.paint_uniform_color([0.08, 0.08, 0.08])
    return pcd


def _make_voxel_wireframe(
    grid_coord: np.ndarray,
    voxel_size: float,
    origin: np.ndarray,
    values: Optional[np.ndarray] = None,
):
    """Build voxel cube outlines so overlaid points stay visible."""
    import matplotlib.pyplot as plt
    import open3d as o3d

    if len(grid_coord) == 0:
        return o3d.geometry.LineSet()

    cmap = plt.get_cmap("turbo")
    if values is None:
        centers = origin + (grid_coord.astype(np.float64) + 0.5) * voxel_size
        color_values = centers[:, 2]
    else:
        if len(values) != len(grid_coord):
            raise ValueError(
                f"Expected {len(grid_coord)} color values, got {len(values)}"
            )
        color_values = values

    span = color_values.max() - color_values.min()
    if span == 0:
        color_values = np.zeros_like(color_values, dtype=np.float64)
    else:
        color_values = (color_values - color_values.min()) / span

    points = []
    lines = []
    colors = []
    corners = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.float64,
    )
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    for coord, color_value in zip(grid_coord, color_values):
        start = len(points)
        voxel_min = origin + coord.astype(np.float64) * voxel_size
        points.extend(voxel_min + corners * voxel_size)
        color = list(cmap(float(color_value))[:3])
        for edge_start, edge_end in edges:
            lines.append([start + edge_start, start + edge_end])
            colors.append(color)

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64)),
        lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32)),
    )
    line_set.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return line_set


def _maybe_subsample_frame(frame: VoxelFrame, max_voxels: int, seed: int) -> VoxelFrame:
    if max_voxels <= 0 or len(frame.grid_coord) <= max_voxels:
        return frame

    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(frame.grid_coord), max_voxels, replace=False))
    feature_norm = None if frame.feature_norm is None else frame.feature_norm[keep]
    xyz = None if frame.xyz is None else frame.xyz[keep]
    return VoxelFrame(
        name=f"{frame.name} (showing {max_voxels}/{len(frame.grid_coord)} voxels)",
        grid_coord=frame.grid_coord[keep],
        voxel_size=frame.voxel_size,
        origin=frame.origin,
        xyz=xyz,
        feature_norm=feature_norm,
    )


def capture_inference_voxels(
    predictor: GraspPredictor,
    points: np.ndarray,
    *,
    seed: Optional[int] = 0,
) -> tuple[list[VoxelFrame], dict[str, np.ndarray]]:
    """Run one real inference pass and return voxel frames from the PTv3 encoder."""
    backbone = _find_ptv3_backbone(predictor)
    device = predictor.device
    rng = np.random.default_rng(seed)

    sampled = sample_points(points.astype(np.float32), predictor.num_points, rng)
    centroid = sampled.mean(axis=0, keepdims=True).astype(np.float32)
    centred = sampled - centroid

    base_grid, base_origin_centred = quantize_like_ptv3(
        centred.astype(np.float64),
        backbone.grid_size,
    )
    base_origin = base_origin_centred + centroid.squeeze(0).astype(np.float64)
    unique_base_grid = np.unique(base_grid, axis=0)

    frames: list[VoxelFrame] = [
        VoxelFrame(
            name="Input voxelization",
            grid_coord=unique_base_grid,
            voxel_size=float(backbone.grid_size),
            origin=base_origin,
            xyz=sampled,
        )
    ]

    hooks = []

    def make_hook(stage_index: int, cumulative_stride: int):
        def hook(_module, _inputs, output):
            new_xyz, new_gc, new_feat, new_valid, _inverse = output
            valid = _tensor_to_numpy(new_valid).astype(bool)
            grid = _valid_rows(_tensor_to_numpy(new_gc), valid)
            xyz = _valid_rows(_tensor_to_numpy(new_xyz), valid)
            feat = _valid_rows(_tensor_to_numpy(new_feat), valid)
            xyz = xyz + centroid.squeeze(0)
            frames.append(
                VoxelFrame(
                    name=f"Encoder pool {stage_index}",
                    grid_coord=grid.astype(np.int64),
                    voxel_size=float(backbone.grid_size * cumulative_stride),
                    origin=base_origin,
                    xyz=xyz.astype(np.float64),
                    feature_norm=np.linalg.norm(feat, axis=1),
                )
            )

        return hook

    cumulative_stride = 1
    for stage_index, (down_block, stride) in enumerate(
        zip(backbone.down_blocks, backbone.pool_strides),
        start=1,
    ):
        cumulative_stride *= int(stride)
        hooks.append(down_block.register_forward_hook(make_hook(stage_index, cumulative_stride)))

    try:
        xyz = torch.from_numpy(centred).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = predictor.model(xyz)
    finally:
        for hook in hooks:
            hook.remove()

    scores = preds["confidence"][0].detach().cpu().numpy()
    summary = {
        "sampled": sampled,
        "centroid": centroid.squeeze(0),
        "scores": scores,
    }
    return frames, summary


def visualize_frames(frames: list[VoxelFrame], args) -> None:
    for idx, original_frame in enumerate(frames):
        frame = _maybe_subsample_frame(
            original_frame,
            max_voxels=args.max_voxels,
            seed=args.seed + idx,
        )
        values = None
        if args.color == "feature_norm" and frame.feature_norm is not None:
            values = frame.feature_norm
        elif args.color == "height":
            values = None
        elif args.color == "stage":
            values = np.full(len(frame.grid_coord), idx, dtype=np.float64)

        print(
            f"{frame.name}: voxel_size={frame.voxel_size:.6g} m | "
            f"occupied_voxels={len(original_frame.grid_coord)}"
        )
        voxel_geometry = (
            _make_voxel_wireframe(
                frame.grid_coord,
                frame.voxel_size,
                frame.origin,
                values=values,
            )
            if args.show_points
            else voxels_to_gradient_mesh(
                frame.grid_coord,
                frame.voxel_size,
                frame.origin,
                values=values,
            )
        )
        geometries = [voxel_geometry]
        if args.show_points and frame.xyz is not None:
            geometries.append(_make_point_cloud(frame.xyz))

        draw_geometries(
            geometries,
            window_name=frame.name,
            width=args.width,
            height=args.height,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ContactGraspNet inference and visualize PTv3 voxel grids."
    )
    parser.add_argument("--ckpt", default="checkpoints/best.pt", help="Trained .pt checkpoint")
    parser.add_argument("--points", required=True, help="Input point cloud (.npy/.npz/.ply/.pcd/.xyz)")
    parser.add_argument(
        "--backbone",
        default="ptv3",
        choices=["pn2", "ptv3"],
        help="Fallback backbone if checkpoint config is missing",
    )
    parser.add_argument(
        "--cpe-mode",
        default=None,
        choices=["knn", "conv1d", "sparse3d"],
        help="Override PTv3 cpe_mode when checkpoint config is missing or wrong",
    )
    parser.add_argument("--num-points", type=int, default=4096)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--color",
        choices=["height", "feature_norm", "stage"],
        default="feature_norm",
        help="Voxel color source. Input voxels always fall back to height.",
    )
    parser.add_argument(
        "--max-voxels",
        type=int,
        default=0,
        help="Randomly show at most this many voxels per window; 0 shows all.",
    )
    parser.add_argument("--show-points", action="store_true", help="Overlay pooled point centers")
    parser.add_argument("--width", type=int, default=1100)
    parser.add_argument("--height", type=int, default=750)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    predictor = GraspPredictor(
        ckpt_path=args.ckpt,
        backbone=args.backbone,
        num_points=args.num_points,
        device=str(device),
        cpe_mode=args.cpe_mode,
    )

    points = load_point_cloud(args.points)
    print(f"Loaded {points.shape[0]} points from {args.points}")
    frames, summary = capture_inference_voxels(predictor, points, seed=args.seed)
    scores = summary["scores"]
    print(
        "Ran real model inference "
        f"| sampled_points={len(summary['sampled'])} "
        f"| score min/mean/max={scores.min():.4f}/{scores.mean():.4f}/{scores.max():.4f}"
    )
    visualize_frames(frames, args)


if __name__ == "__main__":
    main()
