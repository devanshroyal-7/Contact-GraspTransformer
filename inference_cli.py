"""CLI entry point for ContactGraspNet inference.

Library API lives in ``inference.py`` (``GraspPredictor``, I/O helpers).
Run::

    python inference_cli.py --ckpt checkpoints/ptv3/<run_folder>/best.pt \\
        --points data/out/train/Camera/<hash>/001.npz --top-k 100
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

import numpy as np
import torch

from inference import (
    DEFAULT_NUM_POINTS,
    GraspPredictor,
    MeshInfo,
    load_point_cloud,
    prepare_model_input,
    resolve_mesh_info,
    write_grasps_h5,
)


def default_run_stem(mesh_info: MeshInfo, points_path: str) -> str:
    """ACRONYM-style ``<Category>_<mesh_hash>_<scale>`` when possible."""
    cat = mesh_info.get("category")
    h = mesh_info.get("mesh_hash")
    scale = mesh_info.get("scale")
    if cat and h and scale is not None:
        return f"{cat}_{h}_{scale}"
    base = os.path.splitext(os.path.basename(points_path))[0]
    return f"{base or 'grasps'}_pred"


def time_model_forward_pass(
    predictor: GraspPredictor,
    points: np.ndarray,
    seed: Optional[int] = None,
) -> float:
    """Measure one complete model forward pass on the inference-ready tensor."""
    _, _, xyz = prepare_model_input(
        points,
        num_points=predictor.num_points,
        device=predictor.device,
        seed=seed,
    )
    is_cuda = xyz.device.type == "cuda"

    predictor.model.eval()
    with torch.no_grad():
        if is_cuda:
            torch.cuda.synchronize(xyz.device)
        start = time.perf_counter()
        predictor.model(xyz)
        if is_cuda:
            torch.cuda.synchronize(xyz.device)
    return (time.perf_counter() - start) * 1000.0


def iter_point_cloud_paths(root: str) -> list[str]:
    exts = {".npy", ".npz", ".ply", ".pcd", ".xyz", ".txt"}
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in exts:
                paths.append(os.path.join(dirpath, filename))
    return sorted(paths)


def time_test_dataset(
    predictor: GraspPredictor,
    test_data_dir: str,
    seed: Optional[int] = None,
) -> dict[str, object]:
    paths = iter_point_cloud_paths(test_data_dir)
    if not paths:
        raise FileNotFoundError(f"No point-cloud files found under {test_data_dir}")

    timings: list[tuple[str, float]] = []
    for path in paths:
        points = load_point_cloud(path)
        timings.append((path, time_model_forward_pass(predictor, points, seed=seed)))

    times_ms = [elapsed for _, elapsed in timings]
    fastest_path, fastest_ms = min(timings, key=lambda item: item[1])
    slowest_path, slowest_ms = max(timings, key=lambda item: item[1])

    return {
        "num_samples": len(timings),
        "fastest_ms": fastest_ms,
        "fastest_path": fastest_path,
        "slowest_ms": slowest_ms,
        "slowest_path": slowest_path,
        "average_ms": sum(times_ms) / len(times_ms),
        "total_ms": sum(times_ms),
        "timings_ms": timings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ContactGraspNet inference")
    parser.add_argument("--ckpt", required=True, help="Path to trained .pt file")
    parser.add_argument("--points", default=None,
                        help="Input point cloud (.npy/.npz/.ply/.pcd/.xyz)")
    parser.add_argument("--test-data-dir", default="data/out/test",
                        help="Test split root used by --time-it")
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
    parser.add_argument("--time-it", action="store_true",
                        help="Time one forward pass for each sample in --test-data-dir")
    args = parser.parse_args()

    if not args.time_it and args.points is None:
        parser.error("--points is required unless --time-it is set")

    predictor = GraspPredictor(
        ckpt_path=args.ckpt,
        backbone=args.backbone,
        num_points=args.num_points,
        device=args.device,
        cpe_mode=args.cpe_mode,
    )

    if args.time_it:
        timing = time_test_dataset(
            predictor,
            test_data_dir=args.test_data_dir,
            seed=args.seed,
        )
        print(
            "Test-set inference forward timing "
            f"({timing['num_samples']} samples): "
            f"fastest={timing['fastest_ms']:.3f} ms, "
            f"slowest={timing['slowest_ms']:.3f} ms, "
            f"average={timing['average_ms']:.3f} ms, "
            f"total={timing['total_ms']:.3f} ms"
        )
        print(f"Fastest sample: {os.path.relpath(timing['fastest_path'])}")
        print(f"Slowest sample: {os.path.relpath(timing['slowest_path'])}")
        return

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
    stem = args.run_name or default_run_stem(mesh_info, args.points)
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
