"""Grasp selection, Trimesh preview and single-IK MuJoCo execution."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Any

import h5py
import mujoco
import mujoco.viewer
import numpy as np
import trimesh

# Allow running as script: python eval/visualize_grasp.py ...
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from eval.ik_retarget import (
    ExecPhase,
    RetargetConfig,
    SimpleIKGraspExecutor,
    build_retarget_plan,
    build_retarget_plan_from_hand_pose,
    compute_pose_error,
    contact_to_hand_pose,
    marker_to_hand_pose,
)
from eval.scene_builder import build_scene
from eval.trimesh_preview import (
    show_grasp_comparison_preview,
    show_grasp_preview,
    show_grasp_set_preview,
)
from eval.utils import cam_centred_to_world, grasp_world_to_mujoco, make_grasp_pose
from eval.visualize_types import GraspSpec, validate_pose_se3
from models.model import ContactGraspNet


CONTACT_FRAME_SOURCES = {"labels", "pred_cgn", "pred_ptv3"}


def _extract_state_dict(ckpt: Any) -> dict[str, Any]:
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]
    else:
        state = ckpt
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a valid state_dict")
    if state:
        k0 = next(iter(state.keys()))
        if isinstance(k0, str) and k0.startswith("module."):
            state = {k[len("module.") :]: v for k, v in state.items()}
    return state


def _checkpoint_config_value(ckpt: Any, name: str, default: Any = None) -> Any:
    if not isinstance(ckpt, dict):
        return default
    cfg = ckpt.get("config")
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _load_model_for_backbone(
    backbone: str,
    checkpoint: str,
    device: str,
    *,
    ptv3_cpe_mode: str = "auto",
) -> ContactGraspNet:
    import torch

    dev = torch.device(device)
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
    ckpt_backbone = _checkpoint_config_value(ckpt, "backbone")
    if ckpt_backbone is not None and str(ckpt_backbone) != str(backbone):
        raise ValueError(
            f"Checkpoint was trained with backbone={ckpt_backbone!r}, "
            f"but --source requested backbone={backbone!r}."
        )

    backbone_kwargs: dict[str, Any] | None = None
    if backbone == "ptv3":
        cpe_mode = (
            str(_checkpoint_config_value(ckpt, "cpe_mode", "knn"))
            if ptv3_cpe_mode == "auto"
            else str(ptv3_cpe_mode)
        )
        backbone_kwargs = {"cpe_mode": cpe_mode}
    else:
        cpe_mode = None

    try:
        model = ContactGraspNet(backbone_type=backbone, backbone_kwargs=backbone_kwargs).to(dev)
    except ImportError as exc:
        if backbone == "ptv3" and cpe_mode == "sparse3d":
            raise ValueError(
                "This PTv3 checkpoint was trained with cpe_mode='sparse3d', "
                "which requires spconv at evaluation time. Install a compatible "
                "spconv package in this environment, or evaluate a PTv3 checkpoint "
                "trained with --cpe_mode knn/conv1d."
            ) from exc
        raise

    state = _extract_state_dict(ckpt)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        detail = str(exc).splitlines()[0]
        raise ValueError(
            "Checkpoint is incompatible with this repo architecture/backbone. "
            f"Requested backbone={backbone!r}"
            + (f", cpe_mode={cpe_mode!r}" if cpe_mode is not None else "")
            + f". Detail: {detail}"
        ) from exc
    model.eval()
    return model


def _load_pred_candidates(
    args,
    obj_xy: tuple[float, float],
    *,
    backbone: str,
    source_name: str,
    grasp_index: int | None = None,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Build ranked MuJoCo replay candidates from a checkpoint prediction."""
    _require(args.view_npz is not None, "--view_npz is required")
    _require(args.checkpoint is not None, f"--checkpoint is required for {source_name}")
    npz_data = _load_npz_dict(args.view_npz)

    import torch

    model = _load_model_for_backbone(
        backbone,
        args.checkpoint,
        args.device,
        ptv3_cpe_mode=args.ptv3_cpe_mode,
    )
    torch.manual_seed(int(args.eval_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.eval_seed))
    points = np.asarray(npz_data["points"], dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points shape (N,3), got {points.shape}")

    with torch.no_grad():
        preds = model(torch.from_numpy(points).float().unsqueeze(0).to(args.device))

    conf = preds["confidence"].squeeze(0).detach().cpu().numpy().reshape(-1)
    app = preds["approach_dirs"].squeeze(0).detach().cpu().numpy()
    base = preds["base_dirs"].squeeze(0).detach().cpu().numpy()
    widths = preds["widths"].squeeze(0).detach().cpu().numpy().reshape(-1)
    camera_pose = np.asarray(npz_data.get("camera_pose", np.eye(4)), dtype=np.float64)

    if not (len(points) == len(conf) == len(app) == len(base) == len(widths)):
        raise ValueError("model prediction arrays must have matching first dimension")
    valid = (
        np.isfinite(conf)
        & np.isfinite(widths)
        & (widths > 0.0)
        & np.all(np.isfinite(app), axis=1)
        & np.all(np.isfinite(base), axis=1)
    )
    indices = np.flatnonzero(valid)
    if len(indices) == 0:
        raise ValueError(f"{source_name} produced no finite grasp candidates")

    ordered = indices[np.argsort(-conf[indices])]
    selected = _slice_order(
        ordered,
        grasp_index=args.grasp_index if grasp_index is None else grasp_index,
        top_k=args.top_k if top_k is None else top_k,
    )
    rank_by_index = {int(idx): int(rank) for rank, idx in enumerate(ordered)}

    out: list[dict[str, Any]] = []
    for trial_rank, idx in enumerate(selected):
        point_idx = int(idx)
        T_local = make_grasp_pose(points[point_idx], app[point_idx], base[point_idx], width=None)
        T_local = validate_pose_se3(T_local, "pred_contact_pose")
        T_world = cam_centred_to_world(T_local, camera_pose, np.zeros(3, dtype=np.float64))
        T_mujoco = grasp_world_to_mujoco(
            T_world, obj_z_offset=0.0, obj_xy_offset=obj_xy
        )
        out.append(
            {
                "kind": "contact_world",
                "contact_pose_world": validate_pose_se3(T_mujoco, "pred_contact_pose_world"),
                "width_m": float(widths[point_idx]),
                "meta": {
                    "source": source_name,
                    "confidence": float(conf[point_idx]),
                    "score": float(conf[point_idx]),
                    "point_index": point_idx,
                    "grasp_idx": point_idx,
                    "rank": int(rank_by_index[point_idx]),
                    "trial_rank": int(trial_rank),
                    "checkpoint": os.path.abspath(args.checkpoint),
                    "view_npz": os.path.abspath(args.view_npz),
                },
            }
        )
    return out


def _load_label_candidates(
    args,
    obj_xy: tuple[float, float],
    *,
    grasp_index: int | None = None,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Build replay candidates from generated data/out labels.

    Generated label confidence is binary, so a plain argmax picks the first
    positive point in storage order. Rank ties by width first because wider
    generated grasps are less likely to slip in the Panda replay, then by
    world-frame approach z for tabletop-friendly candidates.
    """
    _require(args.view_npz is not None, "--view_npz is required")
    npz_data = _load_npz_dict(args.view_npz)
    points = np.asarray(npz_data["points"], dtype=np.float64)
    conf = np.asarray(npz_data["confidence"], dtype=np.float64).reshape(-1)
    app = np.asarray(npz_data["approach_dirs"], dtype=np.float64)
    base = np.asarray(npz_data["base_dirs"], dtype=np.float64)
    widths = np.asarray(npz_data["widths"], dtype=np.float64).reshape(-1)
    camera_pose = np.asarray(npz_data.get("camera_pose", np.eye(4)), dtype=np.float64)
    label_conf_thresh = float(args.label_conf_thresh)

    if len(points) == 0:
        raise ValueError("No points found in view npz")
    if not (len(points) == len(conf) == len(app) == len(base) == len(widths)):
        raise ValueError("view npz label arrays must have matching first dimension")

    valid = (
        (conf >= label_conf_thresh)
        & np.isfinite(conf)
        & np.isfinite(widths)
        & (widths > 0.0)
        & np.all(np.isfinite(app), axis=1)
        & np.all(np.isfinite(base), axis=1)
    )
    indices = np.flatnonzero(valid)
    if len(indices) == 0:
        raise ValueError(
            f"No generated label candidates with confidence >= {label_conf_thresh:g}"
        )

    world_app_z = np.empty(len(indices), dtype=np.float64)
    for j, idx in enumerate(indices):
        T_local = make_grasp_pose(points[idx], app[idx], base[idx], width=None)
        T_world = cam_centred_to_world(T_local, camera_pose, np.zeros(3, dtype=np.float64))
        world_app_z[j] = float(T_world[2, 2])

    local_order = np.lexsort(
        (
            world_app_z,        # lower z means more top-down in this scene
            -widths[indices],   # wider generated labels are more stable
            -conf[indices],     # confidence first
        )
    )
    ordered = indices[local_order]
    selected = _slice_order(
        ordered,
        grasp_index=args.grasp_index if grasp_index is None else grasp_index,
        top_k=args.top_k if top_k is None else top_k,
    )

    out: list[dict[str, Any]] = []
    app_z_by_index = {int(idx): float(world_app_z[j]) for j, idx in enumerate(indices)}
    rank_by_index = {int(idx): int(rank) for rank, idx in enumerate(ordered)}
    for trial_rank, idx in enumerate(selected):
        point_idx = int(idx)
        T_local = make_grasp_pose(points[point_idx], app[point_idx], base[point_idx], width=None)
        T_local = validate_pose_se3(T_local, "label_contact_pose")
        T_world = cam_centred_to_world(T_local, camera_pose, np.zeros(3, dtype=np.float64))
        T_mujoco = grasp_world_to_mujoco(
            T_world, obj_z_offset=0.0, obj_xy_offset=obj_xy
        )
        out.append(
            {
                "kind": "contact_world",
                "contact_pose_world": validate_pose_se3(T_mujoco, "label_contact_pose_world"),
                "width_m": float(widths[point_idx]),
                "meta": {
                    "source": "labels",
                    "confidence": float(conf[point_idx]),
                    "score": float(conf[point_idx]),
                    "point_index": point_idx,
                    "grasp_idx": point_idx,
                    "rank": int(rank_by_index[point_idx]),
                    "trial_rank": int(trial_rank),
                    "world_approach_z": app_z_by_index[point_idx],
                    "view_npz": os.path.abspath(args.view_npz),
                },
            }
        )
    return out


def _target_object_pose_world(mj_model: mujoco.MjModel, mj_data: mujoco.MjData) -> np.ndarray:
    mujoco.mj_forward(mj_model, mj_data)
    bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    if bid < 0:
        raise RuntimeError("MuJoCo body not found: 'target_object'")
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(mj_data.xmat[bid], dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(mj_data.xpos[bid], dtype=np.float64)
    return validate_pose_se3(T, "target_object_pose_world")


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _load_npz_dict(path: str) -> dict[str, np.ndarray]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"view npz not found: {path}")
    data = dict(np.load(path))
    if "points" not in data:
        raise ValueError("view npz must contain 'points'")
    return data


def _parse_view_npz_meta(path: str) -> tuple[str | None, str | None, str | None]:
    """Try to parse data/out/<split>/<category>/<mesh_hash>/<view>.npz structure."""
    norm = os.path.normpath(path)
    parts = norm.split(os.sep)
    if len(parts) < 5:
        return None, None, None
    if parts[-1].endswith(".npz"):
        mesh_hash = parts[-2]
        category = parts[-3]
        split = parts[-4]
        return split, category, mesh_hash
    return None, None, None


def _select_view_npz(args) -> str:
    if args.view_npz:
        return args.view_npz
    pattern = os.path.join(args.data_dir, args.split, "*", "*", "*.npz")
    files = sorted(glob.glob(pattern))
    if args.category:
        files = [f for f in files if f"{os.sep}{args.category}{os.sep}" in f]
    if args.mesh_hash:
        files = [f for f in files if f"{os.sep}{args.mesh_hash}{os.sep}" in f]
    if not files:
        raise FileNotFoundError(
            f"No views found in {pattern!r} with filters "
            f"category={args.category!r}, mesh_hash={args.mesh_hash!r}"
        )
    idx = int(args.view_index)
    if idx < 0 or idx >= len(files):
        raise ValueError(f"--view_index {idx} out of range for {len(files)} matching views")
    return files[idx]


def _resolve_mesh_and_scale_from_manifest(
    args,
) -> tuple[str, float]:
    _require(os.path.isfile(args.manifest), f"manifest not found: {args.manifest}")
    with open(args.manifest, "r") as f:
        manifest = json.load(f)

    split_from_npz, category_from_npz, mesh_hash_from_npz = _parse_view_npz_meta(args.view_npz)
    split = split_from_npz or args.split
    category = category_from_npz or args.category
    mesh_hash = args.mesh_hash or mesh_hash_from_npz
    if not mesh_hash:
        raise ValueError(
            "Could not infer mesh_hash. Provide --mesh_hash or a standard --view_npz path."
        )

    candidates = [e for e in manifest if e.get("mesh_hash") == mesh_hash]
    if split:
        candidates = [e for e in candidates if e.get("split") == split]
    if category:
        candidates = [e for e in candidates if e.get("category") == category]
    if not candidates:
        raise ValueError(
            f"No manifest entry found for mesh_hash={mesh_hash!r}, split={split!r}, "
            f"category={category!r}"
        )
    entry = candidates[0]
    mesh_path = os.path.join(args.acronym_root, entry["mesh_path"])
    scale = float(entry["scale"])
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(f"mesh not found from manifest: {mesh_path}")
    return mesh_path, scale


def _load_manifest(path: str) -> list[dict[str, Any]]:
    _require(os.path.isfile(path), f"manifest not found: {path}")
    with open(path, "r") as f:
        manifest = json.load(f)
    if not isinstance(manifest, list):
        raise ValueError(f"manifest must contain a list of entries: {path}")
    return manifest


def _resolve_mesh_and_scale_from_grasp_h5(args) -> tuple[str, float]:
    """Resolve mesh metadata by matching an ACRONYM grasp filename in the manifest."""
    _require(args.grasp_h5 is not None, "--grasp_h5 is required")
    grasp_file = os.path.basename(args.grasp_h5)
    try:
        manifest = _load_manifest(args.manifest)
        candidates = [e for e in manifest if e.get("grasp_file") == grasp_file]
        if args.category:
            candidates = [e for e in candidates if e.get("category") == args.category]
        if args.mesh_hash:
            candidates = [e for e in candidates if e.get("mesh_hash") == args.mesh_hash]
        if args.split:
            split_matches = [e for e in candidates if e.get("split") == args.split]
            if split_matches:
                candidates = split_matches
        if candidates:
            entry = candidates[0]
            mesh_path = os.path.join(args.acronym_root, entry["mesh_path"])
            if os.path.isfile(mesh_path):
                return mesh_path, float(entry["scale"])
    except (json.JSONDecodeError, OSError, ValueError):
        pass

    stem = os.path.splitext(grasp_file)[0]
    parts = stem.split("_")
    if len(parts) >= 3:
        category = parts[0]
        scale = float(parts[-1])
        mesh_hash = "_".join(parts[1:-1])
        mesh_path = os.path.join(args.acronym_root, "meshes", category, f"{mesh_hash}.obj")
        if os.path.isfile(mesh_path):
            return mesh_path, scale

    raise ValueError(
        f"Could not resolve mesh/scale for grasp file {grasp_file!r}. "
        "Provide --mesh_path and --mesh_scale explicitly."
    )


def _resolve_repo_relative_path(path: str, *, marker: str) -> str:
    if os.path.isfile(path):
        return path
    norm = os.path.normpath(path)
    token = f"{os.sep}{marker}{os.sep}"
    if token in norm:
        rel = norm.split(token, 1)[1]
        candidate = os.path.join(marker, rel)
        if os.path.isfile(candidate):
            return candidate
    return path


def _load_grasp_json(args) -> dict[str, Any]:
    _require(args.grasp_json is not None, "--grasp_json is required for model_h5")
    _require(os.path.isfile(args.grasp_json), f"grasp json not found: {args.grasp_json}")
    with open(args.grasp_json, "r") as f:
        meta = json.load(f)
    if not isinstance(meta, dict):
        raise ValueError("--grasp_json must contain a JSON object")
    return meta


def _resolve_model_json_inputs(args) -> dict[str, Any]:
    meta = _load_grasp_json(args)
    mesh_meta = meta.get("mesh", {})
    if not isinstance(mesh_meta, dict):
        raise ValueError("model grasp JSON must contain object field 'mesh'")

    if args.mesh_path is None:
        mesh_path = mesh_meta.get("mesh_path_abs") or ""
        if not os.path.isfile(mesh_path):
            mesh_rel = mesh_meta.get("mesh_path")
            if mesh_rel:
                mesh_path = os.path.join(args.acronym_root, mesh_rel)
        if not os.path.isfile(mesh_path):
            raise FileNotFoundError(
                "Could not resolve mesh from JSON. Provide --mesh_path explicitly."
            )
        args.mesh_path = mesh_path
    if args.mesh_scale == 1.0 and "scale" in mesh_meta:
        args.mesh_scale = float(mesh_meta["scale"])

    if args.view_npz is None and meta.get("points"):
        args.view_npz = _resolve_repo_relative_path(str(meta["points"]), marker="data/out")
    if args.grasp_h5 is None:
        h5_path = meta.get("h5")
        if h5_path:
            candidate = str(h5_path)
            if not os.path.isfile(candidate):
                candidate = os.path.join(os.path.dirname(args.grasp_json), os.path.basename(candidate))
            args.grasp_h5 = candidate
    return meta


def _scaled_mesh_mean(mesh_path: str, mesh_scale: float) -> np.ndarray:
    mesh = trimesh.load(mesh_path, force="mesh")
    mesh.apply_scale(float(mesh_scale))
    return np.asarray(mesh.vertices.mean(axis=0), dtype=np.float64)


def _slice_order(indices: np.ndarray, *, grasp_index: int, top_k: int) -> np.ndarray:
    start = max(int(grasp_index), 0)
    k = max(int(top_k), 1)
    if start >= len(indices):
        raise ValueError(
            f"--grasp_index {start} out of range for {len(indices)} available grasps"
        )
    return indices[start : start + k]


def _read_h5_array(f: h5py.File, key: str, *, required: bool = True) -> np.ndarray | None:
    if key not in f:
        if required:
            raise ValueError(f"H5 missing required dataset {key!r}")
        return None
    return np.asarray(f[key])


def _read_h5_object_mass(path: str) -> float | None:
    try:
        with h5py.File(path, "r") as f:
            if "object/mass" in f:
                mass = float(np.asarray(f["object/mass"][()]))
                if np.isfinite(mass) and mass > 0:
                    return mass
    except OSError:
        return None
    return None


def _offset_hand_pose_along_approach(hand_pose: np.ndarray, offset_m: float) -> np.ndarray:
    T = validate_pose_se3(hand_pose, "hand_pose").copy()
    T[:3, 3] += float(offset_m) * T[:3, 2]
    return validate_pose_se3(T, "hand_pose_offset")


def _load_dataset_h5_candidates(args) -> list[dict[str, Any]]:
    _require(args.grasp_h5 is not None, "--grasp_h5 is required for dataset_h5")
    _require(os.path.isfile(args.grasp_h5), f"grasp h5 not found: {args.grasp_h5}")
    mesh_mean = _scaled_mesh_mean(args.mesh_path, args.mesh_scale)

    with h5py.File(args.grasp_h5, "r") as f:
        transforms = _read_h5_array(f, "grasps/transforms")
        success = _read_h5_array(f, "grasps/qualities/flex/object_in_gripper")
        widths = _read_h5_array(f, "grasps/widths", required=False)

    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError(f"Expected grasps/transforms shape (G,4,4), got {transforms.shape}")
    if len(transforms) == 0:
        raise ValueError(f"Dataset H5 contains zero grasps: {args.grasp_h5}")
    if widths is None:
        widths = np.full(len(transforms), 0.08, dtype=np.float64)
    widths = np.asarray(widths, dtype=np.float64)
    success = np.asarray(success).reshape(-1)
    if len(success) != len(transforms):
        raise ValueError("H5 success array length does not match transforms")
    if len(widths) != len(transforms):
        raise ValueError("H5 widths array length does not match transforms")

    if args.only_successful_dataset_grasps:
        indices = np.flatnonzero(success > 0)
        if len(indices) == 0:
            raise ValueError(f"No successful dataset grasps in {args.grasp_h5}")
    else:
        indices = np.arange(len(transforms), dtype=np.int64)
    selected = _slice_order(indices, grasp_index=args.grasp_index, top_k=args.top_k)

    out: list[dict[str, Any]] = []
    for rank, idx in enumerate(selected):
        T_marker_obj = validate_pose_se3(transforms[int(idx)], "dataset_h5_marker_pose").copy()
        T_marker_obj[:3, 3] -= mesh_mean
        T_hand_obj = marker_to_hand_pose(T_marker_obj)
        T_hand_obj[:3, 3] += float(args.h5_hand_depth_offset_m) * T_hand_obj[:3, 2]
        out.append(
            {
                "kind": "hand_object",
                "hand_pose_object": T_hand_obj,
                "width_m": float(widths[int(idx)]),
                "meta": {
                    "source": "dataset_h5",
                    "grasp_idx": int(idx),
                    "rank": int(rank),
                    "dataset_success": int(success[int(idx)] > 0),
                    "score": float(success[int(idx)] > 0),
                    "h5_hand_depth_offset_m": float(args.h5_hand_depth_offset_m),
                    "grasp_h5": os.path.abspath(args.grasp_h5),
                },
            }
        )
    return out


def _load_model_h5_candidates(
    args,
    json_meta: dict[str, Any],
    *,
    obj_xy: tuple[float, float],
) -> list[dict[str, Any]]:
    _require(args.grasp_h5 is not None, "--grasp_h5 is required for model_h5")
    _require(os.path.isfile(args.grasp_h5), f"grasp h5 not found: {args.grasp_h5}")
    _require(args.view_npz is not None, "model_h5 requires a view npz from JSON or --view_npz")
    npz_data = _load_npz_dict(args.view_npz)
    camera_pose = np.asarray(npz_data.get("camera_pose", np.eye(4)), dtype=np.float64)

    with h5py.File(args.grasp_h5, "r") as f:
        transforms = _read_h5_array(f, "grasps/transforms")
        widths = _read_h5_array(f, "grasps/widths")
        scores = _read_h5_array(f, "grasps/scores")

    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError(f"Expected grasps/transforms shape (G,4,4), got {transforms.shape}")
    if len(transforms) == 0:
        thresh = json_meta.get("score_thresh", "unknown")
        raise ValueError(
            f"Model H5 contains zero grasps: {args.grasp_h5}. "
            f"Current score_thresh={thresh}. Re-export with a lower score_thresh "
            "or enable a top-1 fallback export before running simulation."
        )
    widths = np.asarray(widths, dtype=np.float64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(widths) != len(transforms) or len(scores) != len(transforms):
        raise ValueError("Model H5 widths/scores lengths must match transforms")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Model H5 scores contain non-finite values")

    order = np.argsort(-scores)
    selected = _slice_order(order, grasp_index=args.grasp_index, top_k=args.top_k)
    frame = str(json_meta.get("frame", "input_point_cloud"))
    if frame != "input_point_cloud":
        raise ValueError(
            f"Unsupported model_h5 frame {frame!r}; expected 'input_point_cloud'"
        )

    out: list[dict[str, Any]] = []
    for rank, idx in enumerate(selected):
        T_marker_cam = validate_pose_se3(transforms[int(idx)], "model_h5_marker_pose")
        T_marker_world = cam_centred_to_world(
            T_marker_cam, camera_pose, np.zeros(3, dtype=np.float64)
        )
        T_marker_world = grasp_world_to_mujoco(
            T_marker_world,
            obj_z_offset=0.0,
            obj_xy_offset=obj_xy,
        )
        out.append(
            {
                "kind": "hand_world",
                "hand_pose_world": _offset_hand_pose_along_approach(
                    marker_to_hand_pose(T_marker_world),
                    float(args.h5_hand_depth_offset_m),
                ),
                "width_m": float(widths[int(idx)]),
                "meta": {
                    "source": "model_h5",
                    "grasp_idx": int(idx),
                    "rank": int(rank),
                    "score": float(scores[int(idx)]),
                    "h5_hand_depth_offset_m": float(args.h5_hand_depth_offset_m),
                    "grasp_h5": os.path.abspath(args.grasp_h5),
                    "grasp_json": os.path.abspath(args.grasp_json),
                    "view_npz": os.path.abspath(args.view_npz),
                },
            }
        )
    return out


def _validate_model_h5_nonempty(args, json_meta: dict[str, Any]) -> None:
    _require(args.grasp_h5 is not None, "--grasp_h5 is required for model_h5")
    _require(os.path.isfile(args.grasp_h5), f"grasp h5 not found: {args.grasp_h5}")
    with h5py.File(args.grasp_h5, "r") as f:
        transforms = _read_h5_array(f, "grasps/transforms")
        _read_h5_array(f, "grasps/widths")
        _read_h5_array(f, "grasps/scores")
    if len(transforms) == 0:
        thresh = json_meta.get("score_thresh", "unknown")
        raise ValueError(
            f"Model H5 contains zero grasps: {args.grasp_h5}. "
            f"Current score_thresh={thresh}. Re-export with a lower score_thresh "
            "or enable a top-1 fallback export before running simulation."
        )


def _prepare_inputs(args) -> dict[str, Any]:
    json_meta: dict[str, Any] = {}
    if args.source in CONTACT_FRAME_SOURCES:
        args.view_npz = _select_view_npz(args)
        if not args.mesh_path:
            args.mesh_path, args.mesh_scale = _resolve_mesh_and_scale_from_manifest(args)
    elif args.source == "dataset_h5":
        _require(args.grasp_h5 is not None, "--grasp_h5 is required for dataset_h5")
        if not args.mesh_path:
            args.mesh_path, args.mesh_scale = _resolve_mesh_and_scale_from_grasp_h5(args)
        args.object_mass_kg = _read_h5_object_mass(args.grasp_h5)
    elif args.source == "model_h5":
        json_meta = _resolve_model_json_inputs(args)
        _require(args.grasp_h5 is not None, "--grasp_h5 is required for model_h5")
        _require(os.path.isfile(args.grasp_h5), f"grasp h5 not found: {args.grasp_h5}")
        _validate_model_h5_nonempty(args, json_meta)
        args.object_mass_kg = _read_h5_object_mass(args.grasp_h5)
    else:
        raise ValueError(f"Unsupported source {args.source!r}")
    _require(args.mesh_path is not None, "--mesh_path could not be resolved")
    _require(os.path.isfile(args.mesh_path), f"mesh not found: {args.mesh_path}")
    return json_meta


def _candidate_plan(
    candidate: dict[str, Any],
    T_obj_world: np.ndarray,
    cfg: RetargetConfig,
) -> tuple[Any, np.ndarray, np.ndarray, str]:
    kind = candidate["kind"]
    width = float(candidate["width_m"])
    if kind == "contact_world":
        T_contact_world = validate_pose_se3(candidate["contact_pose_world"], "contact_pose_world")
        T_contact_obj = np.linalg.inv(T_obj_world) @ T_contact_world
        grasp = GraspSpec(
            contact_pose_SE3=T_obj_world @ T_contact_obj,
            width_m=width,
            source_meta={**candidate["meta"], "grasp_pose_frame": "contact_world"},
        )
        plan = build_retarget_plan(grasp, cfg=cfg)
        return plan, T_contact_obj, plan.grasp_pose, "contact_to_hand_pose"

    if kind == "hand_object":
        T_hand_obj = validate_pose_se3(candidate["hand_pose_object"], "hand_pose_object")
        T_hand_world = validate_pose_se3(T_obj_world @ T_hand_obj, "hand_pose_world")
        plan = build_retarget_plan_from_hand_pose(T_hand_world, cfg=cfg)
        return plan, T_hand_obj, T_hand_world, "object_local_marker_to_hand_pose"

    if kind == "hand_world":
        T_hand_world = validate_pose_se3(candidate["hand_pose_world"], "hand_pose_world")
        T_hand_obj = np.linalg.inv(T_obj_world) @ T_hand_world
        plan = build_retarget_plan_from_hand_pose(T_hand_world, cfg=cfg)
        return plan, T_hand_obj, T_hand_world, "input_point_cloud_marker_to_hand_pose"

    raise ValueError(f"Unsupported candidate kind {kind!r}")


def _preview_candidate_set(
    *,
    candidates: list[dict[str, Any]],
    T_obj_world: np.ndarray,
    cfg: RetargetConfig,
    object_visual_path: str,
    title: str,
    color: list[int],
) -> None:
    grasps: list[dict[str, Any]] = []
    for candidate in candidates:
        plan, _T_preview_obj, _T_exec_world, _frame_desc = _candidate_plan(
            candidate, T_obj_world, cfg
        )
        meta = candidate["meta"]
        grasps.append(
            {
                "executed_hand_pose_world": plan.grasp_pose,
                "color": color,
                "name": f"{meta.get('source')}_{meta.get('rank')}_{meta.get('grasp_idx')}",
            }
        )
    show_grasp_set_preview(
        object_obj_path=object_visual_path,
        object_pose_world=T_obj_world,
        grasps=grasps,
        title=title,
        block=True,
    )


def _candidate_preview_key(candidate: dict[str, Any]) -> int | None:
    grasp_idx = candidate.get("meta", {}).get("grasp_idx")
    if grasp_idx is None:
        return None
    try:
        return int(grasp_idx)
    except (TypeError, ValueError):
        return None


def _candidate_preview_grasps(
    *,
    candidates: list[dict[str, Any]],
    T_obj_world: np.ndarray,
    cfg: RetargetConfig,
    color: list[int],
    highlight_keys: set[int] | None = None,
    highlight_color: list[int] | None = None,
    tube_radius: float = 0.001,
    highlight_tube_radius: float = 0.0025,
) -> list[dict[str, Any]]:
    grasps: list[dict[str, Any]] = []
    highlight_keys = highlight_keys or set()
    for candidate in candidates:
        plan, _T_preview_obj, _T_exec_world, _frame_desc = _candidate_plan(
            candidate, T_obj_world, cfg
        )
        meta = candidate["meta"]
        key = _candidate_preview_key(candidate)
        marker_color = (
            highlight_color
            if key is not None and key in highlight_keys and highlight_color is not None
            else color
        )
        marker_radius = (
            highlight_tube_radius
            if key is not None and key in highlight_keys and highlight_color is not None
            else tube_radius
        )
        grasps.append(
            {
                "executed_hand_pose_world": plan.grasp_pose,
                "color": marker_color,
                "tube_radius": marker_radius,
                "name": f"{meta.get('source')}_{meta.get('rank')}_{meta.get('grasp_idx')}",
            }
        )
    return grasps


def _preview_all_count(args) -> int:
    if int(args.preview_all_limit) <= 0:
        return 1_000_000_000
    return max(int(args.preview_all_limit), int(args.grasp_index) + int(args.top_k))


def _run_executor(args, mj_model, mj_data, plan, width_m: float, cfg: RetargetConfig):
    executor = SimpleIKGraspExecutor(mj_model, mj_data, plan, width_m, cfg=cfg)
    steps = 0

    if args.no_viewer:
        while executor.phase != ExecPhase.DONE and steps < args.max_steps:
            executor.step()
            steps += 1
        if steps >= args.max_steps:
            executor.force_finish(f"max_steps reached ({args.max_steps})")
        return executor, steps

    viewer = mujoco.viewer.launch_passive(
        mj_model,
        mj_data,
        show_left_ui=bool(args.show_viewer_ui),
        show_right_ui=bool(args.show_viewer_ui),
    )
    try:
        _configure_viewer_render_defaults(viewer)
        if executor.phase == ExecPhase.DONE:
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.03)
        else:
            while (
                viewer.is_running()
                and executor.phase != ExecPhase.DONE
                and steps < args.max_steps
            ):
                executor.step()
                viewer.sync()
                steps += 1
                time.sleep(cfg.dt)

            if steps >= args.max_steps:
                executor.force_finish(f"max_steps reached ({args.max_steps})")

            if viewer.is_running():
                viewer.sync()

            if args.hold_viewer and viewer.is_running():
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.03)
    finally:
        viewer.close()
    return executor, steps


def _print_trial_result(candidate: dict[str, Any], executor, steps: int) -> None:
    meta = candidate["meta"]
    print("\nExecution Result")
    print(f"grasp_idx: {meta.get('grasp_idx')}")
    print(f"rank: {meta.get('rank')}")
    print(f"score: {float(meta.get('score', 0.0)):.5f}")
    print(f"success: {executor.result.success}")
    print(f"phase: {executor.result.phase_reached}")
    print(f"steps: {steps}")
    print(f"pose_error_m: {executor.result.pose_error_m:.5f}")
    print(f"pose_error_deg: {executor.result.pose_error_deg:.2f}")
    print(f"hand_lift_height_m: {executor.result.lift_height_m:.5f}")
    print(f"object_lift_m: {executor.result.object_lift_m:.5f}")
    print(f"object_z_initial: {executor.result.object_z_initial:.5f}")
    print(f"object_z_final: {executor.result.object_z_final:.5f}")
    if executor.result.failure_reason:
        print(f"failure_reason: {executor.result.failure_reason}")
    if executor.result.error:
        print(f"error: {executor.result.error}")


def _run(args) -> int:
    json_meta = _prepare_inputs(args)

    mj_model, mj_data, _obj_z, obj_xy, object_visual_path = build_scene(
        args.mesh_path,
        args.mesh_scale,
        object_dynamic=not args.static_object,
        object_mass_kg=getattr(args, "object_mass_kg", None),
    )
    T_obj_world = _target_object_pose_world(mj_model, mj_data)

    pred_backbone = None
    pred_source_name = None
    if args.source == "labels":
        candidates = _load_label_candidates(args, obj_xy)
    elif args.source == "pred_cgn":
        pred_backbone = "pn2"
        pred_source_name = "pred_cgn"
        candidates = _load_pred_candidates(
            args, obj_xy, backbone=pred_backbone, source_name=pred_source_name
        )
    elif args.source == "pred_ptv3":
        pred_backbone = "ptv3"
        pred_source_name = "pred_ptv3"
        candidates = _load_pred_candidates(
            args, obj_xy, backbone=pred_backbone, source_name=pred_source_name
        )
    elif args.source == "dataset_h5":
        candidates = _load_dataset_h5_candidates(args)
    elif args.source == "model_h5":
        candidates = _load_model_h5_candidates(args, json_meta, obj_xy=obj_xy)
    else:
        raise ValueError(f"Unsupported source {args.source!r}")

    cfg = RetargetConfig()
    cfg.start_delay_s = max(float(args.start_delay_s), 0.0)
    cfg.pre_close_pause_s = max(float(args.pre_close_pause_s), 0.0)
    total_successes = 0
    suppress_individual_preview = False
    comparison_preview_shown = False

    if not args.skip_preview and args.compare_labels_preview and args.source in {"pred_cgn", "pred_ptv3"}:
        try:
            label_candidates = _load_label_candidates(args, obj_xy)
            label_preview_candidates = label_candidates
            model_preview_candidates = candidates
            left_label = f"GT labels: green selected top-{len(label_candidates)}"
            right_label = f"{args.source}: blue selected top-{len(candidates)}"

            if args.preview_all_grasps:
                preview_count = _preview_all_count(args)
                label_preview_candidates = _load_label_candidates(
                    args, obj_xy, grasp_index=0, top_k=preview_count
                )
                model_preview_candidates = _load_pred_candidates(
                    args,
                    obj_xy,
                    backbone=str(pred_backbone),
                    source_name=str(pred_source_name),
                    grasp_index=0,
                    top_k=preview_count,
                )
                left_label = (
                    f"GT labels: orange candidates, green selected top-{len(label_candidates)}"
                )
                right_label = (
                    f"{args.source}: orange candidates, blue selected top-{len(candidates)}"
                )

            label_top_keys = {
                key for key in (_candidate_preview_key(c) for c in label_candidates) if key is not None
            }
            model_top_keys = {
                key for key in (_candidate_preview_key(c) for c in candidates) if key is not None
            }
            show_grasp_comparison_preview(
                object_obj_path=object_visual_path,
                object_pose_world=T_obj_world,
                left_grasps=_candidate_preview_grasps(
                candidates=label_preview_candidates,
                T_obj_world=T_obj_world,
                cfg=cfg,
                color=[255, 170, 30, 70],
                highlight_keys=label_top_keys,
                highlight_color=[40, 220, 60],
                tube_radius=0.002,
                highlight_tube_radius=0.0025,
            ),
            right_grasps=_candidate_preview_grasps(
                candidates=model_preview_candidates,
                T_obj_world=T_obj_world,
                cfg=cfg,
                color=[255, 170, 30, 70],
                highlight_keys=model_top_keys,
                highlight_color=[60, 130, 255],
                tube_radius=0.002,
                highlight_tube_radius=0.0025,
            ),
                left_label=left_label,
                right_label=right_label,
            )
            suppress_individual_preview = True
            comparison_preview_shown = True
        except ValueError as exc:
            print(
                f"Warning: --compare_labels_preview skipped because labels could not be loaded: {exc}",
                file=sys.stderr,
            )

    if not args.skip_preview and not comparison_preview_shown and len(candidates) > 1:
        preview_color = [60, 130, 255] if args.source in {"pred_cgn", "pred_ptv3", "model_h5"} else [40, 220, 60]
        _preview_candidate_set(
            candidates=candidates,
            T_obj_world=T_obj_world,
            cfg=cfg,
            object_visual_path=object_visual_path,
            title=f"{args.source} top-{len(candidates)} preview",
            color=preview_color,
        )
        suppress_individual_preview = True

    for trial_idx, candidate in enumerate(candidates):
        if trial_idx > 0:
            mj_model, mj_data, _obj_z, obj_xy, object_visual_path = build_scene(
                args.mesh_path,
                args.mesh_scale,
                object_dynamic=not args.static_object,
                object_mass_kg=getattr(args, "object_mass_kg", None),
            )
            T_obj_world = _target_object_pose_world(mj_model, mj_data)

        plan, T_preview_obj, T_exec_world, frame_desc = _candidate_plan(
            candidate, T_obj_world, cfg
        )
        meta = candidate["meta"]
        consistency = compute_pose_error(T_exec_world, plan.grasp_pose)
        rel_consistency = compute_pose_error(
            np.linalg.inv(T_obj_world) @ T_exec_world,
            np.linalg.inv(T_obj_world) @ plan.grasp_pose,
        )

        print(
            "\nSelected grasp: "
            f"source={args.source}, "
            f"trial={trial_idx + 1}/{len(candidates)}, "
            f"grasp_idx={meta.get('grasp_idx')}, "
            f"rank={meta.get('rank')}, "
            f"score={float(meta.get('score', 0.0)):.5f}, "
            f"width_m={float(candidate['width_m']):.5f}, "
            f"frame={frame_desc}"
        )
        print(
            "Pose consistency: "
            f"world_err_m={consistency.target_vs_live_translation_m:.6e}, "
            f"world_err_deg={consistency.target_vs_live_rotation_deg:.6f}, "
            f"obj_rel_err_m={rel_consistency.target_vs_live_translation_m:.6e}, "
            f"obj_rel_err_deg={rel_consistency.target_vs_live_rotation_deg:.6f}"
        )

        if not args.skip_preview and not suppress_individual_preview:
            title = (
                f"{args.source} grasp {meta.get('grasp_idx')} "
                f"(trial {trial_idx + 1}/{len(candidates)}, executed hand pose)"
            )
            show_grasp_preview(
                object_obj_path=object_visual_path,
                object_pose_world=T_obj_world,
                grasp_contact_pose_object=T_preview_obj,
                executed_hand_pose_world=plan.grasp_pose,
                grasp_width_m=float(candidate["width_m"]),
                contact_to_wrist_m=cfg.contact_to_wrist_m,
                title=title,
                preview=("cgt" if candidate["kind"] != "contact_world" else args.trimesh_preview),
                block=True,
            )

        executor, steps = _run_executor(
            args, mj_model, mj_data, plan, float(candidate["width_m"]), cfg
        )
        _print_trial_result(candidate, executor, steps)
        total_successes += int(bool(executor.result.success))

    if len(candidates) > 1:
        print("\nBatch Summary")
        print(f"trials: {len(candidates)}")
        print(f"successes: {total_successes}")
        print(f"object_lift_success_rate: {total_successes / max(len(candidates), 1):.1%}")

    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Grasp preview and single-IK MuJoCo execution"
    )
    p.add_argument(
        "--source",
        default="labels",
        choices=["labels", "pred_cgn", "pred_ptv3", "dataset_h5", "model_h5"],
        help="Grasp source mode",
    )
    p.add_argument("--mesh_path", default=None, help="Path to object mesh (.obj)")
    p.add_argument("--mesh_scale", type=float, default=1.0, help="Mesh scale")
    p.add_argument("--max_steps", type=int, default=15000, help="Execution step cap")
    p.add_argument(
        "--start_delay_s",
        type=float,
        default=5.0,
        help="Initial wait time before approach motion starts",
    )
    p.add_argument(
        "--pre_close_pause_s",
        type=float,
        default=1.0,
        help="Pause at the reached/reoriented grasp pose before closing fingers",
    )
    p.add_argument("--no_viewer", action="store_true", help="Run headless without MuJoCo window")
    p.add_argument(
        "--hold_viewer",
        action="store_true",
        help="Keep viewer open after execution until window closes",
    )
    p.add_argument(
        "--show_viewer_ui",
        action="store_true",
        help="Show MuJoCo UI panes (default: hidden for clean view)",
    )
    p.add_argument("--skip_preview", action="store_true", help=argparse.SUPPRESS)

    # Model / labels modes
    p.add_argument("--view_npz", type=str, default=None, help="Path to per-view .npz")
    p.add_argument("--data_dir", type=str, default="data/out", help="Generated views root")
    p.add_argument(
        "--label_conf_thresh",
        type=float,
        default=0.5,
        help="Minimum generated-label confidence for labels/GT comparison modes",
    )
    p.add_argument(
        "--manifest", type=str, default="data/acronym/manifest.json", help="ACRONYM manifest JSON"
    )
    p.add_argument("--acronym_root", type=str, default="data/acronym", help="ACRONYM root dir")
    p.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Split for auto view selection")
    p.add_argument("--category", type=str, default=None, help="Optional category filter for auto view selection")
    p.add_argument("--mesh_hash", type=str, default=None, help="Optional mesh_hash filter for auto selection")
    p.add_argument("--view_index", type=int, default=0, help="Index among matching auto-selected views")
    p.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint for pred_*")
    p.add_argument(
        "--ptv3_cpe_mode",
        type=str,
        default="auto",
        choices=["auto", "knn", "conv1d", "sparse3d"],
        help="PTv3 CPE mode for checkpoint loading; auto reads checkpoint config",
    )
    p.add_argument("--grasp_h5", type=str, default=None, help="Dataset or model-exported grasp H5")
    p.add_argument("--grasp_json", type=str, default=None, help="Model-export metadata JSON")
    p.add_argument(
        "--grasp_index",
        type=int,
        default=0,
        help="Starting grasp index/rank for ranked label, prediction, and H5 modes",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of ranked label, prediction, or H5 grasps to simulate",
    )
    p.add_argument(
        "--h5_hand_depth_offset_m",
        type=float,
        default=0.03,
        help=(
            "Calibration offset applied to H5 marker-derived Panda hand poses "
            "along local +z/approach. Menagerie Panda replay needs about 0.03m "
            "to place fingertip pads at ACRONYM contact depth."
        ),
    )
    p.add_argument(
        "--only_successful_dataset_grasps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For dataset_h5, filter to ACRONYM grasps marked object_in_gripper",
    )
    p.add_argument(
        "--device",
        type=str,
        default=("cuda" if _has_cuda() else "cpu"),
        help="Torch device for pred_* modes",
    )
    p.add_argument(
        "--eval_seed",
        type=int,
        default=0,
        help="Random seed for deterministic checkpoint inference in pred_* modes",
    )
    p.add_argument(
        "--static_object",
        action="store_true",
        help="Pin the target object; default is free (physics) with a free joint",
    )
    p.add_argument(
        "--trimesh-preview",
        type=str,
        default="cgt",
        choices=["cgt", "acronym"],
        help='Trimesh: "cgt" = parallel wireframe; "acronym" = NVlabs marker mesh',
    )
    p.add_argument(
        "--compare_labels_preview",
        action="store_true",
        help=(
            "For pred_cgn/pred_ptv3, show generated-label top-k and model "
            "top-k side-by-side in one Trimesh window."
        ),
    )
    p.add_argument(
        "--preview_all_grasps",
        action="store_true",
        help=(
            "With --compare_labels_preview, draw broader GT/model candidates in "
            "orange and highlight the selected top-k in green/blue."
        ),
    )
    p.add_argument(
        "--preview_all_limit",
        type=int,
        default=40,
        help=(
            "Maximum orange background candidates per side for --preview_all_grasps; "
            "use 0 to draw every valid candidate."
        ),
    )
    return p


def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _configure_viewer_render_defaults(viewer) -> None:
    """Force classic Menagerie-like rendering defaults."""
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = 1
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_HAZE] = 1
    if hasattr(mujoco.mjtRndFlag, "mjRND_REFLECTION"):
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        return _run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
