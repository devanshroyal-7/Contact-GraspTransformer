"""Camera/world transforms, grasp-frame construction, and Panda width helpers."""

from __future__ import annotations

import numpy as np

from models.cgn_heads import PANDA_BASELINE_DIST


PANDA_MAX_WIDTH = 0.08
PANDA_CTRL_MAX = 255.0


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


def make_grasp_pose(
    contact: np.ndarray,
    approach: np.ndarray,
    baseline: np.ndarray,
    width: float | None = None,
) -> np.ndarray:
    """Construct a contact/gripper pose from CGN point labels or predictions.

    Local axes follow the evaluator convention:
    - x: ``baseline x approach``
    - y: gripper baseline/opening direction
    - z: gripper approach direction

    When ``width`` is supplied, the translation is shifted from the contact
    point to the CGN/Panda wrist keypoint used by the ADD-S loss.
    """
    a = _normalize(approach)
    b = _normalize(baseline)
    b = _normalize(b - np.dot(b, a) * a)
    x = _normalize(np.cross(b, a))

    t = np.asarray(contact, dtype=np.float64).copy()
    if width is not None:
        half_width = float(np.clip(width, 0.0, PANDA_MAX_WIDTH)) * 0.5
        t = t + half_width * b + float(PANDA_BASELINE_DIST) * a

    T = np.eye(4, dtype=np.float64)
    T[:3, 0] = x
    T[:3, 1] = b
    T[:3, 2] = a
    T[:3, 3] = t
    return T


def cam_centred_to_world(
    grasp_cam: np.ndarray,
    camera_pose: np.ndarray,
    pc_mean: np.ndarray,
    obj_pose: np.ndarray | None = None,
) -> np.ndarray:
    """Convert a mean-centred camera-frame grasp pose back to world frame.

    Generated ``data/out`` samples store ``camera_pose`` so its inverse maps
    directly from the mean-centred camera frame to the data-generation world.
    ``pc_mean`` and ``obj_pose`` are accepted for compatibility with older call
    sites; the mean is already baked into the saved camera pose.
    """
    _ = pc_mean
    c2w = np.linalg.inv(np.asarray(camera_pose, dtype=np.float64))
    G_world = c2w @ np.asarray(grasp_cam, dtype=np.float64)
    if obj_pose is not None:
        G_world = np.asarray(obj_pose, dtype=np.float64) @ G_world
    return G_world


def grasp_world_to_mujoco(
    grasp_world: np.ndarray,
    obj_z_offset: float = 0.0,
    obj_xy_offset: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Shift a data-generation world grasp into the MuJoCo object placement."""
    G = np.asarray(grasp_world, dtype=np.float64).copy()
    G[0, 3] += float(obj_xy_offset[0])
    G[1, 3] += float(obj_xy_offset[1])
    G[2, 3] += float(obj_z_offset)
    return G


def width_to_ctrl(width: float) -> float:
    """Convert gripper width in metres to Menagerie Panda actuator ctrl."""
    w = float(np.clip(width, 0.0, PANDA_MAX_WIDTH))
    return w / PANDA_MAX_WIDTH * PANDA_CTRL_MAX


def ctrl_to_width(ctrl: float) -> float:
    """Convert Menagerie Panda actuator ctrl to gripper width in metres."""
    c = float(np.clip(ctrl, 0.0, PANDA_CTRL_MAX))
    return c / PANDA_CTRL_MAX * PANDA_MAX_WIDTH


__all__ = [
    "PANDA_CTRL_MAX",
    "PANDA_MAX_WIDTH",
    "cam_centred_to_world",
    "ctrl_to_width",
    "grasp_world_to_mujoco",
    "make_grasp_pose",
    "width_to_ctrl",
]
