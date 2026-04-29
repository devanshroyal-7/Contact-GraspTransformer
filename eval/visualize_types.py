"""Shared grasp and retarget data types for visualization and execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def validate_pose_se3(pose: np.ndarray, name: str = "pose") -> np.ndarray:
    """Validate and return a float64 SE(3) matrix."""
    T = np.asarray(pose, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {T.shape}")
    if not np.all(np.isfinite(T)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(T[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-5):
        raise ValueError(f"{name} last row must be [0, 0, 0, 1]")

    R = T[:3, :3]
    RtR = R.T @ R
    det = np.linalg.det(R)
    if not np.allclose(RtR, np.eye(3), atol=1e-3):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(det, 1.0, atol=1e-3):
        raise ValueError(f"{name} rotation determinant must be 1, got {det:.6f}")
    return T


@dataclass
class GraspSpec:
    """Input grasp spec in canonical contact-frame representation.

    Convention (``make_grasp_pose`` / ACRONYM hdf5 column 0 as ``base_dirs`` in training data):
    - local x: b×a
    - local y: **baseline b** (finger opening, from ``base_dirs`` in .npz)
    - local z: **approach a**
    - translation: surface *contact* point
    """

    contact_pose_SE3: np.ndarray
    width_m: float
    source_meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.contact_pose_SE3 = validate_pose_se3(self.contact_pose_SE3, "contact_pose_SE3")
        self.width_m = float(self.width_m)
        if not np.isfinite(self.width_m):
            raise ValueError("width_m must be finite")


@dataclass
class RetargetPlan:
    """Target hand poses for the cleaned single-IK executor."""

    approach_pose: np.ndarray
    grasp_pose: np.ndarray
    lift_pose: np.ndarray

    def __post_init__(self) -> None:
        self.approach_pose = validate_pose_se3(self.approach_pose, "approach_pose")
        self.grasp_pose = validate_pose_se3(self.grasp_pose, "grasp_pose")
        self.lift_pose = validate_pose_se3(self.lift_pose, "lift_pose")


@dataclass
class PoseError:
    """Live tracking error between the current hand pose and target hand pose."""

    target_vs_live_translation_m: float
    target_vs_live_rotation_deg: float
