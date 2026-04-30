"""Standalone Trimesh preview for one generated grasp.

`preview='cgt'` (default): NVlabs parallel-wire marker at the hand retarget pose.
`preview='acronym'`: NVlabs marker-style mesh for dataset comparison.
"""

from __future__ import annotations

import trimesh
import trimesh.util

import numpy as np

from eval.ik_retarget import contact_to_hand_pose
from eval.utils import make_grasp_pose


def _hand_to_marker_pose(T_hand: np.ndarray) -> np.ndarray:
    """Convert hand-frame pose (y=baseline, z=approach) to NVlabs marker frame (x=baseline, z=approach)."""
    T = np.asarray(T_hand, dtype=np.float64).copy().reshape(4, 4)
    Rz_pos90 = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    T[:3, :3] = T[:3, :3] @ Rz_pos90
    return T


def create_gripper_marker(
    color: list | None = None, tube_radius: float = 0.002, sections: int = 6
) -> trimesh.Trimesh:
    """Same as ``acronym_tools.create_gripper_marker`` (NVlabs/acronym, MIT)."""
    if color is None:
        color = [0, 0, 255]
    radius = max(float(tube_radius), 1.0e-4)
    cfl = trimesh.creation.cylinder(
        radius=radius,
        sections=sections,
        segment=[
            [4.10000000e-02, -7.27595772e-12, 6.59999996e-02],
            [4.10000000e-02, -7.27595772e-12, 1.12169998e-01],
        ],
    )
    cfr = trimesh.creation.cylinder(
        radius=radius,
        sections=sections,
        segment=[
            [-4.100000e-02, -7.27595772e-12, 6.59999996e-02],
            [-4.100000e-02, -7.27595772e-12, 1.12169998e-01],
        ],
    )
    cb1 = trimesh.creation.cylinder(
        radius=radius, sections=sections, segment=[[0, 0, 0], [0, 0, 6.59999996e-02]]
    )
    cb2 = trimesh.creation.cylinder(
        radius=radius,
        sections=sections,
        segment=[[-4.100000e-02, 0, 6.59999996e-02], [4.100000e-02, 0, 6.59999996e-02]],
    )

    tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl])
    # Per-face colors (3 values broadcast correctly in trimesh; 4 is safer for viewers)
    rgba = np.array(list(color) + [255], dtype=np.uint8) if len(color) == 3 else np.asarray(color, dtype=np.uint8)
    if rgba.size != 4:
        rgba = np.array([0, 255, 0, 255], dtype=np.uint8)
    tmp.visual.face_colors = np.tile(rgba, (len(tmp.faces), 1))
    return tmp


def show_grasp_preview(
    *,
    object_obj_path: str,
    object_pose_world: np.ndarray,
    grasp_contact_pose_object: np.ndarray,
    executed_hand_pose_world: np.ndarray | None = None,
    grasp_width_m: float,
    contact_to_wrist_m: float = 0.066,
    title: str = "Grasp preview",
    preview: str = "cgt",
    block: bool = True,
) -> None:
    """Object mesh + one grasp.

    Parameters
    ----------
    preview
        - ``"cgt"`` — NVlabs marker mesh at ``contact_to_hand_pose`` (executed hand pose).
        - ``"acronym"`` — NVlabs four-cylinder mesh at ``make_grasp_pose(…, width=)``.
    """
    T_obj = np.asarray(object_pose_world, dtype=np.float64)
    T_co = np.asarray(grasp_contact_pose_object, dtype=np.float64)
    T_contact_world = T_obj @ T_co

    mesh = trimesh.load(object_obj_path, force="mesh")
    # One RGBA for all faces breaks many GL viewers (flicker / z-artifacts when zooming);
    # tile to (n_faces, 4). Slight alpha so a wrist marker inside the bulk stays visible.
    n_f = len(mesh.faces)
    obj_rgba = np.array([165, 200, 230, 220], dtype=np.uint8)
    mesh.visual.face_colors = np.tile(obj_rgba, (n_f, 1))

    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name="object", transform=T_obj)

    if preview == "cgt":
        if executed_hand_pose_world is None:
            T_hand = contact_to_hand_pose(
                T_contact_world, grasp_width_m, contact_to_wrist_m=contact_to_wrist_m
            )
        else:
            T_hand = np.asarray(executed_hand_pose_world, dtype=np.float64).reshape(4, 4)
        T_marker = _hand_to_marker_pose(T_hand)
        g = create_gripper_marker(color=[40, 220, 60])
        g.apply_transform(T_marker)
        scene.add_geometry(g, geom_name="executed_hand_pose")

    if preview == "acronym":
        c = T_co[:3, 3]
        a = T_co[:3, 2]
        b = T_co[:3, 1]
        T_hand_obj = make_grasp_pose(c, a, b, width=float(grasp_width_m))
        T_world = T_obj @ T_hand_obj
        T_marker = _hand_to_marker_pose(T_world)
        g = create_gripper_marker(color=[0, 255, 0])
        g.apply_transform(T_marker)
        scene.add_geometry(g, geom_name="gripper_marker_nvlabs")

    if preview not in ("cgt", "acronym"):
        raise ValueError('preview must be "cgt" or "acronym"')

    print(f"Trimesh preview: close the window to continue.\n  {title}")
    scene.show(block=block)


def show_grasp_set_preview(
    *,
    object_obj_path: str,
    object_pose_world: np.ndarray,
    grasps: list[dict],
    title: str = "Top-k grasp preview",
    block: bool = True,
) -> None:
    """Object mesh + multiple executed hand poses in one Trimesh scene.

    Each grasp dict should contain ``executed_hand_pose_world`` and may include
    ``color`` and ``name`` fields. This is meant for inspecting ranked model or
    label candidates before MuJoCo executes them one-by-one.
    """
    T_obj = np.asarray(object_pose_world, dtype=np.float64)
    mesh = trimesh.load(object_obj_path, force="mesh")
    n_f = len(mesh.faces)
    obj_rgba = np.array([165, 200, 230, 210], dtype=np.uint8)
    mesh.visual.face_colors = np.tile(obj_rgba, (n_f, 1))

    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name="object", transform=T_obj)

    for i, grasp in enumerate(grasps):
        T_hand = np.asarray(grasp["executed_hand_pose_world"], dtype=np.float64).reshape(4, 4)
        T_marker = _hand_to_marker_pose(T_hand)
        color = grasp.get("color", [40, 220, 60])
        g = create_gripper_marker(
            color=color,
            tube_radius=float(grasp.get("tube_radius", 0.002)),
            sections=int(grasp.get("sections", 6)),
        )
        g.apply_transform(T_marker)
        name = str(grasp.get("name", f"grasp_{i}"))
        scene.add_geometry(g, geom_name=name)

    print(f"Trimesh preview: close the window to continue.\n  {title}")
    scene.show(block=block)


def show_grasp_comparison_preview(
    *,
    object_obj_path: str,
    object_pose_world: np.ndarray,
    left_grasps: list[dict],
    right_grasps: list[dict],
    left_label: str = "GT labels",
    right_label: str = "Model predictions",
    separation_m: float = 0.35,
    block: bool = True,
) -> None:
    """Side-by-side object mesh + two ranked grasp sets.

    The object is duplicated left/right so ground-truth label grasps and model
    grasps can be inspected in the same view without overlapping each other.
    """
    T_obj = np.asarray(object_pose_world, dtype=np.float64)
    base_mesh = trimesh.load(object_obj_path, force="mesh")
    obj_rgba = np.array([165, 200, 230, 210], dtype=np.uint8)
    base_mesh.visual.face_colors = np.tile(obj_rgba, (len(base_mesh.faces), 1))

    scene = trimesh.Scene()

    def add_side(
        *,
        side_name: str,
        x_offset: float,
        grasps: list[dict],
        default_color: list[int],
    ) -> None:
        T_shift = np.eye(4, dtype=np.float64)
        T_shift[0, 3] = float(x_offset)

        mesh = base_mesh.copy()
        scene.add_geometry(mesh, geom_name=f"{side_name}_object", transform=T_shift @ T_obj)

        for i, grasp in enumerate(grasps):
            T_hand = np.asarray(grasp["executed_hand_pose_world"], dtype=np.float64).reshape(4, 4)
            T_marker = _hand_to_marker_pose(T_shift @ T_hand)
            color = grasp.get("color", default_color)
            g = create_gripper_marker(
                color=color,
                tube_radius=float(grasp.get("tube_radius", 0.002)),
                sections=int(grasp.get("sections", 6)),
            )
            g.apply_transform(T_marker)
            name = str(grasp.get("name", f"{side_name}_grasp_{i}"))
            scene.add_geometry(g, geom_name=name)

    add_side(
        side_name="left_gt",
        x_offset=-0.5 * float(separation_m),
        grasps=left_grasps,
        default_color=[255, 170, 30],
    )
    add_side(
        side_name="right_model",
        x_offset=0.5 * float(separation_m),
        grasps=right_grasps,
        default_color=[60, 130, 255],
    )

    print(
        "Trimesh comparison preview: close the window to continue.\n"
        f"  LEFT = {left_label}\n"
        f"  RIGHT = {right_label}"
    )
    scene.show(block=block)
