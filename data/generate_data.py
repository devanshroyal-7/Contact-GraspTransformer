"""
Unified single-object data generator for Contact-GraspNet training.

For each object in the manifest, renders depth from multiple camera views,
back-projects to point clouds, transforms ACRONYM grasp labels into the
same camera frame, and saves one .npz per view containing everything
needed for training AND visualization.

Output .npz keys:
    depth          (H, W)    float32   depth image in metres
    points         (N, 3)    float32   mean-centred point cloud (camera frame)
    confidence     (N,)      float32   per-point grasp score (0 or 1)
    approach_dirs  (N, 3)    float32   per-point approach direction
    base_dirs      (N, 3)    float32   per-point base / binormal direction
    widths         (N,)      float32   per-point grasp width
    camera_pose    (4, 4)    float64   camera pose (for reference)

Usage:
    python data/generate_data.py                                 # all 180 objects, 360 views each
    python data/generate_data.py --category Mug                  # just Mug (all its train+test meshes)
    python data/generate_data.py --splits train                  # only train meshes
    python data/generate_data.py --mesh_hash 2997f21fa426...     # one specific mesh
    python data/generate_data.py --n_views 5 --n_points 2048     # quick smoke test

Output layout (one file per view):
    <out_root>/<split>/<category>/<mesh_hash>/NNN.npz
"""

from __future__ import annotations

import os
import json
import argparse
import numpy as np
import h5py
import trimesh
import trimesh.transformations as tra
from scipy.spatial import KDTree

os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender

REALSENSE = dict(fx=616.365, fy=616.203, cx=310.259, cy=236.600,
                 width=640, height=480, znear=0.04, zfar=20.0)


# ─────────────────────────────── camera poses ─────────────────────────────────

def sample_camera_poses(n_views: int = 36,
                        distance_range=(0.55, 0.85),
                        elevation_deg=(20, 70)) -> list[np.ndarray]:
    """Camera poses on a view-sphere above the table (OpenGL convention)."""
    coord_tf = (tra.euler_matrix(np.pi / 2, 0, 0)
                @ tra.euler_matrix(0, np.pi / 2, 0))
    el_lo = np.deg2rad(elevation_deg[0])
    el_hi = np.deg2rad(elevation_deg[1])

    poses = []
    for az in np.linspace(0, 2 * np.pi, n_views, endpoint=False):
        el = np.random.uniform(el_lo, el_hi)
        dist = np.random.uniform(*distance_range)
        ext = np.eye(4)
        ext[0, 3] = dist
        cam = tra.euler_matrix(0, -el, az) @ ext @ coord_tf
        poses.append(cam)
    return poses


# ───────────────────────────── depth → point cloud ────────────────────────────

def depth_to_pointcloud(depth: np.ndarray,
                        fx: float, fy: float,
                        cx: float, cy: float) -> np.ndarray:
    """Back-project depth map to (M, 3) point cloud in the camera frame."""
    ys, xs = np.where(depth > 0)
    z = depth[ys, xs]
    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def regularize_pc(pc: np.ndarray, n: int) -> np.ndarray:
    """Sub- or over-sample *pc* to exactly *n* rows."""
    m = len(pc)
    if m == 0:
        return np.zeros((n, 3), dtype=np.float32)
    if m >= n:
        idx = np.random.choice(m, n, replace=False)
    else:
        idx = np.concatenate([np.arange(m), np.random.choice(m, n - m)])
    return pc[idx]


# ─────────────────────── coordinate-frame conversion ──────────────────────────

def opengl_to_opencv_cam(cam_pose_gl: np.ndarray) -> np.ndarray:
    """Convert an OpenGL camera pose to OpenCV convention (flip Y & Z cols)."""
    pose = cam_pose_gl.copy()
    pose[:3, 1] *= -1
    pose[:3, 2] *= -1
    return pose


def world_to_cam_matrix(cam_pose_gl: np.ndarray) -> np.ndarray:
    """4x4 transform that maps world-frame points into the OpenCV camera frame.

    Equivalent to:  F @ inv(cam_pose_gl)   where F = diag(1,-1,-1,1).
    """
    return np.linalg.inv(opengl_to_opencv_cam(cam_pose_gl))


# ──────────────────────────── scene construction ──────────────────────────────

def load_and_prepare_mesh(mesh_path: str, scale: float):
    """Load mesh, scale, centre.  Returns (trimesh, mesh_mean, obj_pose)."""
    mesh = trimesh.load(mesh_path, force="mesh")
    mesh.apply_scale(scale)
    mesh_mean = mesh.vertices.mean(axis=0)
    mesh.vertices -= mesh_mean
    return mesh, mesh_mean


def build_scene(mesh: trimesh.Trimesh, intr: dict,
                table_dims=(1.0, 1.2, 0.6)):
    """Place the (already centred) mesh on a table, return scene objects."""
    scene = pyrender.Scene()

    table = trimesh.creation.box(table_dims)
    scene.add(pyrender.Mesh.from_trimesh(table), name="table")

    obj_node = scene.add(pyrender.Mesh.from_trimesh(mesh), name="object")
    obj_pose = np.eye(4)
    obj_pose[2, 3] = table_dims[2] / 2 - mesh.bounds[0][2]
    scene.set_pose(obj_node, obj_pose)

    cam = pyrender.IntrinsicsCamera(intr["fx"], intr["fy"],
                                     intr["cx"], intr["cy"],
                                     intr["znear"], intr["zfar"])
    cam_node = scene.add(cam, pose=np.eye(4), name="camera")
    renderer = pyrender.OffscreenRenderer(intr["width"], intr["height"])

    return scene, cam_node, renderer, obj_pose, table_dims


# ──────────────────────────── grasp loading ───────────────────────────────────

def estimate_grasp_widths(transforms: np.ndarray,
                          mesh_vertices: np.ndarray,
                          success_mask: np.ndarray,
                          search_radius: float = 0.10) -> np.ndarray:
    """Estimate per-grasp closing widths from mesh geometry.

    For each successful grasp, finds mesh vertices near the TCP and measures
    the object's extent along the gripper baseline direction.  The default
    search_radius is large (0.10m) because the TCP sits at the gripper base
    frame, which is offset from the object surface by finger length.
    """
    G = len(transforms)
    widths = np.full(G, 0.08, dtype=np.float32)

    succ_idx = np.where(success_mask)[0]
    if len(succ_idx) == 0 or len(mesh_vertices) == 0:
        return widths

    tree = KDTree(mesh_vertices)
    for i in succ_idx:
        tcp = transforms[i, :3, 3]
        baseline = transforms[i, :3, 0]
        approach = transforms[i, :3, 2]
        near = tree.query_ball_point(tcp, r=search_radius)
        if len(near) < 2:
            continue
        verts_near = mesh_vertices[near]
        rel = verts_near - tcp
        # filter to vertices roughly in front of gripper (along approach)
        along_approach = rel @ approach
        in_front = along_approach > 0
        if in_front.sum() < 2:
            in_front = np.ones(len(near), dtype=bool)
        proj = rel[in_front] @ baseline
        widths[i] = np.clip(proj.max() - proj.min(), 0.005, 0.08)

    return widths


def load_grasps(h5_path: str, mesh_mean: np.ndarray,
                mesh_vertices: np.ndarray | None = None):
    """Load ACRONYM grasps and put them in the centred-mesh local frame.

    ACRONYM stores grasp transforms in the *scaled* object frame (real-world
    metres), so translations must NOT be re-scaled.  Only the mesh-mean
    shift is applied to match the centred mesh.

    Returns (transforms, success_mask, widths).
    """
    with h5py.File(h5_path, "r") as f:
        transforms = np.array(f["grasps/transforms"])             # (G, 4, 4)
        success = np.array(f["grasps/qualities/flex/object_in_gripper"])  # (G,)
        if "grasps/widths" in f:
            widths = np.array(f["grasps/widths"]).astype(np.float32)
        else:
            widths = None

    transforms[:, :3, 3] -= mesh_mean
    success_mask = success > 0

    if widths is None:
        widths = estimate_grasp_widths(
            transforms, mesh_vertices if mesh_vertices is not None
            else np.empty((0, 3)), success_mask)

    return transforms, success_mask, widths


def grasps_to_camera_frame(grasp_local: np.ndarray,
                           obj_pose: np.ndarray,
                           w2c: np.ndarray) -> np.ndarray:
    """Chain object-placement and camera transforms: G_cam = w2c @ obj_pose @ G_local."""
    full = w2c @ obj_pose                        # (4, 4)
    return np.einsum("ij,gjk->gik", full, grasp_local)


def assign_grasp_labels(points: np.ndarray,
                        surface_centres: np.ndarray,
                        z_dirs: np.ndarray,
                        x_dirs: np.ndarray,
                        grasp_widths: np.ndarray | None = None,
                        object_mask: np.ndarray | None = None,
                        dist_thresh: float = 0.005):
    """Per-point labels via radius query around surface-projected grasp centres.

    *surface_centres* are grasp TCPs projected onto the nearest mesh vertex
    (in the same centred camera frame as *points*).  This avoids assigning
    labels to table points when the raw TCP is below the visible surface.

    If *object_mask* is provided, the KDTree is built only from object
    points so labels cannot land on table geometry.
    """
    N = len(points)
    confidence    = np.zeros(N, dtype=np.float32)
    approach_dirs = np.zeros((N, 3), dtype=np.float32)
    base_dirs     = np.zeros((N, 3), dtype=np.float32)
    widths        = np.zeros(N, dtype=np.float32)
    best_dist     = np.full(N, np.inf, dtype=np.float64)

    if len(surface_centres) == 0:
        return confidence, approach_dirs, base_dirs, widths

    if grasp_widths is None:
        grasp_widths = np.full(len(surface_centres), 0.08, dtype=np.float32)

    if object_mask is not None and object_mask.any():
        obj_idx = np.where(object_mask)[0]
        tree = KDTree(points[obj_idx])
    else:
        obj_idx = np.arange(N)
        tree = KDTree(points)

    hit_lists = tree.query_ball_point(surface_centres, r=dist_thresh)

    for gi, local_idxs in enumerate(hit_lists):
        if not local_idxs:
            continue
        pt_idxs = obj_idx[np.asarray(local_idxs)]
        d = np.linalg.norm(points[pt_idxs] - surface_centres[gi], axis=-1)
        closer = d < best_dist[pt_idxs]
        update = pt_idxs[closer]
        best_dist[update]     = d[closer]
        confidence[update]    = 1.0
        approach_dirs[update] = z_dirs[gi]
        base_dirs[update]     = x_dirs[gi]
        widths[update]        = grasp_widths[gi]

    return confidence, approach_dirs, base_dirs, widths


# ──────────────────────── render + label one view ─────────────────────────────

def _roi_crop(pc: np.ndarray, object_mask: np.ndarray,
              min_edge: float = 0.30, scale: float = 2.0) -> np.ndarray:
    """Return boolean mask for a local ROI cube around the object.

    Edge length = max(scale * largest_object_span, min_edge), following
    the CGN paper's local-region strategy (Sec. IV-B).  This keeps the
    object centred with ample surrounding table context (~3/4 table).
    """
    if not object_mask.any():
        return np.ones(len(pc), dtype=bool)
    obj_pts = pc[object_mask]
    centre = obj_pts.mean(axis=0)
    span = obj_pts.max(axis=0) - obj_pts.min(axis=0)
    half_edge = max(span.max() * scale, min_edge) / 2.0
    lo = centre - half_edge
    hi = centre + half_edge
    return np.all((pc >= lo) & (pc <= hi), axis=1)


def process_view(scene, cam_node, renderer, cam_pose_gl, intr,
                 grasp_local, success, grasp_widths, obj_pose, mesh_verts,
                 n_points, table_surface_z: float = 0.3):
    """Render one view and compute per-point grasp labels.

    *mesh_verts* are the centred-mesh vertices (N_v, 3), used to project
    each grasp TCP onto the nearest surface vertex before label assignment.

    Returns dict ready for np.savez_compressed.
    """
    scene.set_pose(cam_node, cam_pose_gl)
    _, depth = renderer.render(scene)

    pc_raw = depth_to_pointcloud(depth, intr["fx"], intr["fy"],
                                 intr["cx"], intr["cy"])
    if len(pc_raw) == 0:
        empty = np.zeros((n_points, 3), dtype=np.float32)
        return dict(depth=depth, points=empty,
                    confidence=np.zeros(n_points, dtype=np.float32),
                    approach_dirs=empty, base_dirs=empty,
                    widths=np.zeros(n_points, dtype=np.float32),
                    camera_pose=cam_pose_gl)

    w2c = world_to_cam_matrix(cam_pose_gl)
    c2w = np.linalg.inv(w2c)

    # identify object vs table points via world-frame z
    pc_world = (c2w[:3, :3] @ pc_raw.T + c2w[:3, 3:4]).T
    object_mask_raw = pc_world[:, 2] > table_surface_z + 0.004

    # ROI crop: local region around object with table context
    roi_mask = _roi_crop(pc_raw, object_mask_raw)
    pc_roi = pc_raw[roi_mask]
    object_mask_roi = object_mask_raw[roi_mask]

    pc = regularize_pc(pc_roi, n_points)

    # recompute object_mask for the subsampled cloud
    pc_world_sub = (c2w[:3, :3] @ pc.T + c2w[:3, 3:4]).T
    object_mask = pc_world_sub[:, 2] > table_surface_z + 0.004

    # mean-centre the point cloud
    pc_mean = pc.mean(axis=0, keepdims=True)
    pc_centred = pc - pc_mean

    # transform grasps into the same centred camera frame
    grasp_cam = grasps_to_camera_frame(grasp_local, obj_pose, w2c)
    grasp_cam[:, :3, 3] -= pc_mean.squeeze()

    # project grasp TCPs onto the nearest mesh surface vertex
    succ = success > 0
    if succ.any():
        full_tf = w2c @ obj_pose
        mv_cam = (full_tf[:3, :3] @ mesh_verts.T + full_tf[:3, 3:4]).T
        mv_cam -= pc_mean.squeeze()
        mesh_tree = KDTree(mv_cam)

        succ_tf = grasp_cam[succ]
        tcp = succ_tf[:, :3, 3]
        _, nearest = mesh_tree.query(tcp, k=1)
        surface_centres = mv_cam[nearest.flatten()]
        z_dirs = succ_tf[:, :3, 2]
        x_dirs = succ_tf[:, :3, 0]
        succ_widths = grasp_widths[succ]
    else:
        surface_centres = np.empty((0, 3))
        z_dirs = np.empty((0, 3))
        x_dirs = np.empty((0, 3))
        succ_widths = np.empty(0)

    conf, app, base, w = assign_grasp_labels(
        pc_centred, surface_centres, z_dirs, x_dirs,
        grasp_widths=succ_widths, object_mask=object_mask)

    cam_pose_cv = opengl_to_opencv_cam(cam_pose_gl)
    cam_pose_out = np.linalg.inv(cam_pose_cv)
    cam_pose_out[:3, 3] -= pc_mean.squeeze()

    return dict(depth=depth, points=pc_centred,
                confidence=conf, approach_dirs=app,
                base_dirs=base, widths=w,
                camera_pose=cam_pose_out)


# ───────────────────────────── per-object loop ────────────────────────────────

def generate_object(acronym_root: str, entry: dict,
                    out_dir: str, n_views: int, n_points: int):
    """Generate all views for one object."""
    mesh_path = os.path.join(acronym_root, entry["mesh_path"])
    h5_path   = os.path.join(acronym_root, "grasps", entry["grasp_file"])
    if not os.path.exists(mesh_path):
        print(f"  SKIP mesh not found: {mesh_path}")
        return
    if not os.path.exists(h5_path):
        print(f"  SKIP grasp file not found: {h5_path}")
        return

    scale = entry["scale"]
    mesh, mesh_mean = load_and_prepare_mesh(mesh_path, scale)
    mesh_verts = np.array(mesh.vertices)
    grasp_local, success, grasp_widths = load_grasps(
        h5_path, mesh_mean, mesh_vertices=mesh_verts)

    intr = REALSENSE
    scene, cam_node, renderer, obj_pose, table_dims = build_scene(mesh, intr)

    cam_poses = sample_camera_poses(n_views)
    os.makedirs(out_dir, exist_ok=True)

    total_pos = 0
    table_z = table_dims[2] / 2
    for i, cam_gl in enumerate(cam_poses):
        cam_gl[2, 3] += table_dims[2]            # shift camera above table

        sample = process_view(scene, cam_node, renderer, cam_gl, intr,
                              grasp_local, success, grasp_widths,
                              obj_pose, mesh_verts,
                              n_points, table_surface_z=table_z)
        np.savez_compressed(os.path.join(out_dir, f"{i:03d}.npz"), **sample)
        total_pos += int(sample["confidence"].sum())

    renderer.delete()
    cat = entry["category"]
    print(f"  {cat}: saved {n_views} views → {out_dir}  "
          f"(avg {total_pos / max(n_views, 1):.0f} positive pts/view)")


# ─────────────────────────────────── CLI ──────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate depth + point cloud + grasp label .npz files")
    parser.add_argument("--acronym_root", default="data/acronym")
    parser.add_argument("--out_root", default="data/out")
    parser.add_argument("--category", default=None,
                        help="Process only this category (e.g. Mug)")
    parser.add_argument("--mesh_hash", default=None,
                        help="Process only this mesh hash (for debugging)")
    parser.add_argument("--splits", nargs="+", default=["train", "test"],
                        choices=["train", "test"],
                        help="Which manifest splits to render")
    parser.add_argument("--n_views", type=int, default=360)
    parser.add_argument("--n_points", type=int, default=4096)
    args = parser.parse_args()

    with open(os.path.join(args.acronym_root, "manifest.json")) as f:
        manifest = json.load(f)

    splits = set(args.splits)
    selected = []
    for entry in manifest:
        if args.category and entry["category"] != args.category:
            continue
        if args.mesh_hash and entry.get("mesh_hash") != args.mesh_hash:
            continue
        if entry.get("split") not in splits:
            continue
        selected.append(entry)

    print(f"Rendering {len(selected)} object(s) across splits={sorted(splits)} "
          f"with {args.n_views} views each")

    for entry in selected:
        mesh_hash = entry.get("mesh_hash") or os.path.splitext(
            os.path.basename(entry["mesh_path"]))[0]
        print(f"Generating {entry['split']}/{entry['category']}/{mesh_hash} …")
        generate_object(
            acronym_root=args.acronym_root,
            entry=entry,
            out_dir=os.path.join(args.out_root,
                                 entry["split"],
                                 entry["category"],
                                 mesh_hash),
            n_views=args.n_views,
            n_points=args.n_points,
        )

    print("Done.")


if __name__ == "__main__":
    main()
