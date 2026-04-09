"""Visualize .npz files produced by ``data/generate_data.py``.

Each .npz contains: depth, points, confidence, approach_dirs, base_dirs,
widths, camera_pose.

Usage:
    python data/visualizer.py data/out/Mug/000.npz                 # depth + point cloud
    python data/visualizer.py data/out/Mug/000.npz --mode depth    # depth only
    python data/visualizer.py data/out/Mug/000.npz --mode pc       # point cloud only
    python data/visualizer.py data/out/Mug/000.npz --mode grasps   # pc coloured by confidence
    python data/visualizer.py data/out/Mug/000.npz --mode poses    # 3D gripper poses on pc
    python data/visualizer.py data/out/Mug/000.npz --mode poses --save out.png  # save to file
    python data/visualizer.py --mode gt Mug                        # all ACRONYM grasps on mesh
    python data/visualizer.py --mode gt Mug --max_grasps 50 --save gt.png
    python data/visualizer.py data/out/Mug/ --mode depth --grid    # depth grid of all views
"""

from __future__ import annotations

import os

os.environ.setdefault("DISPLAY", ":0")
os.environ.setdefault("XDG_SESSION_TYPE", "x11")

import glob
import argparse
import numpy as np


# ──────────────────────────────── depth ───────────────────────────────────────

def show_depth(depth: np.ndarray, title: str = "Depth", ax=None):
    import matplotlib.pyplot as plt

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 6))

    masked = np.where(depth > 0, depth, np.nan)
    im = ax.imshow(masked, cmap="viridis")
    ax.set_title(title, fontsize=10)
    ax.axis("off")

    if standalone:
        plt.colorbar(im, ax=ax, label="depth (m)", shrink=0.8)
        plt.tight_layout()
        plt.show()
    return im


# ──────────────────────────── point cloud ─────────────────────────────────────

def show_pc(pc: np.ndarray, title: str = "Point Cloud",
            max_pts: int = 6000):
    """Show point cloud coloured by Z height (matplotlib fallback safe)."""
    try:
        import open3d as o3d
        import matplotlib.pyplot as plt
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
        z = pc[:, 2]
        z_norm = (z - z.min()) / (z.ptp() + 1e-8)
        pcd.colors = o3d.utility.Vector3dVector(plt.cm.viridis(z_norm)[:, :3])
        o3d.visualization.draw_geometries([pcd], window_name=title,
                                           width=960, height=720)
    except Exception:
        _pc_matplotlib(pc, title=title, max_pts=max_pts)


def _pc_matplotlib(pc, title="Point Cloud", max_pts=6000,
                   colors=None, cmap="viridis", ax=None):
    import matplotlib.pyplot as plt

    standalone = ax is None
    if standalone:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

    n = len(pc)
    if n > max_pts:
        idx = np.random.choice(n, max_pts, replace=False)
        pc = pc[idx]
        if colors is not None:
            colors = colors[idx]

    c = colors if colors is not None else pc[:, 2]
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2],
               s=0.4, c=c, cmap=cmap, alpha=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    if standalone:
        plt.tight_layout()
        plt.show()


# ────────────────────────── grasp visualisation ───────────────────────────────

def show_grasps(data: dict, title: str = "Grasps", max_pts: int = 6000):
    """Point cloud coloured by grasp confidence (blue=0, red=1)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    pc   = data["points"]
    conf = data["confidence"]
    n_pos = int((conf > 0.5).sum())

    cmap = LinearSegmentedColormap.from_list("br", ["#3060cf", "#cf3030"])
    colors = cmap(conf)[:, :3]

    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries(
            [pcd], window_name=f"{title}  ({n_pos} positive pts)",
            width=960, height=720)
    except Exception:
        _pc_matplotlib(pc, title=f"{title}  ({n_pos} positive pts)",
                       max_pts=max_pts, colors=conf, cmap=cmap)


# ──────────────────────── 3-D gripper pose drawing ────────────────────────

PANDA_FINGER_BASE = 0.0584    # wrist → finger attach (m)
PANDA_FINGER_TIP  = 0.1053   # wrist → fingertip (m)

def _gripper_lines(wrist, approach, binormal, width=0.08):
    """Return (6×3 points, 6×2 line-indices) for one Panda gripper.

    ACRONYM places the gripper **wrist** at *wrist*.  Fingers extend
    forward along *approach* and close along *binormal*.
    """
    half_w = width / 2
    fb_l = wrist + PANDA_FINGER_BASE * approach + half_w * binormal
    fb_r = wrist + PANDA_FINGER_BASE * approach - half_w * binormal
    ft_l = wrist + PANDA_FINGER_TIP  * approach + half_w * binormal
    ft_r = wrist + PANDA_FINGER_TIP  * approach - half_w * binormal

    pts = np.array([wrist, fb_l, fb_r, ft_l, ft_r])
    lines = np.array([[0, 1], [0, 2],   # wrist → left/right base
                       [1, 3], [2, 4],   # left/right fingers
                       [1, 2],           # back bar (connecting bases)
                       [3, 4]])          # front bar (connecting tips)
    return pts, lines


def _deduplicate_grasps(data, angle_thresh_deg=5.0):
    """Collapse nearby positive-grasp points that share the same direction."""
    conf = data["confidence"]
    pos = np.where(conf > 0.5)[0]
    if len(pos) == 0:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3)), np.empty(0)

    pts = data["points"][pos]
    app = data["approach_dirs"][pos]
    base = data["base_dirs"][pos]
    widths = data["widths"][pos]

    app_norm = app / (np.linalg.norm(app, axis=1, keepdims=True) + 1e-9)
    keep = [0]
    for i in range(1, len(pos)):
        cos_sim = np.einsum("j,kj->k", app_norm[i], app_norm[np.array(keep)])
        dists = np.linalg.norm(pts[i] - pts[np.array(keep)], axis=1)
        if (dists < 0.005).any() and (cos_sim > np.cos(np.deg2rad(angle_thresh_deg))).any():
            continue
        keep.append(i)

    keep = np.array(keep)
    return pts[keep], app[keep], base[keep], widths[keep]


def show_poses(data: dict, title: str = "Grasp Poses", max_bg: int = 8000,
               save_path: str | None = None,
               max_grasps: int | None = None):
    """Draw 3-D parallel-jaw grippers on the point cloud (Open3D or mpl)."""
    centres, approaches, binormals, widths = _deduplicate_grasps(data)
    n_unique = len(centres)
    if max_grasps is not None and n_unique > max_grasps:
        idx = np.random.default_rng(0).choice(n_unique, max_grasps, replace=False)
        centres = centres[idx]
        approaches = approaches[idx]
        binormals = binormals[idx]
        widths = widths[idx]

    n_grasps = len(centres)
    print(f"  Drawing {n_grasps} unique gripper poses")

    rng = np.random.default_rng(42)
    grasp_colors = rng.uniform(0.25, 1.0, size=(n_grasps, 3))

    if save_path:
        _show_poses_mpl(data, centres, approaches, binormals, widths,
                        grasp_colors, title, max_bg, save_path=save_path)
        return

    try:
        import open3d as o3d
        _show_poses_o3d(data, centres, approaches, binormals, widths,
                        grasp_colors, title, max_bg)
    except Exception:
        _show_poses_mpl(data, centres, approaches, binormals, widths,
                        grasp_colors, title, max_bg)


def _show_poses_o3d(data, centres, approaches, binormals, widths,
                    grasp_colors, title, max_bg):
    import open3d as o3d

    pc = data["points"]
    idx = np.random.choice(len(pc), min(max_bg, len(pc)), replace=False)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[idx])
    pcd.paint_uniform_color([0.65, 0.65, 0.65])

    geometries = [pcd]

    for i in range(len(centres)):
        pts, lines = _gripper_lines(centres[i], approaches[i],
                                    binormals[i], widths[i])
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines = o3d.utility.Vector2iVector(lines)
        c = grasp_colors[i]
        ls.colors = o3d.utility.Vector3dVector(
            np.tile(c, (len(lines), 1)))
        geometries.append(ls)

    o3d.visualization.draw_geometries(
        geometries, window_name=f"{title}  ({len(centres)} grasps)",
        width=1024, height=768)


def _show_poses_mpl(data, centres, approaches, binormals, widths,
                    grasp_colors, title, max_bg, save_path=None):
    import matplotlib
    if save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    pc = data["points"]
    idx = np.random.choice(len(pc), min(max_bg, len(pc)), replace=False)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pc[idx, 0], pc[idx, 1], pc[idx, 2],
               s=0.3, c="silver", alpha=0.4)

    for i in range(len(centres)):
        pts, lines = _gripper_lines(centres[i], approaches[i],
                                    binormals[i], widths[i])
        segs = [(pts[a], pts[b]) for a, b in lines]
        lc = Line3DCollection(segs, colors=[grasp_colors[i]] * len(segs),
                              linewidths=1.8)
        ax.add_collection3d(lc)

    ax.set_title(f"{title}  ({len(centres)} grasps)", fontsize=11)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
        plt.close(fig)
    else:
        plt.show()


# ─────────── ground-truth ACRONYM grasps (mesh local frame) ───────────────

def _load_gt(category: str, acronym_root: str = "data/acronym"):
    """Load mesh surface points + all successful grasps for *category*."""
    import json
    import h5py
    import trimesh

    manifest_path = os.path.join(acronym_root, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    entry = next((e for e in manifest if e["category"] == category), None)
    if entry is None:
        raise ValueError(
            f"Category '{category}' not in manifest. "
            f"Available: {[e['category'] for e in manifest]}")

    mesh_path = os.path.join(acronym_root, entry["mesh_path"])
    h5_path = os.path.join(acronym_root, "grasps", entry["grasp_file"])
    scale = entry["scale"]

    mesh = trimesh.load(mesh_path, force="mesh")
    mesh.apply_scale(scale)
    mesh_mean = mesh.vertices.mean(axis=0)
    mesh.vertices -= mesh_mean

    surface_pts = mesh.sample(10000).astype(np.float32)

    with h5py.File(h5_path, "r") as f:
        transforms = np.array(f["grasps/transforms"])
        success = np.array(f["grasps/qualities/flex/object_in_gripper"])

    transforms[:, :3, 3] -= mesh_mean
    succ_tf = transforms[success > 0]

    return surface_pts, succ_tf, entry


def show_gt(category: str, acronym_root: str = "data/acronym",
            max_grasps: int = 100, save_path: str | None = None):
    """Visualize all ground-truth ACRONYM grasps on the object mesh."""
    surface_pts, succ_tf, entry = _load_gt(category, acronym_root)
    n_total = len(succ_tf)

    if n_total > max_grasps:
        idx = np.random.default_rng(0).choice(n_total, max_grasps, replace=False)
        succ_tf = succ_tf[idx]

    centres = succ_tf[:, :3, 3]
    approaches = succ_tf[:, :3, 2]
    binormals = succ_tf[:, :3, 0]
    n_show = len(centres)

    title = f"{category} GT — {n_show}/{n_total} successful grasps"
    print(f"  {title}")

    rng = np.random.default_rng(42)
    colors = rng.uniform(0.25, 1.0, size=(n_show, 3))

    if save_path:
        _show_gt_mpl(surface_pts, centres, approaches, binormals,
                     colors, title, save_path=save_path)
        return

    try:
        import open3d as o3d
        _show_gt_o3d(surface_pts, centres, approaches, binormals,
                     colors, title)
    except Exception:
        _show_gt_mpl(surface_pts, centres, approaches, binormals,
                     colors, title)


def _show_gt_o3d(surface_pts, centres, approaches, binormals,
                 grasp_colors, title):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(surface_pts)
    pcd.paint_uniform_color([0.55, 0.55, 0.55])

    geometries = [pcd]
    for i in range(len(centres)):
        pts, lines = _gripper_lines(centres[i], approaches[i], binormals[i])
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector(
            np.tile(grasp_colors[i], (len(lines), 1)))
        geometries.append(ls)

    o3d.visualization.draw_geometries(
        geometries, window_name=title, width=1024, height=768)


def _show_gt_mpl(surface_pts, centres, approaches, binormals,
                 grasp_colors, title, save_path=None):
    import matplotlib
    if save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    idx = np.random.choice(len(surface_pts),
                           min(6000, len(surface_pts)), replace=False)
    ax.scatter(surface_pts[idx, 0], surface_pts[idx, 1], surface_pts[idx, 2],
               s=0.5, c="silver", alpha=0.4)

    for i in range(len(centres)):
        pts, lines = _gripper_lines(centres[i], approaches[i], binormals[i])
        segs = [(pts[a], pts[b]) for a, b in lines]
        lc = Line3DCollection(segs, colors=[grasp_colors[i]] * len(segs),
                              linewidths=1.5)
        ax.add_collection3d(lc)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_aspect("equal")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
        plt.close(fig)
    else:
        plt.show()


# ─────────────────────────── combined views ───────────────────────────────────

def show_single(path: str, mode: str = "both", save: str | None = None,
                max_grasps: int | None = None):
    data = dict(np.load(path))
    name = os.path.basename(path)

    if mode in ("depth", "both") and "depth" in data:
        show_depth(data["depth"], title=f"{name} – depth")

    if mode in ("pc", "both") and "points" in data:
        show_pc(data["points"], title=f"{name} – point cloud")

    if mode == "grasps" and "confidence" in data:
        show_grasps(data, title=name)

    if mode == "poses" and "confidence" in data:
        show_poses(data, title=name, save_path=save, max_grasps=max_grasps)


# ──────────────────────────── folder / grid ───────────────────────────────────

def show_depth_grid(folder: str, cols: int = 6):
    import matplotlib.pyplot as plt

    files = sorted(glob.glob(os.path.join(folder, "*.npz")))
    if not files:
        print(f"No .npz files in {folder}")
        return

    n = len(files)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_2d(axes)

    for i, f in enumerate(files):
        r, c = divmod(i, cols)
        d = np.load(f)
        if "depth" in d:
            show_depth(d["depth"], title=os.path.basename(f), ax=axes[r, c])

    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis("off")

    cat = os.path.basename(os.path.normpath(folder))
    fig.suptitle(f"{cat} – {n} depth views", fontsize=13)
    plt.tight_layout()
    plt.show()


def show_folder(folder: str, mode: str = "both", grid: bool = False,
                save: str | None = None,
                max_grasps: int | None = None):
    if grid and mode in ("depth", "both"):
        show_depth_grid(folder)
        return

    files = sorted(glob.glob(os.path.join(folder, "*.npz")))
    if not files:
        print(f"No .npz files in {folder}")
        return

    for f in files:
        out = None
        if save:
            os.makedirs(save, exist_ok=True)
            base = os.path.splitext(os.path.basename(f))[0]
            out = os.path.join(save, f"{base}_{mode}.png")
        show_single(f, mode=mode, save=out, max_grasps=max_grasps)


# ─────────────────────────────────── CLI ──────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize depth / point cloud / grasp label .npz files")
    parser.add_argument("path",
                        help="Path to .npz / folder, or category name for gt mode")
    parser.add_argument("--mode",
                        choices=["depth", "pc", "grasps", "poses", "gt", "both"],
                        default="both")
    parser.add_argument("--grid", action="store_true",
                        help="Tile all depth maps in one figure (folder mode)")
    parser.add_argument("--save", default=None,
                        help="Save to file (single) or directory (folder)")
    parser.add_argument("--acronym_root", default="data/acronym",
                        help="Root of ACRONYM data (for gt mode)")
    parser.add_argument("--max_grasps", type=int, default=100,
                        help="Max grasps to draw in gt mode, and max unique poses"
                             " to draw in poses mode")
    args = parser.parse_args()

    if args.mode == "gt":
        show_gt(args.path, acronym_root=args.acronym_root,
                max_grasps=args.max_grasps, save_path=args.save)
    elif os.path.isdir(args.path):
        show_folder(args.path, mode=args.mode, grid=args.grid,
                    save=args.save, max_grasps=args.max_grasps)
    elif os.path.isfile(args.path):
        show_single(args.path, mode=args.mode, save=args.save,
                    max_grasps=args.max_grasps)
    else:
        print(f"Not found: {args.path}")


if __name__ == "__main__":
    main()
