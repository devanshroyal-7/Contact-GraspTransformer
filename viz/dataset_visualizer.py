"""Visualize .npz files produced by ``data/generate_data.py``.

Dataset-artifact viewer (depth / point cloud / grasp labels / ACRONYM GT).
Lives under ``viz/`` with the other visualization CLIs; ``data/`` owns
generation and loading only.

Usage:
    python viz/dataset_visualizer.py data/out/train/Mug/<hash>/000.npz
    python viz/dataset_visualizer.py data/out/train/Mug/<hash>/000.npz --mode grasps
    python viz/dataset_visualizer.py Mug --mode gt --max_grasps 50
    python viz/dataset_visualizer.py Mug --mode gt --debug
"""

from __future__ import annotations

import glob
import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Script-mode bootstrap only (`python viz/dataset_visualizer.py`).
if __package__ is None:  # pragma: no cover
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from models.cgn_heads import PANDA_FINGER_BASE, PANDA_FINGER_TIP
from models.types import GraspSampleNP


def _open3d_display_unavailable(exc: BaseException) -> bool:
    """True when Open3D failed due to missing deps or a headless/display issue."""
    if isinstance(exc, (ImportError, OSError)):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        needles = (
            "glfw",
            "display",
            "x11",
            "wayland",
            "egl",
            "headless",
            "failed to create",
            "cannot connect",
            "no protocol specified",
        )
        return any(tok in msg for tok in needles)
    return False


def configure_display() -> None:
    """Set DISPLAY/XDG defaults for interactive Open3D/GLFW on Linux.

    Call from CLI ``main()`` (or other interactive entry points) only — do not
    run at import time so library importers are not forced onto ``:0`` / x11.
    """
    os.environ.setdefault("DISPLAY", ":0")
    os.environ.setdefault("XDG_SESSION_TYPE", "x11")


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
    except Exception as e:
        if not _open3d_display_unavailable(e):
            raise
        print(f"Open3D point-cloud view failed ({e}); falling back to matplotlib")
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

def show_grasps(data: GraspSampleNP, title: str = "Grasps", max_pts: int = 6000):
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
    except Exception as e:
        if not _open3d_display_unavailable(e):
            raise
        print(f"Open3D grasp view failed ({e}); falling back to matplotlib")
        _pc_matplotlib(pc, title=f"{title}  ({n_pos} positive pts)",
                       max_pts=max_pts, colors=conf, cmap=cmap)


# ──────────────────────── 3-D gripper pose drawing ────────────────────────

def _gripper_lines(wrist, approach, binormal, width=0.08):
    """Return (6×3 points, 6×2 line-indices) for one Panda gripper.

    ACRONYM places the gripper **wrist** at *wrist*.  Fingers extend
    forward along *approach* and close along *binormal*.
    Finger length constants come from ``models.cgn_heads``.
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


def _deduplicate_grasps(data: GraspSampleNP, angle_thresh_deg=5.0):
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


def show_poses(data: GraspSampleNP, title: str = "Grasp Poses", max_bg: int = 8000,
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
    except Exception as e:
        if not _open3d_display_unavailable(e):
            raise
        print(f"Open3D pose view failed ({e}); falling back to matplotlib")
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


def _resolve_acronym_root(cli_path: str) -> str:
    """Use CLI path if it has manifest.json; else try repo ``data/acronym``."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    candidates = [
        cli_path,
        os.path.join(repo_root, "data", "acronym"),
        os.path.join(here, "acronym"),
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "manifest.json")):
            if os.path.normpath(os.path.abspath(c)) != os.path.normpath(
                os.path.abspath(cli_path)
            ):
                print(f"  Using acronym_root: {c}", file=sys.stderr)
            return c
    return cli_path


def show_gt(category: str, acronym_root: str = "data/acronym",
            max_grasps: int = 100, save_path: str | None = None,
            debug: bool = False):
    """Visualize all ground-truth ACRONYM grasps on the object mesh."""
    surface_pts, succ_tf, entry = _load_gt(category, acronym_root)
    n_total = len(succ_tf)

    if debug:
        mesh_path = os.path.join(acronym_root, entry["mesh_path"])
        h5_path = os.path.join(acronym_root, "grasps", entry["grasp_file"])
        print("[visualizer debug]")
        print(f"  acronym_root   : {os.path.abspath(acronym_root)}")
        print(f"  mesh_path      : {mesh_path} (exists={os.path.isfile(mesh_path)})")
        print(f"  h5_path        : {h5_path} (exists={os.path.isfile(h5_path)})")
        print(f"  manifest entry : {entry}")
        print(f"  surface_pts    : {surface_pts.shape}")
        print(f"  successful TF  : {succ_tf.shape}")

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
    except Exception as e:
        if not _open3d_display_unavailable(e):
            raise
        print(f"Open3D GT grasp view failed ({e}); falling back to matplotlib")
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

def _as_grasp_sample(data: dict) -> GraspSampleNP:
    """Narrow a loaded NPZ mapping into ``GraspSampleNP`` (required keys)."""
    required = ("points", "confidence", "approach_dirs", "base_dirs", "widths")
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(
            f"NPZ missing GraspSampleNP keys: {', '.join(missing)}"
        )
    return GraspSampleNP(
        points=np.asarray(data["points"]),
        confidence=np.asarray(data["confidence"]),
        approach_dirs=np.asarray(data["approach_dirs"]),
        base_dirs=np.asarray(data["base_dirs"]),
        widths=np.asarray(data["widths"]),
    )


def show_single(path: str, mode: str = "both", save: str | None = None,
                max_grasps: int | None = None):
    loaded = dict(np.load(path))
    # Re-bind optional NPZ fields so key-flow analysis sees local writes.
    depth = loaded.get("depth")
    points = loaded.get("points")
    name = os.path.basename(path)

    if mode in ("depth", "both") and depth is not None:
        show_depth(depth, title=f"{name} – depth")

    if mode in ("pc", "both") and points is not None:
        show_pc(points, title=f"{name} – point cloud")

    if mode in ("grasps", "poses"):
        try:
            grasp = _as_grasp_sample(loaded)
        except KeyError:
            grasp = None
        if grasp is not None:
            if mode == "grasps":
                show_grasps(grasp, title=name)
            else:
                show_poses(
                    grasp,
                    title=name,
                    save_path=save,
                    max_grasps=max_grasps,
                )


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
    if grid:
        if mode == "depth":
            show_depth_grid(folder)
            return
        if mode == "both":
            show_depth_grid(folder)
            mode = "pc"  # depth already shown; iterate files for point clouds
        else:
            raise ValueError(
                f"--grid only supports mode 'depth' or 'both', got {mode!r}"
            )

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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="In gt mode, print resolved ACRONYM paths and array shapes "
             "(replaces the old data/viz_test.py helper).",
    )
    args = parser.parse_args()
    configure_display()

    if args.mode == "gt":
        root = _resolve_acronym_root(args.acronym_root)
        show_gt(args.path, acronym_root=root,
                max_grasps=args.max_grasps, save_path=args.save,
                debug=args.debug)
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
