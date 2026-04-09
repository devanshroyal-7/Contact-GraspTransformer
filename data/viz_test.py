#!/usr/bin/env python3
"""Self-contained debug script: Mug (or any manifest category) + ACRONYM GT grasps.

Copied from ``data/visualizer.py`` (GT path only) so you can edit / breakpoint here
without going through the main visualizer.

Requires ``data/acronym/`` (or pass ``--acronym_root``).

Usage (from repo root):
    python data/viz_test.py
    python data/viz_test.py --max_grasps 20          # includes RGB XYZ axes at origin
    python data/viz_test.py --verbose
    python data/viz_test.py --max_grasps 200 --save mug_gt.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("DISPLAY", ":0")
os.environ.setdefault("XDG_SESSION_TYPE", "x11")

import h5py
import numpy as np
import trimesh

# ─── gripper wireframe (matches visualizer GT drawing) ───────────────────────

PANDA_FINGER_BASE = 0.0584   # wrist → finger attach (m)
PANDA_FINGER_TIP = 0.1053    # wrist → fingertip (m)


def _gripper_lines(wrist, approach, binormal, width=0.08):
    """Return (5×3 points, 6×2 line-indices) for one Panda gripper.

    ACRONYM places the gripper **wrist** at *wrist*. Fingers extend forward
    along *approach* and close along *binormal*.
    """
    half_w = width / 2
    fb_l = wrist + PANDA_FINGER_BASE * approach + half_w * binormal
    fb_r = wrist + PANDA_FINGER_BASE * approach - half_w * binormal
    ft_l = wrist + PANDA_FINGER_TIP * approach + half_w * binormal
    ft_r = wrist + PANDA_FINGER_TIP * approach - half_w * binormal

    pts = np.array([wrist, fb_l, fb_r, ft_l, ft_r])
    lines = np.array(
        [
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 4],
            [1, 2],
            [3, 4],
        ]
    )
    return pts, lines


def _axes_length(surface_pts: np.ndarray) -> float:
    """World-frame axis arrow length from mesh extent (metres)."""
    if len(surface_pts) == 0:
        return 0.1
    ext = float((surface_pts.max(axis=0) - surface_pts.min(axis=0)).max())
    return max(0.05, ext * 0.18)


# ─── load ACRONYM + mesh (centred mesh frame, same as generate_data) ─────────

def _load_gt(category: str, acronym_root: str):
    manifest_path = os.path.join(acronym_root, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    entry = next((e for e in manifest if e["category"] == category), None)
    if entry is None:
        raise ValueError(
            f"Category '{category}' not in manifest. "
            f"Available: {[e['category'] for e in manifest]}"
        )

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

    transforms[:, :3, 3] *= scale
    transforms[:, :3, 3] -= mesh_mean
    succ_tf = transforms[success > 0]

    return surface_pts, succ_tf, entry, mesh_path, h5_path, mesh_mean, transforms, success


def _show_gt_o3d(
    surface_pts, centres, approaches, binormals, grasp_colors, title, axis_len: float
):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(surface_pts)
    pcd.paint_uniform_color([0.55, 0.55, 0.55])

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=axis_len, origin=[0, 0, 0]
    )

    geometries = [pcd]
    for i in range(len(centres)):
        pts, lines = _gripper_lines(centres[i], approaches[i], binormals[i])
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector(
            np.tile(grasp_colors[i], (len(lines), 1))
        )
        geometries.append(ls)
    geometries.append(coord)

    o3d.visualization.draw_geometries(
        geometries, window_name=title, width=1024, height=768
    )


def _show_gt_mpl(
    surface_pts,
    centres,
    approaches,
    binormals,
    grasp_colors,
    title,
    axis_len: float,
    save_path=None,
):
    import matplotlib

    if save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    idx = np.random.choice(len(surface_pts), min(6000, len(surface_pts)), replace=False)
    ax.scatter(
        surface_pts[idx, 0],
        surface_pts[idx, 1],
        surface_pts[idx, 2],
        s=0.5,
        c="silver",
        alpha=0.4,
    )

    for i in range(len(centres)):
        pts, lines = _gripper_lines(centres[i], approaches[i], binormals[i])
        segs = [(pts[a], pts[b]) for a, b in lines]
        lc = Line3DCollection(
            segs, colors=[grasp_colors[i]] * len(segs), linewidths=1.5
        )
        ax.add_collection3d(lc)

    o = np.zeros(3, dtype=np.float64)
    axis_segs = [
        (o, np.array([axis_len, 0, 0])),
        (o, np.array([0, axis_len, 0])),
        (o, np.array([0, 0, axis_len])),
    ]
    ax_lc = Line3DCollection(
        axis_segs, colors=["#e41a1c", "#4daf4a", "#377eb8"], linewidths=2.8
    )
    ax.add_collection3d(ax_lc)

    ax.set_title(title + "  (RGB = X,Y,Z at origin)", fontsize=11)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_aspect("equal")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
        plt.close(fig)
    else:
        plt.show()


def show_gt(
    category: str,
    acronym_root: str,
    max_grasps: int,
    save_path: str | None,
    verbose: bool,
):
    loaded = _load_gt(category, acronym_root)
    surface_pts, succ_tf, entry, mesh_path, h5_path, mesh_mean, transforms, success = loaded

    n_all = len(transforms)
    n_total = len(succ_tf)

    if verbose:
        print("[viz_test debug]")
        print(f"  acronym_root   : {os.path.abspath(acronym_root)}")
        print(f"  mesh_path      : {mesh_path} (exists={os.path.isfile(mesh_path)})")
        print(f"  h5_path        : {h5_path} (exists={os.path.isfile(h5_path)})")
        print(f"  manifest entry : {entry}")
        print(f"  mesh_mean      : {mesh_mean}")
        print(f"  transforms     : shape {transforms.shape}, success shape {success.shape}")
        print(f"  success count  : {(success > 0).sum()} / {n_all}")
        print(f"  surface_pts    : {surface_pts.shape}")

    if n_total > max_grasps:
        idx = np.random.default_rng(0).choice(n_total, max_grasps, replace=False)
        succ_tf = succ_tf[idx]

    centres = succ_tf[:, :3, 3]
    approaches = succ_tf[:, :3, 2]
    binormals = succ_tf[:, :3, 0]
    n_show = len(centres)

    title = f"{category} GT — {n_show}/{n_total} successful grasps"
    print(title)

    rng = np.random.default_rng(42)
    colors = rng.uniform(0.25, 1.0, size=(n_show, 3))

    axis_len = _axes_length(surface_pts)
    if verbose:
        print(f"  axis_len (RGB XYZ triad): {axis_len:.4f} m")

    if save_path:
        _show_gt_mpl(
            surface_pts,
            centres,
            approaches,
            binormals,
            colors,
            title,
            axis_len,
            save_path=save_path,
        )
        return

    try:
        import open3d as o3d  # noqa: F401

        _show_gt_o3d(
            surface_pts, centres, approaches, binormals, colors, title, axis_len
        )
    except Exception as e:
        if verbose:
            print(f"  Open3D failed ({type(e).__name__}: {e}), falling back to matplotlib")
        _show_gt_mpl(
            surface_pts, centres, approaches, binormals, colors, title, axis_len
        )


def _resolve_acronym_root(cli_path: str) -> str:
    """Use CLI path if it has manifest.json; else try repo ``data/acronym`` / ``data/acronym`` next to this file."""
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ACRONYM ground-truth grasps on mesh (self-contained debug script)."
    )
    parser.add_argument("--category", default="Mug", help="Manifest category (default: Mug).")
    parser.add_argument(
        "--acronym_root",
        default="data/acronym",
        help="ACRONYM root (default: data/acronym).",
    )
    parser.add_argument("--max_grasps", type=int, default=100, help="Max grasps to draw.")
    parser.add_argument(
        "--save",
        default=None,
        metavar="PATH",
        help="Save PNG (matplotlib) instead of interactive window.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print paths, shapes, and Open3D fallback reason."
    )
    args = parser.parse_args()

    root = _resolve_acronym_root(args.acronym_root)
    if not os.path.isdir(root):
        sys.exit(f"acronym_root not found: {args.acronym_root!r} (resolved: {root!r})")

    show_gt(
        args.category,
        acronym_root=root,
        max_grasps=args.max_grasps,
        save_path=args.save,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
