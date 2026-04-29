import argparse
from pathlib import Path

import numpy as np


DEFAULT_MUG_PATH = Path(
    "data/out/train/Mug/2997f21fa426e18a6ab1a25d0e8f3590/000.npz"
)


def load_mug_points(path=DEFAULT_MUG_PATH):
    """Load the mug view point cloud from a generated .npz sample."""
    path = Path(path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path

    with np.load(path) as data:
        if "points" not in data:
            raise KeyError(f"{path} does not contain a 'points' array")
        points = data["points"].astype(np.float64)

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected points with shape (N, 3+), got {points.shape}")

    return points[:, :3], path


def _part1by2(n):
    """Spread the low 10 bits by inserting two 0-bits between each bit."""
    n = n.astype(np.int64) & 0x000003FF
    n = (n ^ (n << 16)) & 0xFF0000FF
    n = (n ^ (n << 8)) & 0x0F00F00F
    n = (n ^ (n << 4)) & 0xC30C30C3
    n = (n ^ (n << 2)) & 0x49249249
    return n


def morton_encode(grid_coord):
    """NumPy equivalent of models.backbone_ptv3.morton_encode."""
    x, y, z = grid_coord[:, 0], grid_coord[:, 1], grid_coord[:, 2]
    return _part1by2(x) | (_part1by2(y) << 1) | (_part1by2(z) << 2)


def quantize_like_ptv3(points, grid_size):
    """Match PTv3 _quantize: floor(xyz / grid_size), then subtract per-scene min."""
    raw_grid = np.floor(points / grid_size).astype(np.int64)
    grid_min = raw_grid.min(axis=0, keepdims=True)
    return raw_grid - grid_min, grid_min.squeeze(0) * grid_size


def voxel_pool_bitshift(points, grid_coord, pool_shift=1):
    """Match PTv3 voxel pooling: grid_coord >> pool_shift, Morton cluster, mean xyz."""
    coarse_grid = grid_coord >> pool_shift
    code = morton_encode(coarse_grid)
    _, first_indices, inverse = np.unique(code, return_index=True, return_inverse=True)

    down_points = np.zeros((len(first_indices), 3), dtype=points.dtype)
    np.add.at(down_points, inverse, points)
    counts = np.bincount(inverse).reshape(-1, 1)
    down_points /= counts

    return down_points, coarse_grid[first_indices]


def voxelize_to_occupied(points, grid_size):
    """Return one occupied voxel and one mean xyz feature row per voxel."""
    grid_coord, origin = quantize_like_ptv3(points, grid_size)
    code = morton_encode(grid_coord)
    _, first_indices, inverse = np.unique(code, return_index=True, return_inverse=True)

    voxel_points = np.zeros((len(first_indices), 3), dtype=points.dtype)
    np.add.at(voxel_points, inverse, points)
    counts = np.bincount(inverse).astype(points.dtype).reshape(-1, 1)
    voxel_points /= counts

    return grid_coord[first_indices], voxel_points, counts.squeeze(1), origin


def normalize_values(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    span = values.max() - values.min()
    if span == 0:
        return np.zeros_like(values)
    return (values - values.min()) / span


def voxels_to_gradient_mesh(grid_coord, voxel_size, origin, values=None, normalize=True):
    """Build voxel cubes colored by height or by a per-voxel scalar value."""
    import open3d as o3d
    import matplotlib.pyplot as plt

    if len(grid_coord) == 0:
        return o3d.geometry.TriangleMesh()

    cmap = plt.get_cmap("turbo")
    merged = o3d.geometry.TriangleMesh()
    half = np.array([voxel_size, voxel_size, voxel_size], dtype=np.float64) * 0.5
    centers = origin + (grid_coord.astype(np.float64) + 0.5) * voxel_size
    if values is None:
        color_values = normalize_values(centers[:, 2])
    else:
        if len(values) != len(grid_coord):
            raise ValueError(
                f"Expected {len(grid_coord)} color values, got {len(values)}"
            )
        color_values = normalize_values(values) if normalize else np.clip(values, 0.0, 1.0)

    for center, color_value in zip(centers, color_values):
        box = o3d.geometry.TriangleMesh.create_box(voxel_size, voxel_size, voxel_size)
        box.translate(center - half)
        box.paint_uniform_color(list(cmap(float(color_value))[:3]))
        merged += box

    merged.compute_vertex_normals()
    return merged


def draw_geometries(geometries, **kwargs):
    import open3d as o3d

    print_camera = kwargs.pop("print_camera", False)
    print_camera_key = kwargs.pop("print_camera_key", "P")

    if "lookat" not in kwargs:
        bounds = [
            geometry.get_axis_aligned_bounding_box()
            for geometry in geometries
            if not geometry.is_empty()
        ]
        if bounds:
            min_bound = np.min([bound.get_min_bound() for bound in bounds], axis=0)
            max_bound = np.max([bound.get_max_bound() for bound in bounds], axis=0)
            kwargs["lookat"] = ((min_bound + max_bound) * 0.5).tolist()

    kwargs.setdefault("front", [0.716676, 0.049010, -0.695683])
    kwargs.setdefault("up", [-0.122142, -0.973289, -0.194395])
    kwargs.setdefault("zoom", 0.7)

    if print_camera:
        window_name = kwargs.pop("window_name", "Open3D")
        width = kwargs.pop("width", 1920)
        height = kwargs.pop("height", 1080)
        left = kwargs.pop("left", 50)
        top = kwargs.pop("top", 50)

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(
            window_name=window_name,
            width=width,
            height=height,
            left=left,
            top=top,
        )
        for geometry in geometries:
            vis.add_geometry(geometry)

        view = vis.get_view_control()
        view.set_front(kwargs["front"])
        view.set_up(kwargs["up"])
        view.set_lookat(kwargs["lookat"])
        view.set_zoom(kwargs["zoom"])

        def print_current_camera(vis):
            view = vis.get_view_control()
            print("Current Open3D camera:")
            try:
                print(f'  front = {np.asarray(view.get_front()).round(6).tolist()}')
                print(f'  up = {np.asarray(view.get_up()).round(6).tolist()}')
                print(f'  lookat = {np.asarray(view.get_lookat()).round(6).tolist()}')
                print(f"  zoom = {view.get_zoom():.6f}")
            except AttributeError:
                camera = view.convert_to_pinhole_camera_parameters()
                print("  This Open3D version does not expose front/up getters.")
                print("  extrinsic =")
                print(np.asarray(camera.extrinsic).round(6))
            return False

        key_code = ord(print_camera_key.upper())
        vis.register_key_callback(key_code, print_current_camera)
        print(f"Press {print_camera_key.upper()} in the Open3D window to print camera values.")
        vis.run()
        vis.destroy_window()
        return

    o3d.visualization.draw_geometries(geometries, **kwargs)


BASE_GRID_SIZE = 0.005
SPARSE_COLOR_MODES = ("delta_norm", "after_norm", "channel", "signed_delta")
SERIALIZATION_PATTERNS = ("z", "tz", "hilbert", "thilbert")
SERIALIZATION_TITLES = {
    "z": "Z-order / Morton",
    "tz": "Transposed Z-order / Morton",
    "hilbert": "Hilbert",
    "thilbert": "Transposed Hilbert",
}


def make_sparse_input_features(voxel_points, counts):
    """Simple deterministic voxel features: normalized xyz plus log occupancy."""
    center = voxel_points.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(voxel_points - center, axis=1).max()
    scale = max(float(scale), 1e-6)
    xyz_norm = (voxel_points - center) / scale
    count_feat = np.log1p(counts).reshape(-1, 1)
    count_feat = normalize_values(count_feat[:, 0]).reshape(-1, 1)
    return np.concatenate([xyz_norm, count_feat], axis=1).astype(np.float32)


def sparse_color_values(before, after, color_mode, feature_channel):
    if feature_channel < 0 or feature_channel >= before.shape[1]:
        raise ValueError(
            f"--feature-channel must be in [0, {before.shape[1] - 1}], "
            f"got {feature_channel}"
        )

    if color_mode == "delta_norm":
        return np.linalg.norm(after - before, axis=1)
    if color_mode == "after_norm":
        return np.linalg.norm(after, axis=1)
    if color_mode == "channel":
        return after[:, feature_channel]
    if color_mode == "signed_delta":
        return after[:, feature_channel] - before[:, feature_channel]
    raise ValueError(f"Unknown sparse color mode: {color_mode}")


def _hilbert_axes_to_transpose(coords, bits):
    """Convert integer axes coordinates to Hilbert transpose form."""
    dims = len(coords)
    x = [int(v) for v in coords]
    m = 1 << (bits - 1)

    q = m
    while q > 1:
        p = q - 1
        for i in range(dims):
            if x[i] & q:
                x[0] ^= p
            else:
                t = (x[0] ^ x[i]) & p
                x[0] ^= t
                x[i] ^= t
        q >>= 1

    for i in range(1, dims):
        x[i] ^= x[i - 1]

    t = 0
    q = m
    while q > 1:
        if x[dims - 1] & q:
            t ^= q - 1
        q >>= 1
    for i in range(dims):
        x[i] ^= t

    return x


def hilbert_encode_3d(grid_coord, bits):
    """Encode 3-D integer grid coordinates into Hilbert distances."""
    max_coord = (1 << bits) - 1
    if np.any(grid_coord < 0) or np.any(grid_coord > max_coord):
        raise ValueError(
            f"Hilbert encode expects coordinates in [0, {max_coord}] for bits={bits}"
        )

    distances = np.zeros(len(grid_coord), dtype=np.int64)
    for i, coord in enumerate(grid_coord):
        axes = _hilbert_axes_to_transpose(coord, bits)
        index = 0
        for bit_level in range(bits):
            for axis_idx, axis in enumerate(axes):
                bit = (axis >> bit_level) & 1
                index |= bit << (bit_level * 3 + (2 - axis_idx))
        distances[i] = index
    return distances


def make_serialization_path_lineset(sorted_centers, step=1):
    import open3d as o3d

    if len(sorted_centers) < 2:
        return o3d.geometry.LineSet()

    step = max(1, int(step))
    sampled = sorted_centers[::step]
    if len(sampled) < 2:
        return o3d.geometry.LineSet()

    lines = np.column_stack(
        [
            np.arange(len(sampled) - 1, dtype=np.int32),
            np.arange(1, len(sampled), dtype=np.int32),
        ]
    )
    line_colors = np.tile(np.array([[0.05, 0.05, 0.05]], dtype=np.float64), (len(lines), 1))
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(sampled),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(line_colors)
    return line_set


def serialization_keys(grid_coord, pattern, bits):
    if pattern in ("tz", "thilbert"):
        key_coord = grid_coord[:, [1, 2, 0]]
    else:
        key_coord = grid_coord

    if pattern in ("z", "tz"):
        return morton_encode(key_coord)
    if pattern in ("hilbert", "thilbert"):
        return hilbert_encode_3d(key_coord, bits)
    raise ValueError(f"Unknown serialization pattern: {pattern}")


def visualize_serialization(points, args, resolved_path, pattern):
    grid_coord, _, _, origin = voxelize_to_occupied(points, args.grid_size)
    if len(grid_coord) == 0:
        print(f"No occupied voxels found; skipping {pattern} visualization.")
        return

    shifted = grid_coord - grid_coord.min(axis=0, keepdims=True)
    span = shifted.max(axis=0) + 1
    required_bits = int(np.ceil(np.log2(max(2, int(span.max())))))
    bits = max(required_bits, args.hilbert_bits)
    curve_side = 1 << bits

    keys = serialization_keys(shifted, pattern, bits)
    order = np.argsort(keys)
    centers = origin + (grid_coord.astype(np.float64) + 0.5) * args.grid_size
    sorted_centers = centers[order]
    title = SERIALIZATION_TITLES[pattern]

    print(f"Loaded mug point cloud: {resolved_path}")
    print(
        f"{title} serialization "
        f"| occupied voxels: {len(grid_coord)} "
        f"| curve side: {curve_side} "
        f"| bits/axis: {bits}"
    )
    print(
        f"Rendering voxels colored by {pattern} key and a path showing "
        "1-D traversal order through occupied cells."
    )

    voxel_mesh = voxels_to_gradient_mesh(
        grid_coord,
        args.grid_size,
        origin,
        values=keys,
    )
    path_lines = make_serialization_path_lineset(
        sorted_centers,
        step=args.curve_line_step,
    )
    draw_geometries(
        [voxel_mesh, path_lines],
        window_name=f"{title} 1D serialization over occupied voxels",
        width=1200,
        height=800,
        print_camera=args.print_camera,
    )


def visualize_sparse_conv(points, args, resolved_path):
    try:
        import torch
        from models.backbone_ptv3 import SparseCPE
    except ImportError as exc:
        raise ImportError(
            "Sparse-conv visualization requires torch and spconv. "
            "Install the matching spconv wheel, e.g. `pip install spconv-cu120`."
        ) from exc

    grid_coord, voxel_points, counts, origin = voxelize_to_occupied(
        points, args.grid_size
    )
    features = make_sparse_input_features(voxel_points, counts)

    torch.manual_seed(args.seed)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    device = torch.device(device_name)
    try:
        cpe = SparseCPE(
            channels=features.shape[1], indice_key="viz_sparse_cpe"
        ).to(device)
    except ImportError as exc:
        raise ImportError(
            "Sparse-conv visualization requires spconv. Install the wheel matching "
            "your CUDA/PyTorch setup, e.g. `pip install spconv-cu120`."
        ) from exc
    cpe.eval()

    feat_t = torch.from_numpy(features).unsqueeze(0).to(device)
    grid_t = torch.from_numpy(grid_coord).long().unsqueeze(0).to(device)
    valid_t = torch.ones(1, len(grid_coord), dtype=torch.bool, device=device)

    with torch.no_grad():
        update = cpe(feat_t, grid_t, valid_t)
        after_t = feat_t + update

    before = feat_t.squeeze(0).detach().cpu().numpy()
    after = after_t.squeeze(0).detach().cpu().numpy()
    values = sparse_color_values(before, after, args.sparse_color, args.feature_channel)

    print(f"Loaded mug point cloud: {resolved_path}")
    print(
        "SparseCPE visualization "
        f"| voxels: {len(grid_coord)} "
        f"| color: {args.sparse_color} "
        f"| min/mean/max: {values.min():.4g}/{values.mean():.4g}/{values.max():.4g}"
    )
    print(
        "Note: SubMConv3d keeps the same active voxel coordinates; colors show "
        "feature changes on that fixed grid."
    )
    draw_geometries(
        [voxels_to_gradient_mesh(grid_coord, args.grid_size, origin, values=values)],
        window_name=f"SparseCPE feature color: {args.sparse_color}",
        width=1000,
        height=700,
        print_camera=args.print_camera,
    )


def visualize_voxel_pooling(points, args, resolved_path):
    current_points = points
    current_grid, origin = quantize_like_ptv3(current_points, args.grid_size)

    input_grid = np.unique(current_grid, axis=0)
    print(f"Loaded mug point cloud: {resolved_path}")
    print(f"Visualizing voxelized mug (grid size {args.grid_size} m)...")
    draw_geometries(
        [voxels_to_gradient_mesh(input_grid, args.grid_size, origin)],
        window_name="Input voxelization",
        width=800,
        height=600,
        print_camera=args.print_camera,
    )

    current_grid_size = args.grid_size

    for stage in range(1, args.stages + 1):
        current_grid_size *= 2
        current_points, current_grid = voxel_pool_bitshift(
            current_points,
            current_grid,
            pool_shift=1,
        )
        num_p = len(current_grid)
        print(f"Stage {stage}: grid {current_grid_size:.2f} m | occupied voxels: {num_p}")

        draw_geometries(
            [voxels_to_gradient_mesh(current_grid, current_grid_size, origin)],
            window_name=f"Encoder stage {stage} (grid {current_grid_size:.2f} m)",
            width=1000,
            height=700,
            print_camera=args.print_camera,
        )

    print("\nVisualization complete. All stages processed.")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize Mug voxel pooling, sparse-conv feature changes, "
            "and PTv3 space-filling-curve serialization."
        )
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=DEFAULT_MUG_PATH,
        help="Path to a generated .npz sample containing a `points` array.",
    )
    parser.add_argument(
        "--mode",
        choices=("pooling", "sparse", *SERIALIZATION_PATTERNS, "all"),
        default="pooling",
        help="Which visualization to run.",
    )
    parser.add_argument(
        "--grid-size",
        type=float,
        default=BASE_GRID_SIZE,
        help="Base voxel size in meters.",
    )
    parser.add_argument(
        "--stages",
        type=int,
        default=4,
        help="Number of voxel-pooling stages to show.",
    )
    parser.add_argument(
        "--sparse-color",
        choices=SPARSE_COLOR_MODES,
        default="delta_norm",
        help=(
            "How to color the fixed sparse-conv voxel grid: "
            "delta_norm=||after-before||, after_norm=||after||, "
            "channel=after[channel], signed_delta=after[channel]-before[channel]."
        ),
    )
    parser.add_argument(
        "--feature-channel",
        type=int,
        default=0,
        help="Feature channel used by --sparse-color channel/signed_delta.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the untrained sparse CPE weights.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Device for sparse-conv visualization.",
    )
    parser.add_argument(
        "--hilbert-bits",
        type=int,
        default=0,
        help=(
            "Minimum Hilbert bits per axis. "
            "0 auto-selects based on occupied voxel span."
        ),
    )
    parser.add_argument(
        "--curve-line-step",
        "--hilbert-line-step",
        dest="curve_line_step",
        type=int,
        default=1,
        help=(
            "Connect every N-th voxel along the serialization order to reduce "
            "path density (1 keeps all)."
        ),
    )
    parser.add_argument(
        "--print-camera",
        action="store_true",
        help="Press P in the Open3D window to print the current camera values.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    points, resolved_path = load_mug_points(args.path)

    if args.mode in ("pooling", "all"):
        visualize_voxel_pooling(points, args, resolved_path)
    if args.mode in ("sparse", "all"):
        visualize_sparse_conv(points, args, resolved_path)
    if args.mode == "all":
        for pattern in SERIALIZATION_PATTERNS:
            visualize_serialization(points, args, resolved_path, pattern)
    elif args.mode in SERIALIZATION_PATTERNS:
        visualize_serialization(points, args, resolved_path, args.mode)


if __name__ == "__main__":
    main()
