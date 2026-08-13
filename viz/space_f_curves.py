from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np  # type: ignore[reportMissingImports]

# Script-mode bootstrap only (`python viz/space_f_curves.py`).
if __package__ is None:  # pragma: no cover
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from models.serialization_numpy import SERIALIZATION_PATTERNS, serialization_keys

PATTERNS = SERIALIZATION_PATTERNS
PATTERN_TITLES = {
    "z": "Z / Morton",
    "tz": "Transposed Z / Morton",
    "hilbert": "Hilbert",
    "thilbert": "Transposed Hilbert",
}


def grid_coords(bits: int) -> np.ndarray:
    """Return all integer coordinates in a 2**bits cube."""
    side = 1 << bits
    return np.indices((side, side, side), dtype=np.int64).reshape(3, -1).T


def curve_points(pattern: str, bits: int) -> np.ndarray:
    """Return normalized voxel-center points in PTv3 serialization order."""
    coords = grid_coords(bits)
    order = np.argsort(serialization_keys(coords, pattern, bits), kind="stable")
    side = 1 << bits
    return (coords[order].astype(np.float64) + 0.5) / side


def draw_unit_cube(ax, color: str = "0.2", linewidth: float = 0.45) -> None:
    corners = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.float64,
    )
    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    for start, end in edges:
        segment = corners[[start, end]]
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=color, lw=linewidth)


def draw_edge_scale(ax, bits: int, color: str = "0.1") -> None:
    """Draw a compact x-edge ruler in voxel units for the current iteration."""
    side = 1 << bits
    y = -0.08
    z = -0.02
    tick_len = 0.035
    ticks = (0.0, 0.5, 1.0)
    labels = ("0", str(side // 2), str(side))

    ax.plot([0, 1], [y, y], [z, z], color=color, lw=0.8)
    for x, label in zip(ticks, labels):
        ax.plot([x, x], [y, y], [z, z + tick_len], color=color, lw=0.8)
        ax.text(
            x,
            y - 0.035,
            z - 0.01,
            label,
            ha="center",
            va="top",
            fontsize=7,
            color=color,
        )

    ax.text(
        0.5,
        y - 0.095,
        z - 0.01,
        f"edge = {side} cells",
        ha="center",
        va="top",
        fontsize=7,
        color=color,
    )


def plot_curve(
    ax,
    points: np.ndarray,
    title: str,
    line_width: float,
    bits: int,
) -> None:
    ax.plot(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        color="navy",
        lw=line_width,
        solid_capstyle="round",
    )
    draw_unit_cube(ax)
    draw_edge_scale(ax, bits)
    ax.set_title(title, y=-0.12, fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.20, 1.02)
    ax.set_zlim(-0.05, 1.02)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=24, azim=-55)
    ax.set_axis_off()


def selected_patterns(pattern: str) -> tuple[str, ...]:
    if pattern == "all":
        return PATTERNS
    if pattern not in PATTERNS:
        raise ValueError(f"Unknown pattern: {pattern}")
    return (pattern,)


def make_figure(args: argparse.Namespace):
    import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]

    patterns = selected_patterns(args.pattern)
    iterations = tuple(args.iterations)
    nrows = len(patterns)
    ncols = len(iterations)
    fig = plt.figure(figsize=(args.cell_size * ncols, args.cell_size * nrows))

    for row, pattern in enumerate(patterns):
        for col, bits in enumerate(iterations):
            ax = fig.add_subplot(nrows, ncols, row * ncols + col + 1, projection="3d")
            pts = curve_points(pattern, bits)
            if len(patterns) == 1:
                title = f"{bits} iteration{'s' if bits != 1 else ''}"
            else:
                title = (
                    f"{PATTERN_TITLES[pattern]}\n"
                    f"{bits} iteration{'s' if bits != 1 else ''}"
                )
            plot_curve(ax, pts, title, args.line_width, bits)

    fig.suptitle(args.title, fontsize=13)
    fig.tight_layout()
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate PTv3-style 3-D space-filling-curve diagrams for Z, "
            "transposed Z, Hilbert, and transposed Hilbert orderings."
        )
    )
    parser.add_argument(
        "--pattern",
        choices=(*PATTERNS, "all"),
        default="hilbert",
        help="Serialization pattern to draw. Use 'all' for a pattern comparison grid.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="Iteration counts / bits per axis to render.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("space_f_curves.png"),
        help="PNG/PDF/SVG output path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive Matplotlib window after saving.",
    )
    parser.add_argument(
        "--line-width",
        type=float,
        default=1.1,
        help="Curve line width.",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=3.2,
        help="Figure size, in inches, allocated per subplot.",
    )
    parser.add_argument(
        "--title",
        default="PTv3 Space-Filling Curve Orderings",
        help="Figure title.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(bits < 1 or bits > 10 for bits in args.iterations):
        raise ValueError("All iterations must be in [1, 10]; PTv3 uses 10-bit axes.")

    if not args.show:
        import matplotlib  # type: ignore[reportMissingImports]

        matplotlib.use("Agg")

    fig = make_figure(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    print(f"Saved space-filling curve figure to {args.output}")

    if args.show:
        import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]

        plt.show()


if __name__ == "__main__":
    main()
