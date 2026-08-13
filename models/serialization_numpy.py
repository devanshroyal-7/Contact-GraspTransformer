"""NumPy space-filling-curve helpers for visualization / offline tooling.

Torch training implementations live in ``models.backbone_ptv3`` (different
runtime: batched tensors, silent OOB masking for speed). Algorithm comments
there should continue to point at this module as the readable NumPy reference.
"""

from __future__ import annotations

import numpy as np

# Canonical SFC pattern order shared by PTv3 training + viz tooling.
SERIALIZATION_PATTERNS: tuple[str, ...] = ("z", "tz", "hilbert", "thilbert")


def _part1by2(n: np.ndarray) -> np.ndarray:
    """Spread the low 10 bits by inserting two 0-bits between each bit."""
    n = n.astype(np.int64) & 0x000003FF
    n = (n ^ (n << 16)) & 0xFF0000FF
    n = (n ^ (n << 8)) & 0x0F00F00F
    n = (n ^ (n << 4)) & 0xC30C30C3
    n = (n ^ (n << 2)) & 0x49249249
    return n


def morton_encode(grid_coord: np.ndarray) -> np.ndarray:
    """NumPy Morton / Z-order encode for integer grid coordinates (N, 3)."""
    x, y, z = grid_coord[:, 0], grid_coord[:, 1], grid_coord[:, 2]
    return _part1by2(x) | (_part1by2(y) << 1) | (_part1by2(z) << 2)


def _hilbert_axes_to_transpose(
    coords: np.ndarray | list[int] | tuple[int, ...], bits: int
) -> list[int]:
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


def hilbert_encode_3d(grid_coord: np.ndarray, bits: int) -> np.ndarray:
    """Encode 3-D integer grid coordinates into Hilbert distances.

    Contract: ``bits >= 1`` and coordinates in ``[0, 2**bits - 1]``.
    Out-of-range inputs raise ``ValueError`` (unlike the training torch path
    in ``models.backbone_ptv3``, which masks coordinates for speed).
    """
    if bits < 1:
        raise ValueError("bits must be >= 1")

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


def serialization_keys(
    grid_coord: np.ndarray, pattern: str, bits: int
) -> np.ndarray:
    """Return per-voxel sort keys for a PTv3 serialization pattern."""
    if pattern in ("tz", "thilbert"):
        key_coord = grid_coord[:, [1, 2, 0]]
    else:
        key_coord = grid_coord

    if pattern in ("z", "tz"):
        return morton_encode(key_coord)
    if pattern in ("hilbert", "thilbert"):
        return hilbert_encode_3d(key_coord, bits)
    raise ValueError(f"Unknown serialization pattern: {pattern}")
