import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ───────────────────── space-filling curve helpers ───────────────────────────

def _part1by2(n: torch.Tensor) -> torch.Tensor:
    """Spread the low 10 bits of *n* by inserting two 0-bits between each."""
    n = n & 0x000003FF
    n = (n ^ (n << 16)) & 0xFF0000FF
    n = (n ^ (n <<  8)) & 0x0F00F00F
    n = (n ^ (n <<  4)) & 0xC30C30C3
    n = (n ^ (n <<  2)) & 0x49249249
    return n


def morton_encode(xyz_int: torch.Tensor) -> torch.Tensor:
    """Z-order / Morton code from integer grid coords (..., 3) -> (...)."""
    x, y, z = xyz_int[..., 0], xyz_int[..., 1], xyz_int[..., 2]
    return _part1by2(x) | (_part1by2(y) << 1) | (_part1by2(z) << 2)


# ────────────────────── Hilbert curve (3D) ───────────────────────────────────

_HILBERT_BITS = 10

def _rot3d(n: int, coords: torch.Tensor, rx: torch.Tensor,
           ry: torch.Tensor, rz: torch.Tensor):
    """In-place rotation/reflection for 3D Hilbert encoding."""
    mask_rz = rz == 0
    mask_ry = ry == 0
    mask_rx = rx == 0

    x, y, z = coords[..., 0].clone(), coords[..., 1].clone(), coords[..., 2].clone()

    swap_xz = mask_rz & mask_ry
    x2 = x.clone()
    x[swap_xz] = (n - 1 - z[swap_xz])
    z[swap_xz] = (n - 1 - x2[swap_xz])

    swap_yz = mask_rz & (~mask_ry)
    y2 = y.clone()
    y[swap_yz] = (n - 1 - z[swap_yz])
    z[swap_yz] = (n - 1 - y2[swap_yz])

    flip_x = (~mask_rz) & mask_rx
    x[flip_x] = n - 1 - x[flip_x]

    coords[..., 0] = x
    coords[..., 1] = y
    coords[..., 2] = z


def hilbert_encode_3d(xyz_int: torch.Tensor, bits: int = _HILBERT_BITS) -> torch.Tensor:
    """Encode integer grid coordinates to 3D Hilbert curve index.

    Uses the standard iterative algorithm based on quadrant rotation.
    """
    x = xyz_int[..., 0].clone()
    y = xyz_int[..., 1].clone()
    z = xyz_int[..., 2].clone()

    d = torch.zeros_like(x)
    coords = torch.stack([x, y, z], dim=-1)

    s = (1 << (bits - 1))
    while s > 0:
        rx = ((coords[..., 0] & s) > 0).long()
        ry = ((coords[..., 1] & s) > 0).long()
        rz = ((coords[..., 2] & s) > 0).long()

        level_val = s * s * (3 * rx ^ (ry if isinstance(ry, int) else ry))
        level_val = s * s * (4 * rz + (1 + 2 * ry - rx) * rz + (3 * rx) * (1 - rz) ^ ry)
        d += s * s * s * ((4 * rz) + (2 * ry * (1 - rz) + ry * rz) + (rx ^ ry))

        _rot3d(s, coords, rx, ry, rz)
        s >>= 1

    return d


def _simple_hilbert_encode_3d(xyz_int: torch.Tensor) -> torch.Tensor:
    """Practical 3D Hilbert-like encoding via interleaved Gray codes.

    This produces a space-filling curve with better locality than Morton
    by applying Gray code to each axis before interleaving.
    """
    def to_gray(n: torch.Tensor) -> torch.Tensor:
        return n ^ (n >> 1)

    x = to_gray(xyz_int[..., 0])
    y = to_gray(xyz_int[..., 1])
    z = to_gray(xyz_int[..., 2])
    return _part1by2(x) | (_part1by2(y) << 1) | (_part1by2(z) << 2)


# ───────────────── multi-pattern serialization ───────────────────────────────

PATTERNS = ["z", "tz", "hilbert", "thilbert"]


def _quantize(xyz: torch.Tensor, grid_size: float) -> torch.Tensor:
    """Quantize to non-negative integer grid coordinates."""
    q = torch.floor(xyz / grid_size).long()
    return q - q.min(dim=1, keepdim=True).values


def _sort_by_keys(keys: torch.Tensor):
    """Return (sort_idx, unsort_idx) from sorting keys along dim=-1."""
    sort_idx = keys.argsort(dim=-1, stable=True)
    unsort_idx = sort_idx.argsort(dim=-1)
    return sort_idx, unsort_idx


def serialize_point_cloud_multi(
    xyz: torch.Tensor, grid_size: float = 0.01
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Compute 4 serialization orderings for a (B, N, 3) point cloud.

    Returns list of (sort_idx, unsort_idx) for patterns:
    [Z-order, Trans-Z, Hilbert (Gray-interleave), Trans-Hilbert].
    """
    q = _quantize(xyz, grid_size)
    q_trans = q[..., [1, 2, 0]]

    orderings = []

    orderings.append(_sort_by_keys(morton_encode(q)))
    orderings.append(_sort_by_keys(morton_encode(q_trans)))
    orderings.append(_sort_by_keys(_simple_hilbert_encode_3d(q)))
    orderings.append(_sort_by_keys(_simple_hilbert_encode_3d(q_trans)))

    return orderings


# ──────────────────────── drop path ──────────────────────────────────────────

class DropPath(nn.Module):
    """Stochastic depth (per-sample drop) for residual blocks."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device,
                                          dtype=x.dtype))
        return x * mask / keep


# ──────────────────────── attention block ────────────────────────────────────

class GridWindowAttention(nn.Module):
    """Pre-norm windowed self-attention with xCPE and drop path."""

    def __init__(self, channels: int, num_heads: int = 4,
                 window_size: int = 64, drop_path: float = 0.0):
        super().__init__()
        self.window_size = window_size

        # xCPE: depthwise conv + skip (replaces linear pos_enc)
        self.cpe = nn.Conv1d(channels, channels, kernel_size=3, padding=1,
                             groups=channels, bias=True)

        self.norm1 = nn.LayerNorm(channels)
        self.mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (B, N, C)"""
        B, N, C = feat.shape
        W = self.window_size

        # xCPE: positional encoding via depthwise conv on serialized sequence
        feat = feat + self.cpe(feat.transpose(1, 2)).transpose(1, 2)

        pad_len = (W - N % W) % W
        if pad_len:
            feat = F.pad(feat, (0, 0, 0, pad_len))

        Np = feat.shape[1]
        nw = Np // W
        wf = feat.reshape(B * nw, W, C)

        # pre-norm attention
        res = wf
        wf = self.norm1(wf)
        wf = res + self.drop_path(self.mha(wf, wf, wf)[0])

        # pre-norm FFN
        res = wf
        wf = self.norm2(wf)
        wf = res + self.drop_path(self.mlp(wf))

        return wf.reshape(B, Np, C)[:, :N, :]


# ─────────────────────── transition layers ───────────────────────────────────

class TransitionDown(nn.Module):
    """Grid pooling along the serialized sequence via strided 1-D conv."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 4):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=stride, stride=stride)

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor):
        """xyz: (B, N, 3), feat: (B, N, C) -> (B, N//stride, 3), (B, N//stride, C_out)"""
        B, N, _ = xyz.shape
        s = self.stride
        trim = (N // s) * s
        xyz_t = xyz[:, :trim, :]
        feat_t = feat[:, :trim, :]
        xyz_down = xyz_t.reshape(B, trim // s, s, 3).mean(dim=2)
        feat_down = self.conv(feat_t.transpose(1, 2)).transpose(1, 2)
        return xyz_down, feat_down


class TransitionUp(nn.Module):
    """Sequence upsample + skip connection via addition."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj_up   = nn.Linear(in_channels, out_channels)
        self.proj_skip = nn.Linear(out_channels, out_channels)

    def forward(self, feat_skip: torch.Tensor, feat_coarse: torch.Tensor):
        up = F.interpolate(feat_coarse.transpose(1, 2),
                           size=feat_skip.shape[1],
                           mode="nearest").transpose(1, 2)
        return self.proj_up(up) + self.proj_skip(feat_skip)


# ──────────────────────── full backbone ──────────────────────────────────────

class PTv3Wrapper(nn.Module):
    """Point Transformer V3 backbone with multi-pattern serialization.

    Implements the core PTv3 mechanisms:
    - 4 serialization patterns (Z, Trans-Z, Hilbert, Trans-Hilbert)
    - Shuffle Order: random pattern assignment per block per forward pass
    - xCPE (depthwise conv positional encoding)
    - Stochastic depth (drop path)
    - U-Net encoder-decoder with windowed attention
    """

    def __init__(self, out_channels: int = 128, grid_size: float = 0.01,
                 window_size: int = 64, drop_path_rate: float = 0.3):
        super().__init__()
        self.grid_size = grid_size
        self.num_patterns = len(PATTERNS)
        self.in_proj = nn.Linear(3, 32)

        num_blocks = 8  # 4 encoder + 4 decoder
        dp_rates = [drop_path_rate * i / (num_blocks - 1)
                     for i in range(num_blocks)]

        self.enc1  = GridWindowAttention(32, num_heads=4, window_size=window_size,
                                         drop_path=dp_rates[0])
        self.down1 = TransitionDown(32, 64, stride=4)

        self.enc2  = GridWindowAttention(64, num_heads=4, window_size=window_size,
                                         drop_path=dp_rates[1])
        self.down2 = TransitionDown(64, 128, stride=4)

        self.enc3  = GridWindowAttention(128, num_heads=8, window_size=window_size,
                                         drop_path=dp_rates[2])
        self.down3 = TransitionDown(128, 256, stride=4)

        self.enc4  = GridWindowAttention(256, num_heads=8, window_size=window_size,
                                         drop_path=dp_rates[3])

        self.up3  = TransitionUp(256, 128)
        self.dec3 = GridWindowAttention(128, num_heads=8, window_size=window_size,
                                         drop_path=dp_rates[4])

        self.up2  = TransitionUp(128, 64)
        self.dec2 = GridWindowAttention(64, num_heads=4, window_size=window_size,
                                         drop_path=dp_rates[5])

        self.up1  = TransitionUp(64, 32)
        self.dec1 = GridWindowAttention(32, num_heads=4, window_size=window_size,
                                         drop_path=dp_rates[6])

        self.out_proj = nn.Linear(32, out_channels)

        self._block_list = [
            self.enc1, self.enc2, self.enc3, self.enc4,
            self.dec3, self.dec2, self.dec1,
        ]

    def _reorder(self, feat: torch.Tensor, idx: torch.Tensor,
                 batch_idx: torch.Tensor) -> torch.Tensor:
        """Reorder features by sort or unsort indices."""
        return feat[batch_idx, idx]

    def _pick_ordering(self, orderings_for_level, block_idx, pattern_assignment):
        """Select a serialization ordering for a given block."""
        pidx = pattern_assignment[block_idx].item()
        return orderings_for_level[pidx]

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """xyz: (B, N, 3) -> features (B, C, N)"""
        B, N, _ = xyz.shape
        device = xyz.device

        # Shuffle Order: randomly assign a pattern to each attention block
        n_blocks = len(self._block_list)
        if self.training:
            pattern_assignment = torch.randint(0, self.num_patterns, (n_blocks,))
        else:
            pattern_assignment = torch.arange(n_blocks) % self.num_patterns

        def make_batch_idx(n):
            return torch.arange(B, device=device).view(B, 1).expand(B, n)

        def apply_block(block, feat, level_xyz, block_idx):
            """Serialize -> attend -> unserialize for one block."""
            n = feat.shape[1]
            ords = serialize_point_cloud_multi(level_xyz, self.grid_size)
            s_idx, u_idx = self._pick_ordering(ords, block_idx, pattern_assignment)
            bi = make_batch_idx(n)
            s_feat = self._reorder(feat, s_idx, bi)
            s_feat = block(s_feat)
            return self._reorder(s_feat, u_idx, bi)

        feat = self.in_proj(xyz)  # (B, N, 32)

        # ── encoder ──
        e1 = apply_block(self.enc1, feat, xyz, 0)

        xyz2, e1_down = self.down1(xyz, e1)
        e2 = apply_block(self.enc2, e1_down, xyz2, 1)

        xyz3, e2_down = self.down2(xyz2, e2)
        e3 = apply_block(self.enc3, e2_down, xyz3, 2)

        xyz4, e3_down = self.down3(xyz3, e3)
        e4 = apply_block(self.enc4, e3_down, xyz4, 3)

        # ── decoder ──
        d3 = self.up3(e3, e4)
        d3 = apply_block(self.dec3, d3, xyz3, 4)

        d2 = self.up2(e2, d3)
        d2 = apply_block(self.dec2, d2, xyz2, 5)

        d1 = self.up1(e1, d2)
        d1 = apply_block(self.dec1, d1, xyz, 6)

        out = self.out_proj(d1)
        return out.transpose(1, 2)  # (B, C, N)
