"""Point Transformer V3 backbone.

Implements the PTv3 mechanisms with pragmatic simplifications so the backbone
still accepts a dense (B, N, 3) point cloud and returns (B, C, N) per-point
features.  Key mechanisms:

- Multi-pattern space-filling-curve serialization (Z / Trans-Z / 3-D Hilbert
  and its transposed variant, the latter via Skilling's AxesToTranspose).
- Shuffle Order: per-forward permutation of the ordering list, each block
  uses a fixed ``order_index`` within its stage (matches the Pointcept
  reference semantics).
- xCPE: configurable conditional positional encoding -- 1-D depthwise conv
  (default), k-NN neighborhood depthwise conv, or spconv submanifold 3-D
  convolution.
- Voxel-grid pooling / unpooling via Morton-code clustering + scatter; points
  are padded back to a dense tensor with a ``valid`` mask so the whole
  backbone keeps the (B, N, C) contract.
- Windowed self-attention with a ``key_padding_mask`` so padded tokens never
  participate in softmax.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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


HILBERT_BITS = 10  # 10 bits per axis -> 30-bit Hilbert index, fits in int64.


def hilbert_encode_3d(xyz_int: torch.Tensor,
                      bits: int = HILBERT_BITS) -> torch.Tensor:
    """3-D Hilbert curve index from integer grid coords (..., 3) -> (...).

    Implements Skilling's ``AxesToTranspose`` algorithm (J. Skilling,
    "Programming the Hilbert Curve", AIP Conf. Proc. 707, 381, 2004),
    followed by a standard bit interleave from the transpose representation
    to the Hilbert index.

    ``bits`` controls the per-axis precision: values must satisfy
    ``0 <= coord < 2**bits``.  At ``bits=10`` the index fits in 30 bits.
    """
    assert xyz_int.shape[-1] == 3, "expected (..., 3) coord tensor"
    x = (xyz_int[..., 0] & ((1 << bits) - 1)).clone().long()
    y = (xyz_int[..., 1] & ((1 << bits) - 1)).clone().long()
    z = (xyz_int[..., 2] & ((1 << bits) - 1)).clone().long()

    n = 3
    M = 1 << (bits - 1)

    # Inverse undo walk (MSB -> one-above-LSB). For each axis i:
    # if bit Q of X[i] is set -> X[0] ^= (Q-1) (invert low bits of X[0])
    # else                    -> exchange low bits of X[0] and X[i]
    Q = M
    while Q > 1:
        P = Q - 1
        for i in range(n):
            Xi = (x, y, z)[i]
            bit_set = (Xi & Q) != 0
            t = (x ^ Xi) & P
            x_if_set = x ^ P
            x_else = x ^ t
            x = torch.where(bit_set, x_if_set, x_else)
            if i == 1:
                y = torch.where(bit_set, y, y ^ t)
            elif i == 2:
                z = torch.where(bit_set, z, z ^ t)
        Q >>= 1

    # Gray encode: X[i] ^= X[i-1] for i = 1..n-1
    y = y ^ x
    z = z ^ y

    # Trailing gray undo. t accumulates (Q-1) for every Q where X[n-1] bit is set.
    t = torch.zeros_like(x)
    Q = M
    while Q > 1:
        bit_set = (z & Q) != 0
        t = torch.where(bit_set, t ^ (Q - 1), t)
        Q >>= 1
    x = x ^ t
    y = y ^ t
    z = z ^ t

    # Interleave: bit (j*n + (n-1-i)) of index = bit j of X[i]. X[0] is MSB of
    # each level's triplet, X[n-1] is LSB, with level j=0 being the LSB level.
    index = torch.zeros_like(x)
    axes = (x, y, z)
    for j in range(bits):
        for i in range(n):
            bit = (axes[i] >> j) & 1
            index = index | (bit << (j * n + (n - 1 - i)))
    return index


# ───────────────── multi-pattern serialization ───────────────────────────────

PATTERNS = ("z", "tz", "hilbert", "thilbert")
NUM_PATTERNS = len(PATTERNS)


def _quantize(xyz: torch.Tensor, grid_size: float) -> torch.Tensor:
    """Quantize to non-negative integer grid coordinates, per batch.

    xyz: (B, N, 3) float -> (B, N, 3) int64 with per-batch min subtracted.
    """
    q = torch.floor(xyz / grid_size).long()
    return q - q.min(dim=1, keepdim=True).values


def _sort_by_keys(keys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (sort_idx, unsort_idx) from sorting keys along dim=-1."""
    sort_idx = keys.argsort(dim=-1, stable=True)
    unsort_idx = sort_idx.argsort(dim=-1)
    return sort_idx, unsort_idx


def compute_orderings(
    grid_coord: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Compute the 4 serialization orderings for `grid_coord` (B, N, 3) int.

    If `valid` is provided, invalid points receive a large sentinel key so they
    sort to the end of every ordering (keeping real points contiguous).
    Returns list of (sort_idx, unsort_idx) pairs in ``PATTERNS`` order.
    """
    gc = grid_coord
    gc_trans = gc[..., [1, 2, 0]]

    keys_list = [
        morton_encode(gc),
        morton_encode(gc_trans),
        hilbert_encode_3d(gc),
        hilbert_encode_3d(gc_trans),
    ]

    if valid is not None:
        sentinel = torch.iinfo(keys_list[0].dtype).max
        invalid = ~valid
        keys_list = [torch.where(invalid, torch.full_like(k, sentinel), k)
                     for k in keys_list]

    return [_sort_by_keys(k) for k in keys_list]


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


# ──────────────────────── CPE variants ───────────────────────────────────────

class Conv1DCPE(nn.Module):
    """xCPE via depthwise 1-D conv along the serialized sequence.

    Cheap, dependency-free, but only positional along the 1-D curve --- not
    a true 3-D neighborhood operator.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1,
                              groups=channels, bias=True)
        self.proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, feat: torch.Tensor, grid_coord: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        # feat: (B, N, C). grid_coord/valid unused here.
        out = self.conv(feat.transpose(1, 2)).transpose(1, 2)
        return self.norm(self.proj(out))


class KNNCPE(nn.Module):
    """xCPE via a depthwise k-NN neighborhood aggregation in 3-D.

    Approximates the reference's ``spconv.SubMConv3d(k=3)`` without adding a
    sparse-conv dependency.  Neighbors are computed on ``grid_coord`` (per
    batch, per forward) and features are aggregated with per-neighbor
    depthwise weights.
    """

    def __init__(self, channels: int, k: int = 16):
        super().__init__()
        self.k = k
        self.weight = nn.Parameter(torch.randn(k, channels) * 0.02)
        self.proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, feat: torch.Tensor, grid_coord: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        # feat: (B, N, C); grid_coord: (B, N, 3) int; valid: (B, N) bool
        B, N, C = feat.shape
        gc = grid_coord.float()

        # Pairwise squared distance in grid-coord space. Block invalid points
        # from being selected as neighbors by setting their distance to +inf
        # when they appear as the "source" in a query. Invalid query rows are
        # harmless because their output will be masked downstream.
        dists = torch.cdist(gc, gc, p=2.0)   # (B, N, N)
        if valid is not None:
            inv = ~valid                     # (B, N)
            dists = dists.masked_fill(inv.unsqueeze(1), float('inf'))

        k = min(self.k, N)
        _, knn_idx = dists.topk(k, dim=-1, largest=False)  # (B, N, k)

        idx = knn_idx.unsqueeze(-1).expand(-1, -1, -1, C)  # (B, N, k, C)
        gathered = torch.gather(
            feat.unsqueeze(1).expand(-1, N, -1, -1), 2, idx
        )  # (B, N, k, C)

        # Depthwise per-neighbor weight (k, C) -> sum over k.
        # If k got truncated below self.k, slice the weight accordingly.
        w = self.weight[:k]                  # (k, C)
        out = (gathered * w).sum(dim=2)      # (B, N, C)

        return self.norm(self.proj(out))


class SparseCPE(nn.Module):
    """xCPE via spconv submanifold 3-D convolution (matches reference).

    Requires `spconv` to be importable at construction time.  If unavailable,
    select ``cpe_mode='knn'`` or ``'conv1d'`` instead.
    """

    def __init__(self, channels: int, indice_key: str):
        super().__init__()
        try:
            import spconv.pytorch as spconv  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "cpe_mode='sparse3d' requires `spconv`. "
                "Install e.g. `pip install spconv-cu120` or switch "
                "cpe_mode to 'knn' / 'conv1d'."
            ) from exc

        import spconv.pytorch as spconv
        self._spconv = spconv
        self.indice_key = indice_key
        self.conv = spconv.SubMConv3d(channels, channels, kernel_size=3,
                                      bias=True, indice_key=indice_key)
        self.proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, feat: torch.Tensor, grid_coord: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        # feat: (B, N, C); grid_coord: (B, N, 3) int; valid: (B, N) bool
        B, N, C = feat.shape

        # Flatten valid points to (M, C) with spconv indices (batch, x, y, z).
        indices_list = []
        feats_list = []
        for b in range(B):
            v = valid[b]
            gc = grid_coord[b][v]                 # (Nv, 3)
            ft = feat[b][v]                       # (Nv, C)
            batch_col = torch.full((gc.shape[0], 1), b,
                                   dtype=torch.int32, device=gc.device)
            indices_list.append(torch.cat([batch_col, gc.int()], dim=1))
            feats_list.append(ft)
        indices = torch.cat(indices_list, dim=0).contiguous()
        feats = torch.cat(feats_list, dim=0).contiguous()

        spatial_shape = (
            grid_coord[..., 0].max().item() + 1,
            grid_coord[..., 1].max().item() + 1,
            grid_coord[..., 2].max().item() + 1,
        )
        sp_tensor = self._spconv.SparseConvTensor(
            features=feats, indices=indices,
            spatial_shape=spatial_shape, batch_size=B,
        )
        out = self.conv(sp_tensor)
        out_feats = self.norm(self.proj(out.features))

        # Scatter back to the padded (B, N, C) tensor.
        new_feat = torch.zeros_like(feat)
        offsets = [0]
        for b in range(B):
            offsets.append(offsets[-1] + int(valid[b].sum().item()))
        for b in range(B):
            v = valid[b]
            new_feat[b][v] = out_feats[offsets[b]:offsets[b + 1]]
        return new_feat


def _make_cpe(cpe_mode: str, channels: int, *, indice_key: str,
              knn_k: int = 16) -> nn.Module:
    if cpe_mode == "conv1d":
        return Conv1DCPE(channels)
    if cpe_mode == "knn":
        return KNNCPE(channels, k=knn_k)
    if cpe_mode == "sparse3d":
        return SparseCPE(channels, indice_key=indice_key)
    raise ValueError(f"Unknown cpe_mode: {cpe_mode}")


# ──────────────────────── attention block ────────────────────────────────────

class GridWindowAttention(nn.Module):
    """Pre-norm windowed self-attention with xCPE, drop path, and valid mask.

    Operates on (B, N, C) with a companion (B, N) ``valid`` mask and a
    (B, N, 3) ``grid_coord``.  Attention windows are formed along the current
    serialized sequence; padded tokens (real points that fall in the window
    tail, or points invalidated by upstream pooling) are masked out via
    ``key_padding_mask`` so they don't participate in softmax.
    """

    def __init__(self, channels: int, num_heads: int = 4,
                 window_size: int = 256, drop_path: float = 0.0,
                 cpe_mode: str = "knn", knn_k: int = 16,
                 indice_key: Optional[str] = None):
        super().__init__()
        self.window_size = window_size
        self.cpe = _make_cpe(cpe_mode, channels,
                             indice_key=(indice_key or "cpe"),
                             knn_k=knn_k)
        self.norm1 = nn.LayerNorm(channels)
        self.mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        feat: torch.Tensor,          # (B, N, C) -- in serialized order
        valid: torch.Tensor,         # (B, N) bool -- in serialized order
        grid_coord: torch.Tensor,    # (B, N, 3) int -- in serialized order
    ) -> torch.Tensor:
        B, N, C = feat.shape
        W = self.window_size

        # xCPE residual: depends on mode. grid_coord & valid are passed so
        # neighborhood CPE variants can see real 3-D neighbors.
        feat = feat + self.cpe(feat, grid_coord, valid)

        pad_len = (W - N % W) % W
        if pad_len:
            feat = F.pad(feat, (0, 0, 0, pad_len))
            valid_pad = F.pad(valid, (0, pad_len), value=False)
        else:
            valid_pad = valid

        Np = feat.shape[1]
        nw = Np // W
        wf = feat.reshape(B * nw, W, C)
        key_padding_mask = (~valid_pad).reshape(B * nw, W)

        # Some windows may be entirely padded. Give them at least one "real"
        # slot so MHA's softmax doesn't NaN; their output is discarded later.
        all_padded = key_padding_mask.all(dim=-1)
        if all_padded.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_padded, 0] = False

        # pre-norm attention
        res = wf
        wf = self.norm1(wf)
        attn_out, _ = self.mha(wf, wf, wf,
                               key_padding_mask=key_padding_mask,
                               need_weights=False)
        wf = res + self.drop_path(attn_out)

        # pre-norm FFN
        res = wf
        wf = self.norm2(wf)
        wf = res + self.drop_path(self.mlp(wf))

        return wf.reshape(B, Np, C)[:, :N, :]


# ─────────────────────── transition layers ───────────────────────────────────

def _voxel_pool(
    xyz: torch.Tensor,           # (B, N, 3) float
    grid_coord: torch.Tensor,    # (B, N, 3) int
    feat: torch.Tensor,          # (B, N, C)
    valid: torch.Tensor,         # (B, N) bool
    pool_shift: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    """Voxel-grid pool by shifting grid coords right by ``pool_shift`` bits.

    Points sharing the coarsened voxel are averaged. Output tensors are padded
    up to the max number of clusters across the batch. Returns:
        new_xyz: (B, M, 3) float
        new_grid: (B, M, 3) int
        new_feat: (B, M, C)
        new_valid: (B, M) bool
        inverse: (B, N) long -- fine -> coarse index (clamped for padded)
        M: int scalar (max cluster count across batch)
    """
    B, N, C = feat.shape
    gc_coarse = grid_coord >> pool_shift            # (B, N, 3)
    code = morton_encode(gc_coarse)                 # (B, N) int64

    sentinel = torch.iinfo(code.dtype).max
    code_masked = torch.where(valid, code, torch.full_like(code, sentinel))

    pooled_xyz: List[torch.Tensor] = []
    pooled_gc: List[torch.Tensor] = []
    pooled_feat: List[torch.Tensor] = []
    inverse_list: List[torch.Tensor] = []
    valid_counts: List[int] = []
    max_M = 0
    for b in range(B):
        code_b = code_masked[b]
        unique_b, inverse_b = torch.unique(code_b, return_inverse=True)
        # torch.unique returns sorted output; invalid-sentinel lands last.
        has_invalid = (unique_b.numel() > 0 and
                       unique_b[-1].item() == sentinel)
        M = unique_b.numel() - (1 if has_invalid else 0)
        M = max(M, 1)  # keep at least one slot to avoid zero-size tensors

        # Scatter-mean for xyz / grid_coord / feat across clusters.
        counts = torch.zeros(unique_b.numel(),
                             device=feat.device, dtype=feat.dtype)
        ones = torch.ones(N, device=feat.device, dtype=feat.dtype)
        counts.index_add_(0, inverse_b, ones)
        counts_safe = counts.clamp_min(1).unsqueeze(-1)

        sum_xyz = torch.zeros(unique_b.numel(), 3,
                              device=xyz.device, dtype=xyz.dtype)
        sum_xyz.index_add_(0, inverse_b, xyz[b])
        mean_xyz = sum_xyz / counts_safe.to(sum_xyz.dtype)

        # Grid coord within a cluster is by construction identical; pick the
        # first representative by scatter (last-write wins is fine).
        rep_gc = torch.zeros(unique_b.numel(), 3,
                             device=grid_coord.device, dtype=grid_coord.dtype)
        rep_gc[inverse_b] = gc_coarse[b]

        sum_feat = torch.zeros(unique_b.numel(), C,
                               device=feat.device, dtype=feat.dtype)
        sum_feat.index_add_(0, inverse_b, feat[b])
        mean_feat = sum_feat / counts_safe

        pooled_xyz.append(mean_xyz[:M])
        pooled_gc.append(rep_gc[:M])
        pooled_feat.append(mean_feat[:M])
        # Clamp inverse of invalid points so it points into the kept range.
        inv_b = inverse_b.clone()
        inv_b = inv_b.clamp_max(M - 1)
        inverse_list.append(inv_b)
        valid_counts.append(M)
        max_M = max(max_M, M)

    device = feat.device
    new_xyz = torch.zeros(B, max_M, 3, device=device, dtype=xyz.dtype)
    new_gc = torch.zeros(B, max_M, 3, device=device, dtype=grid_coord.dtype)
    new_feat = torch.zeros(B, max_M, C, device=device, dtype=feat.dtype)
    new_valid = torch.zeros(B, max_M, device=device, dtype=torch.bool)
    inverse = torch.stack(inverse_list, dim=0)  # (B, N)
    for b in range(B):
        M = valid_counts[b]
        new_xyz[b, :M] = pooled_xyz[b]
        new_gc[b, :M] = pooled_gc[b]
        new_feat[b, :M] = pooled_feat[b]
        new_valid[b, :M] = True

    return new_xyz, new_gc, new_feat, new_valid, inverse, max_M


class VoxelPoolDown(nn.Module):
    """Voxel-grid pooling via Morton-cluster scatter, with BN+GELU stem.

    Replaces the old sequence-strided 1-D conv transition. Points that share
    the coarsened voxel (``grid_coord >> pool_shift``) are averaged; the
    resulting (possibly jagged) per-scene lists are padded to a dense
    (B, M, C) tensor and tracked with a boolean ``valid`` mask.
    """

    def __init__(self, in_c: int, out_c: int, pool_stride: int = 2):
        super().__init__()
        assert pool_stride >= 2, "pool_stride >= 2 required"
        self.pool_shift = (pool_stride - 1).bit_length()  # stride=2 -> 1, 4 -> 2
        self.proj = nn.Linear(in_c, out_c)
        self.bn = nn.BatchNorm1d(out_c, eps=1e-3, momentum=0.01)
        self.act = nn.GELU()

    def forward(self, xyz, grid_coord, feat, valid):
        feat = self.proj(feat)
        new_xyz, new_gc, new_feat, new_valid, inverse, M = _voxel_pool(
            xyz, grid_coord, feat, valid, self.pool_shift
        )
        # BN over valid tokens only to avoid contaminating running stats with
        # zero-padded rows.
        flat = new_feat.reshape(-1, new_feat.shape[-1])
        vmask = new_valid.reshape(-1)
        if vmask.any():
            v_flat = flat[vmask]
            v_flat = self.bn(v_flat)
            flat = flat.clone()
            flat[vmask] = v_flat
            flat = self.act(flat)
        new_feat = flat.reshape(new_feat.shape)
        return new_xyz, new_gc, new_feat, new_valid, inverse


class VoxelUnpoolUp(nn.Module):
    """Scatter pooled features back to the fine level and add the skip.

    Uses the ``inverse`` map saved by the paired :class:`VoxelPoolDown` to
    gather each fine point's cluster representative.
    """

    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        self.proj_up = nn.Linear(in_c, out_c)
        self.proj_skip = nn.Linear(skip_c, out_c)
        self.bn = nn.BatchNorm1d(out_c, eps=1e-3, momentum=0.01)
        self.act = nn.GELU()

    def forward(
        self,
        coarse_feat: torch.Tensor,   # (B, M, in_c)
        inverse: torch.Tensor,       # (B, N_fine) long
        skip_feat: torch.Tensor,     # (B, N_fine, skip_c)
        skip_valid: torch.Tensor,    # (B, N_fine) bool -- fine level mask
    ) -> torch.Tensor:
        coarse_proj = self.proj_up(coarse_feat)           # (B, M, out_c)
        idx = inverse.unsqueeze(-1).expand(-1, -1, coarse_proj.shape[-1])
        gathered = torch.gather(coarse_proj, 1, idx)      # (B, N_fine, out_c)
        skip_proj = self.proj_skip(skip_feat)
        out = gathered + skip_proj

        flat = out.reshape(-1, out.shape[-1])
        vmask = skip_valid.reshape(-1)
        if vmask.any():
            v_flat = flat[vmask]
            v_flat = self.bn(v_flat)
            flat = flat.clone()
            flat[vmask] = v_flat
            flat = self.act(flat)
        return flat.reshape(out.shape)


# ──────────────────────── full backbone ──────────────────────────────────────

class PTv3Wrapper(nn.Module):
    """Point Transformer V3 backbone with multi-pattern serialization.

    Changes vs. the initial version:
    - Voxel-grid pool/unpool (scatter on Morton-code clusters) instead of
      sequence-strided 1-D conv pooling.
    - Configurable xCPE: ``"conv1d" | "knn" | "sparse3d"``.
    - Shuffle Order implemented as a one-time per-forward permutation of the
      ordering list, with each block using ``order_index = i % num_patterns``
      within its stage (matches Pointcept).
    - Attention uses a ``key_padding_mask`` so zero-padded / invalid tokens
      never leak into softmax.
    - Stacked blocks per stage, BN+GELU after stem and transitions,
      reversed decoder drop-path schedule.
    - Accepts optional per-point feature channels (normals / colors) via
      ``in_channels`` + ``feat`` argument to ``forward``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 128,
        grid_size: float = 0.01,
        window_size: int = 256,
        enc_channels: Tuple[int, ...] = (32, 64, 128, 256),
        enc_num_heads: Tuple[int, ...] = (4, 4, 8, 8),
        enc_depths: Tuple[int, ...] = (2, 2, 2, 2),
        dec_depths: Tuple[int, ...] = (2, 2, 2),
        pool_strides: Tuple[int, ...] = (2, 2, 2),
        drop_path_rate: float = 0.3,
        cpe_mode: str = "knn",
        knn_k: int = 16,
    ):
        super().__init__()
        assert len(enc_channels) == len(enc_num_heads) == len(enc_depths)
        assert len(dec_depths) == len(enc_channels) - 1 == len(pool_strides)

        self.in_channels = in_channels
        self.grid_size = grid_size
        self.window_size = window_size
        self.cpe_mode = cpe_mode
        self.num_patterns = NUM_PATTERNS
        self.enc_depths = tuple(enc_depths)
        self.dec_depths = tuple(dec_depths)
        self.pool_strides = tuple(pool_strides)

        # Stem: Linear -> BN -> GELU  (Tier 3.2)
        self.in_proj = nn.Linear(in_channels, enc_channels[0])
        self.in_bn = nn.BatchNorm1d(enc_channels[0], eps=1e-3, momentum=0.01)
        self.in_act = nn.GELU()

        # Drop-path schedules (Tier 3.3)
        enc_dp = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                   max(sum(enc_depths), 1))]
        dec_dp = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                   max(sum(dec_depths), 1))]

        # Encoder (Tier 3.1: depth per stage)
        self.enc_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        dp_cursor = 0
        for s, (C, H, D) in enumerate(zip(enc_channels, enc_num_heads, enc_depths)):
            stage_blocks = nn.ModuleList()
            for i in range(D):
                stage_blocks.append(GridWindowAttention(
                    C, num_heads=H, window_size=window_size,
                    drop_path=enc_dp[dp_cursor + i],
                    cpe_mode=cpe_mode, knn_k=knn_k,
                    indice_key=f"stage{s}",
                ))
            dp_cursor += D
            self.enc_blocks.append(stage_blocks)
            if s < len(enc_channels) - 1:
                self.down_blocks.append(VoxelPoolDown(
                    in_c=C, out_c=enc_channels[s + 1],
                    pool_stride=pool_strides[s],
                ))

        # Decoder: depths mirror encoder stages 0..num_stages-2.
        # Reference reverses the per-stage drop-path slice.
        self.dec_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for s in range(len(enc_channels) - 1):
            stage_dp = dec_dp[sum(dec_depths[:s]):sum(dec_depths[:s + 1])]
            stage_dp = list(reversed(stage_dp))
            C = enc_channels[s]
            H = enc_num_heads[s]
            self.up_blocks.append(VoxelUnpoolUp(
                in_c=enc_channels[s + 1], skip_c=C, out_c=C,
            ))
            stage_blocks = nn.ModuleList()
            for i in range(dec_depths[s]):
                stage_blocks.append(GridWindowAttention(
                    C, num_heads=H, window_size=window_size,
                    drop_path=stage_dp[i],
                    cpe_mode=cpe_mode, knn_k=knn_k,
                    indice_key=f"stage{s}",
                ))
            self.dec_blocks.append(stage_blocks)

        self.out_proj = nn.Linear(enc_channels[0], out_channels)

    # --------------------------------------------------------------------

    def _reorder(self, feat: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Gather ``feat`` (B, N, C) along dim=1 using (B, N) index tensor."""
        B, N = idx.shape
        C = feat.shape[-1]
        return torch.gather(feat, 1, idx.unsqueeze(-1).expand(B, N, C))

    def _reorder_mask(self, mask: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return torch.gather(mask, 1, idx)

    def _reorder_gc(self, gc: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        B, N = idx.shape
        return torch.gather(gc, 1, idx.unsqueeze(-1).expand(B, N, 3))

    def _orderings_for_level(
        self,
        cache: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]],
        level: int,
        grid_coord: torch.Tensor,
        valid: torch.Tensor,
        perm: List[int],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Return the 4 orderings for this level (cached), permuted by ``perm``."""
        if level not in cache:
            ords = compute_orderings(grid_coord, valid)
            cache[level] = ords
        ords = cache[level]
        return [ords[p] for p in perm]

    def _run_stage(
        self,
        blocks: nn.ModuleList,
        feat: torch.Tensor,
        valid: torch.Tensor,
        grid_coord: torch.Tensor,
        orderings: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Run every block in a stage, each with its own (possibly permuted) ordering."""
        for i, block in enumerate(blocks):
            s_idx, u_idx = orderings[i % self.num_patterns]
            s_feat = self._reorder(feat, s_idx)
            s_valid = self._reorder_mask(valid, s_idx)
            s_gc = self._reorder_gc(grid_coord, s_idx)
            s_feat = block(s_feat, s_valid, s_gc)
            feat = self._reorder(s_feat, u_idx)
        return feat

    # --------------------------------------------------------------------

    def forward(
        self,
        xyz: torch.Tensor,
        feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """xyz: (B, N, 3). Optional ``feat``: (B, N, in_channels - 3).

        Returns (B, out_channels, N) per-point features.
        """
        B, N, _ = xyz.shape

        # Assemble input features.
        if feat is None:
            if self.in_channels == 3:
                x0 = xyz
            else:
                raise ValueError(
                    f"PTv3Wrapper configured with in_channels={self.in_channels} "
                    "but no `feat` tensor was provided."
                )
        else:
            if feat.shape[-1] + 3 != self.in_channels:
                raise ValueError(
                    f"Expected feat with {self.in_channels - 3} extra channels, "
                    f"got {feat.shape[-1]}."
                )
            x0 = torch.cat([xyz, feat], dim=-1)

        # Stem: Linear -> BN -> GELU
        h = self.in_proj(x0)
        h_flat = h.reshape(-1, h.shape[-1])
        h_flat = self.in_bn(h_flat)
        h_flat = self.in_act(h_flat)
        h = h_flat.reshape(B, N, -1)

        grid_coord = _quantize(xyz, self.grid_size)          # (B, N, 3) int64
        valid = torch.ones(B, N, dtype=torch.bool, device=xyz.device)

        # Shuffle Order: one permutation of the 4 patterns per forward pass.
        if self.training:
            perm = torch.randperm(self.num_patterns).tolist()
        else:
            perm = list(range(self.num_patterns))

        # Encoder -------------------------------------------------------
        ord_cache: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
        skips: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                          torch.Tensor, torch.Tensor]] = []
        # each skip entry: (xyz, grid_coord, feat, valid, inverse_to_coarse or None)

        cur_xyz, cur_gc, cur_feat, cur_valid = xyz, grid_coord, h, valid
        for s, stage_blocks in enumerate(self.enc_blocks):
            ords = self._orderings_for_level(
                ord_cache, s, cur_gc, cur_valid, perm
            )
            cur_feat = self._run_stage(stage_blocks, cur_feat, cur_valid,
                                       cur_gc, ords)
            if s < len(self.down_blocks):
                skips.append((cur_xyz, cur_gc, cur_feat, cur_valid))
                down = self.down_blocks[s]
                cur_xyz, cur_gc, cur_feat, cur_valid, inverse = down(
                    cur_xyz, cur_gc, cur_feat, cur_valid
                )
                # Store inverse as last element of previous skip entry.
                skips[-1] = skips[-1] + (inverse,)

        # Decoder -------------------------------------------------------
        # Iterate from deepest encoder stage (which was not pushed to skips)
        # back up through each skip (paired with its stored inverse).
        for s in reversed(range(len(self.up_blocks))):
            skip_xyz, skip_gc, skip_feat, skip_valid, inverse = skips[s]
            up = self.up_blocks[s]
            cur_feat = up(cur_feat, inverse, skip_feat, skip_valid)
            cur_xyz, cur_gc, cur_valid = skip_xyz, skip_gc, skip_valid
            # Decoder stage blocks: use a fresh cache key so orderings are
            # computed for this fine level once.
            ords = self._orderings_for_level(
                ord_cache, s, cur_gc, cur_valid, perm,
            )
            stage_blocks = self.dec_blocks[s]
            cur_feat = self._run_stage(stage_blocks, cur_feat, cur_valid,
                                       cur_gc, ords)

        # Output projection -> (B, C_out, N)
        out = self.out_proj(cur_feat)
        return out.transpose(1, 2)
