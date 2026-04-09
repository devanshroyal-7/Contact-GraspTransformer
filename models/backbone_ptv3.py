import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────── space-filling curve helpers ───────────────────────

def _part1by2(n: torch.Tensor) -> torch.Tensor:
    """Spread the low 10 bits of *n* by inserting two 0-bits between each."""
    n = n & 0x000003FF
    n = (n ^ (n << 16)) & 0xFF0000FF
    n = (n ^ (n <<  8)) & 0x0F00F00F
    n = (n ^ (n <<  4)) & 0xC30C30C3
    n = (n ^ (n <<  2)) & 0x49249249
    return n


def morton_encode(xyz_int: torch.Tensor) -> torch.Tensor:
    """Z-order / Morton code from integer grid coords (…, 3) → (…,)."""
    x, y, z = xyz_int[..., 0], xyz_int[..., 1], xyz_int[..., 2]
    return _part1by2(x) | (_part1by2(y) << 1) | (_part1by2(z) << 2)


def serialize_point_cloud(xyz: torch.Tensor, grid_size: float = 0.01
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    """Serialize a (B, N, 3) cloud via Morton codes.

    Returns (sort_idx, unsort_idx) each (B, N).
    Finer *grid_size* produces more unique keys and better locality.
    """
    quantized = torch.floor(xyz / grid_size).long()
    quantized = quantized - quantized.min(dim=1, keepdim=True).values  # shift to positive

    keys = morton_encode(quantized)
    sort_idx = keys.argsort(dim=-1, stable=True)
    unsort_idx = sort_idx.argsort(dim=-1)
    return sort_idx, unsort_idx


# ──────────────────────────── attention block ────────────────────────────────

class GridWindowAttention(nn.Module):
    """Pre-norm windowed self-attention over a spatially serialized sequence."""

    def __init__(self, channels: int, num_heads: int = 4, window_size: int = 48):
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(channels)
        self.mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )
        self.pos_enc = nn.Sequential(nn.Linear(3, channels), nn.LayerNorm(channels))

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        B, N, C = feat.shape
        W = self.window_size

        pad_len = (W - N % W) % W
        if pad_len:
            feat = F.pad(feat, (0, 0, 0, pad_len))
            xyz  = F.pad(xyz,  (0, 0, 0, pad_len))

        Np = feat.shape[1]
        nw = Np // W
        wf = feat.reshape(B * nw, W, C)
        wx = xyz.reshape(B * nw, W, 3)

        # relative positional encoding within each window
        wf = wf + self.pos_enc(wx - wx[:, 0:1, :])

        # pre-norm attention
        res = wf
        wf = self.norm1(wf)
        wf = res + self.mha(wf, wf, wf)[0]

        # pre-norm FFN
        res = wf
        wf = self.norm2(wf)
        wf = res + self.mlp(wf)

        return wf.reshape(B, Np, C)[:, :N, :]


# ─────────────────────────── transition layers ───────────────────────────────

class TransitionDown(nn.Module):
    """Grid pooling along the serialized sequence via strided 1-D conv."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 4):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=stride, stride=stride)

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor):
        B, N, _ = xyz.shape
        s = self.stride

        # Trim to an exact multiple of stride so pooling is lossless
        trim = (N // s) * s
        xyz_t  = xyz[:, :trim, :]
        feat_t = feat[:, :trim, :]

        xyz_pooled = xyz_t.reshape(B, trim // s, s, 3).mean(dim=2)
        feat_down  = self.conv(feat_t.transpose(1, 2)).transpose(1, 2)
        return xyz_pooled, feat_down


class TransitionUp(nn.Module):
    """Sequence upsample + skip connection via addition."""

    def __init__(self, in_channels: int, out_channels: int, scale: int = 4):
        super().__init__()
        self.scale = scale
        self.proj_up   = nn.Linear(in_channels, out_channels)
        self.proj_skip = nn.Linear(out_channels, out_channels)

    def forward(self, feat_skip: torch.Tensor, feat_coarse: torch.Tensor):
        up = F.interpolate(feat_coarse.transpose(1, 2),
                           size=feat_skip.shape[1], mode="nearest").transpose(1, 2)
        return self.proj_up(up) + self.proj_skip(feat_skip)


# ──────────────────────────── full backbone ──────────────────────────────────

class PTv3Wrapper(nn.Module):
    """Point Transformer V3 style backbone.

    Serializes via Morton codes once, runs a windowed-attention U-Net,
    then unserializes back to the original point ordering.
    """

    def __init__(self, out_channels: int = 128, grid_size: float = 0.01):
        super().__init__()
        self.grid_size = grid_size
        self.in_proj = nn.Linear(3, 32)

        self.enc1  = GridWindowAttention(32)
        self.down1 = TransitionDown(32, 64, stride=4)

        self.enc2  = GridWindowAttention(64)
        self.down2 = TransitionDown(64, 128, stride=4)

        self.enc3  = GridWindowAttention(128)
        self.down3 = TransitionDown(128, 256, stride=4)

        self.enc4  = GridWindowAttention(256)

        self.up3  = TransitionUp(256, 128, scale=4)
        self.dec3 = GridWindowAttention(128)

        self.up2  = TransitionUp(128, 64, scale=4)
        self.dec2 = GridWindowAttention(64)

        self.up1  = TransitionUp(64, 32, scale=4)
        self.dec1 = GridWindowAttention(32)

        self.out_proj = nn.Linear(32, out_channels)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """xyz: (B, N, 3) → features (B, C, N)"""
        B, N, _ = xyz.shape

        # ── serialization via Morton codes ──
        sort_idx, unsort_idx = serialize_point_cloud(xyz, self.grid_size)
        batch_idx = torch.arange(B, device=xyz.device).view(B, 1).expand_as(sort_idx)
        s_xyz = xyz[batch_idx, sort_idx]

        s_feat = self.in_proj(s_xyz)

        # ── encoder ──
        e1 = self.enc1(s_xyz, s_feat)
        x2, e2 = self.down1(s_xyz, e1)

        e2 = self.enc2(x2, e2)
        x3, e3 = self.down2(x2, e2)

        e3 = self.enc3(x3, e3)
        x4, e4 = self.down3(x3, e3)

        e4 = self.enc4(x4, e4)

        # ── decoder ──
        d3 = self.up3(e3, e4)
        d3 = self.dec3(x3, d3)

        d2 = self.up2(e2, d3)
        d2 = self.dec2(x2, d2)

        d1 = self.up1(e1, d2)
        d1 = self.dec1(s_xyz, d1)

        s_out = self.out_proj(d1)

        # ── unsort back to original order ──
        out = s_out[batch_idx, unsort_idx]
        return out.transpose(1, 2)
