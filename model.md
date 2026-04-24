# Model Architectures

The training pipeline plugs one of two point-cloud backbones into a shared
`ContactGraspNet` wrapper with per-point CGN prediction heads. The backbone is
selected at train time via `--backbone {pn2, ptv3}`; both consume a dense
`(B, N, 3)` point cloud and return per-point features `(B, 64, N)` that are
fed to the CGN heads (confidence, base/approach directions via Gram-Schmidt,
10-bin width classifier).

Tables below follow the same format as our reference CNN table: per-stage
hidden-layer structure on the left, fixed architectural constants and
tunable hyper-parameters on the right. Code references are provided under
each table so the numbers can be verified directly against the source.

## 1. Shared wrapper and heads

The wrapper simply dispatches to the chosen backbone and then runs the CGN
heads on top of the returned per-point features.

```17:33:models/model.py
    def __init__(self, backbone_type='pn2', backbone_kwargs=None):
        super().__init__()
        self.head_in_channels = 64
        backbone_kwargs = dict(backbone_kwargs or {})

        if backbone_type == 'pn2':
            self.backbone = SimplePointNet2(
                out_channels=self.head_in_channels, **backbone_kwargs
            )
        elif backbone_type == 'ptv3':
            self.backbone = PTv3Wrapper(
                out_channels=self.head_in_channels, **backbone_kwargs
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone_type}")

        self.heads = CGNHeads(in_channels=self.head_in_channels)
```

The CGN heads produce: (a) 1 confidence logit per point, (b) two 3-D vectors
refined by Gram-Schmidt to a base/approach frame, and (c) a 10-bin width
classifier over `[0, 0.08] m`.

```35:73:models/cgn_heads.py
    def __init__(self, in_channels, num_width_bins=NUM_WIDTH_BINS,
                 gripper_width_max=GRIPPER_WIDTH_MAX):
        super().__init__()
        self.num_width_bins = num_width_bins
        self.gripper_width_max = gripper_width_max

        bin_width = gripper_width_max / num_width_bins
        centres = torch.arange(num_width_bins).float() * bin_width + bin_width / 2
        self.register_buffer("bin_centres", centres)

        hidden = max(in_channels, 64)

        self.conv_conf = nn.Sequential(
            nn.Conv1d(in_channels, hidden, 1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, 1, 1),
        )

        self.conv_z1 = nn.Sequential(
            nn.Conv1d(in_channels, hidden, 1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, 3, 1),
        )

        self.conv_z2 = nn.Sequential(
            nn.Conv1d(in_channels, hidden, 1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, 3, 1),
        )

        self.conv_width = nn.Sequential(
            nn.Conv1d(in_channels, hidden, 1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, num_width_bins, 1),
        )
```

---

## 2. PointNet++ (`pn2`) backbone

A 4-level U-Net built from PointNet `SetAbstraction` (SA) encoder blocks and
`FeaturePropagation` (FP) decoder blocks. Each SA block does FPS sampling,
ball-query grouping, and a shared-MLP per group followed by max-pooling.
Each FP block does 3-NN inverse-distance-weighted interpolation from the
coarse level, concatenates the fine-level skip, and applies a 1D MLP.

### Per-stage architecture

| Stage                  | Hidden layers                                                  | Parameters / hyper-parameters                                           |
| ---------------------- | -------------------------------------------------------------- | ----------------------------------------------------------------------- |
| SA1 (encoder)          | 3 x {Conv2d(1x1), BN, ReLU}, max over group, then MaxPool       | npoint = 512, radius = 0.1, nsample = 32, MLP = [32, 32, 64]            |
| SA2 (encoder)          | 3 x {Conv2d(1x1), BN, ReLU}, max over group, then MaxPool       | npoint = 128, radius = 0.2, nsample = 32, MLP = [64, 64, 128]           |
| SA3 (encoder)          | 3 x {Conv2d(1x1), BN, ReLU}, max over group, then MaxPool       | npoint = 32,  radius = 0.4, nsample = 32, MLP = [128, 128, 256]         |
| SA4 (encoder)          | 3 x {Conv2d(1x1), BN, ReLU}, max over group, then MaxPool       | npoint = 8,   radius = 0.8, nsample = 32, MLP = [256, 256, 512]         |
| FP4 (decoder)          | 2 x {Conv1d(1x1), BN, ReLU} after 3-NN interp + skip concat    | in = 512 + 256, MLP = [256, 256]                                        |
| FP3 (decoder)          | 2 x {Conv1d(1x1), BN, ReLU} after 3-NN interp + skip concat    | in = 256 + 128, MLP = [256, 128]                                        |
| FP2 (decoder)          | 2 x {Conv1d(1x1), BN, ReLU} after 3-NN interp + skip concat    | in = 128 + 64,  MLP = [128, 64]                                         |
| FP1 (decoder, to N)    | 3 x {Conv1d(1x1), BN, ReLU} after 3-NN interp to N points      | in = 64, MLP = [64, 64, `out_channels` = 64]                            |
| Heads (shared)         | 4 x {Conv1d -> BN -> ReLU -> Conv1d}                           | conf (1 ch), z1 (3 ch), z2 (3 ch), width (10 bins), hidden = 64         |

The `+3` in the SA `in_channel` arguments comes from concatenating the
relative `grouped_xyz_norm` to the grouped per-point features.

### Code references (verify the architecture)

```146:180:models/backbone_pn2.py
class SimplePointNet2(nn.Module):
    """
    A PointNet++ U-Net Architecture Backbone with Set Abstraction and Feature Propagation.
    Designed to return per-point features mapping back to the original N points.
    """
    def __init__(self, out_channels=64):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(512, 0.1, 32, 3 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(128, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(32, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(8, 0.8, 32, 256 + 3, [256, 256, 512], False)
        
        self.fp4 = PointNetFeaturePropagation(512 + 256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(256 + 128, [256, 128])
        self.fp2 = PointNetFeaturePropagation(128 + 64,  [128, 64])
        self.fp1 = PointNetFeaturePropagation(64,        [64, 64, out_channels])
```

SA block internals (shared-MLP over grouped neighbours, max-pool over the group):

```70:102:models/backbone_pn2.py
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = xyz, points
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points
```

FP block internals (3-NN inverse-distance interp + skip concat + 1D MLP):

```104:144:models/backbone_pn2.py
class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, 1, N)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            
            interpolated_points = torch.sum(index_points(points2.permute(0, 2, 1), idx) * weight.view(B, N, 3, 1), dim=2)
            interpolated_points = interpolated_points.permute(0, 2, 1)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=1)
        else:
            new_points = interpolated_points
```

---

## 3. Point Transformer V3 (`ptv3`) backbone

A 4-stage encoder / 3-stage decoder U-Net that replaces PointNet++ FPS+ball
grouping with **space-filling-curve serialization + windowed attention**.
Points are voxel-quantized, ordered by 4 different curves (Z, transposed Z,
3-D Hilbert, transposed Hilbert), split into fixed-length windows, and
passed through pre-norm multi-head self-attention with an xCPE residual.
Down/up transitions are scatter-based voxel pool / unpool keyed on Morton
codes, with a `valid` padding mask threaded through every block.

### Per-stage architecture (defaults)

| Stage                   | Hidden layers                                                                  | Parameters / hyper-parameters                                                    |
| ----------------------- | ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| Stem                    | Linear -> BN -> GELU                                                           | in = `in_channels` (3 by default), out = 32                                      |
| Enc stage 0             | 2 x {xCPE -> LN -> MHA -> LN -> MLP(4x)}                                       | C = 32,  H = 4 heads, window = 256, drop-path in [0, 0.3]                        |
| Down 0 (pool)           | Linear -> Morton-cluster scatter-mean -> BN -> GELU                            | 32 -> 64,  pool_stride = 2 (`pool_shift` = 1)                                    |
| Enc stage 1             | 2 x {xCPE -> LN -> MHA -> LN -> MLP(4x)}                                       | C = 64,  H = 4 heads, window = 256                                               |
| Down 1 (pool)           | Linear -> Morton-cluster scatter-mean -> BN -> GELU                            | 64 -> 128, pool_stride = 2                                                       |
| Enc stage 2             | 2 x {xCPE -> LN -> MHA -> LN -> MLP(4x)}                                       | C = 128, H = 8 heads, window = 256                                               |
| Down 2 (pool)           | Linear -> Morton-cluster scatter-mean -> BN -> GELU                            | 128 -> 256, pool_stride = 2                                                      |
| Enc stage 3 (bottleneck)| 2 x {xCPE -> LN -> MHA -> LN -> MLP(4x)}                                       | C = 256, H = 8 heads, window = 256                                               |
| Up 2 (unpool)           | Linear(coarse) + Linear(skip) -> gather by inverse -> BN -> GELU               | 256 -> 128, skip = 128                                                           |
| Dec stage 2             | 2 x {xCPE -> LN -> MHA -> LN -> MLP(4x)}                                       | C = 128, H = 8 heads, window = 256, drop-path reversed                           |
| Up 1 (unpool)           | Linear(coarse) + Linear(skip) -> gather by inverse -> BN -> GELU               | 128 -> 64,  skip = 64                                                            |
| Dec stage 1             | 2 x {xCPE -> LN -> MHA -> LN -> MLP(4x)}                                       | C = 64,  H = 4 heads, window = 256                                               |
| Up 0 (unpool)           | Linear(coarse) + Linear(skip) -> gather by inverse -> BN -> GELU               | 64  -> 32,  skip = 32                                                            |
| Dec stage 0             | 2 x {xCPE -> LN -> MHA -> LN -> MLP(4x)}                                       | C = 32,  H = 4 heads, window = 256                                               |
| Output projection       | Linear                                                                         | 32 -> `out_channels` = 64                                                        |
| Heads (shared)          | 4 x {Conv1d -> BN -> ReLU -> Conv1d}                                           | conf (1 ch), z1 (3 ch), z2 (3 ch), width (10 bins), hidden = 64                  |

Tunable knobs exposed on the `PTv3Wrapper` constructor:

| Knob                | Default                    | Role                                                                   |
| ------------------- | -------------------------- | ---------------------------------------------------------------------- |
| `in_channels`       | 3                          | xyz only (3) or xyz + extra per-point features (6 = xyz+normals, ...)  |
| `out_channels`      | 128 (64 when used here)    | per-point embedding width handed to the heads                          |
| `grid_size`         | 0.01 m                     | voxel side for quantization / Morton+Hilbert keys                      |
| `window_size`       | 256                        | tokens per local attention window                                      |
| `enc_channels`      | (32, 64, 128, 256)         | per-stage channel widths                                               |
| `enc_num_heads`     | (4, 4, 8, 8)               | heads per stage                                                        |
| `enc_depths`        | (2, 2, 2, 2)               | attention blocks per encoder stage                                     |
| `dec_depths`        | (2, 2, 2)                  | attention blocks per decoder stage (mirrors enc stages 0..N-2)         |
| `pool_strides`      | (2, 2, 2)                  | voxel pool stride between stages (shift = `log2(stride)`)              |
| `drop_path_rate`    | 0.3                        | max stochastic-depth rate (linear schedule, reversed in decoder)       |
| `cpe_mode`          | "knn"                      | xCPE variant: `"conv1d"` / `"knn"` / `"sparse3d"`                      |
| `knn_k`             | 16                         | k for `KNNCPE` (ignored for other CPE modes)                           |

### Code references (verify the architecture)

Backbone constructor and shape arithmetic:

```599:682:models/backbone_ptv3.py
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
```

A single attention block: xCPE residual -> pre-norm MHA -> pre-norm MLP(4x)
with drop-path and a `key_padding_mask` that prevents padded tokens from
participating in softmax:

```346:411:models/backbone_ptv3.py
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
```

Voxel pool / unpool transitions (Morton-cluster scatter-mean with BN+GELU):

```505:538:models/backbone_ptv3.py
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
```

```541:577:models/backbone_ptv3.py
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
```

xCPE (conditional positional encoding) variants — pick one via `cpe_mode`:

```187:205:models/backbone_ptv3.py
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
```

```208:252:models/backbone_ptv3.py
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
```

```255:280:models/backbone_ptv3.py
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
```

Shuffle-order serialization (each forward picks a random permutation over the
four curves, each block in a stage then uses `order_index = i % 4`):

```771:798:models/backbone_ptv3.py
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
```

---

## 4. Training hyper-parameters (shared)

These knobs are swept by `sweep_config.yaml` (Bayesian search on
`val/loss`). The corresponding CLI flags live in `train.py`.

| Hyper-parameter        | Values / range                                      | Source                                               |
| ---------------------- | --------------------------------------------------- | ---------------------------------------------------- |
| `backbone`             | `pn2`, `ptv3`                                       | `train.py --backbone`                                |
| `cpe_mode` (PTv3 only) | `conv1d`, `knn`, `sparse3d`                         | `train.py --cpe_mode`                                |
| `num_points` (N)       | 2048, 4096, 8192                                    | sweep (`num_points`)                                 |
| `batch_size`           | 4, 8, 16                                            | sweep (`batch_size`)                                 |
| `optimizer`            | `adam`, `adamw`                                     | sweep (`optimizer`)                                  |
| `lr`                   | log-uniform `[1e-5, 1e-2]`                          | sweep (`lr`)                                         |
| `weight_decay`         | log-uniform `[1e-6, 1e-2]`                          | sweep (`weight_decay`)                               |
| `scheduler`            | `none`, `cosine`, `step` (+ `reduce_lr_on_plateau`) | sweep (`scheduler`) / CLI                            |
| `scheduler_gamma`      | uniform `[0.1, 0.5]`                                | sweep (`scheduler_gamma`)                            |
| `grad_clip_max_norm`   | uniform `[0.5, 5.0]` (0 disables)                   | sweep (`grad_clip_max_norm`)                         |
| `loss_adds_weight`     | uniform `[1.0, 20.0]`                               | sweep (`loss_adds_weight`)                           |
| `loss_width_weight`    | uniform `[0.1, 5.0]`                                | sweep (`loss_width_weight`)                          |
| `epochs`               | 30 (default)                                        | `train.py --epochs`                                  |

Sweep declaration:

```1:39:sweep_config.yaml
program: train.py
method: bayes
metric:
  name: val/loss
  goal: minimize

parameters:
  lr:
    distribution: log_uniform_values
    min: 1e-5
    max: 1e-2
  optimizer:
    values: [adam, adamw]
  weight_decay:
    distribution: log_uniform_values
    min: 1e-6
    max: 1e-2
  batch_size:
    values: [4, 8, 16]
  scheduler:
    values: [none, cosine, step]
  scheduler_gamma:
    distribution: uniform
    min: 0.1
    max: 0.5
  grad_clip_max_norm:
    distribution: uniform
    min: 0.5
    max: 5.0
  loss_adds_weight:
    distribution: uniform
    min: 1.0
    max: 20.0
  loss_width_weight:
    distribution: uniform
    min: 0.1
    max: 5.0
  num_points:
    values: [2048, 4096, 8192]
```

CLI flag plumbing:

```62:94:train.py
    parser.add_argument("--data_dir", type=str, default="data/out", help="Path to datasets")
    parser.add_argument("--backbone", type=str, default="ptv3", choices=["pn2", "ptv3"], help="Backbone type")
    parser.add_argument(
        "--cpe_mode",
        type=str,
        default="sparse3d",
        choices=["knn", "conv1d", "sparse3d"],
        help="PTv3 xCPE (conditional positional encoding); ignored when backbone is pn2",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"])
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine", "step", "reduce_lr_on_plateau"])
    parser.add_argument("--scheduler_gamma", type=float, default=0.3)
    parser.add_argument("--grad_clip_max_norm", type=float, default=0.0, help="0 disables gradient clipping")
    parser.add_argument("--loss_adds_weight", type=float, default=10.0)
    parser.add_argument("--loss_width_weight", type=float, default=1.0)
    parser.add_argument("--num_points", type=int, default=4096)
```
