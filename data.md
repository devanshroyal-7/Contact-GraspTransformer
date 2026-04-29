# Data Generation Pipeline

## Source data (ACRONYM)

```
data/acronym/
├── manifest.json            180 objects (15 categories x 12: 10 train + 2 test)
├── training_budgets.json    caps on train meshes per category (1/2/5/10)
├── meshes/<Category>/*.obj  raw ShapeNet meshes
└── grasps/*.h5              grasp transforms + success labels
```

The 15 categories are:
`Mug, Bowl, Bottle, Cup, Pan, Book, Vase, CellPhone, Pencil, ToyFigure,
Knife, FoodItem, Camera, SodaCan, WineBottle`.

### Rebuilding the subset from the external ACRONYM checkout

```bash
python data/acronym/build_acronym_subset.py [--dry_run]
```

This script reads `/home/devansh/dev/contact_graspnet_pytorch/acronym/`,
deterministically picks 12 unique meshes per category (smallest scale
per mesh hash, sorted lexicographically), copies them into
`data/acronym/meshes/` and `data/acronym/grasps/`, removes any stale
files not in the new selection, and writes `manifest.json`.

### `manifest.json` schema

One entry per object. Every entry has `split` ("train" or "test") and
a `rank` field. `rank` is 1..10 within train and 1..2 within test.

```json
{
  "category": "Mug",
  "mesh_hash": "2997f21fa426e18a6ab1a25d0e8f3590",
  "mesh_path": "meshes/Mug/2997f21fa426e18a6ab1a25d0e8f3590.obj",
  "grasp_file": "Mug_2997f21fa426e18a6ab1a25d0e8f3590_0.021360488699532477.h5",
  "scale": 0.0213604887,
  "split": "train",
  "rank": 1
}
```

### `training_budgets.json` schema

Controls how many **training** meshes per category are actually used
at train time. The test split always uses both test meshes per category.

```json
{
  "active_preset": "5_per_cat",
  "presets": {
    "1_per_cat":  {"train_objects_per_category": 1},
    "2_per_cat":  {"train_objects_per_category": 2},
    "5_per_cat":  {"train_objects_per_category": 5},
    "10_per_cat": {"train_objects_per_category": 10}
  },
  "categories": ["Mug", "Bowl", ...]
}
```

At train time the active cap is resolved in this order:

1. `--train_objects_per_category` CLI flag (integer override).
2. `--budget_preset` CLI flag (named preset).
3. `active_preset` in the JSON.

The active cap is logged to W&B as `active_train_objects_per_category`.

## Pipeline overview (`data/generate_data.py`)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PER OBJECT (from manifest)                       │
│                                                                         │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────────┐   │
│  │  Load .obj   │───>│  Apply scale     │───>│  Centre at origin    │   │
│  │  (ShapeNet)  │    │  mesh *= scale   │    │  verts -= mean(verts)│   │
│  └──────────────┘    └──────────────────┘    └─────────┬────────────┘   │
│                                                        │                │
│  ┌──────────────┐    ┌──────────────────┐              │                │
│  │  Load .h5    │───>│  Shift grasps    │<─────────────┘                │
│  │  (ACRONYM)   │    │  trans -= mean   │  (same mean as mesh)          │
│  └──────────────┘    │  (already in m)  │                               │
│                      └────────┬─────────┘                               │
│                               │                                         │
│  ┌────────────────────────────┼─────────────────────────────────────┐   │
│  │                     BUILD PYRENDER SCENE                         │   │
│  │                                                                  │   │
│  │  ┌───────────────────┐  ┌────────────────────────────────────┐   │   │
│  │  │  Table box        │  │  Object placed on table            │   │   │
│  │  │  (1.0 x 1.2 x 0.6 │  │  obj_z = table_z/2 - mesh.min_z    │   │   │
│  │  │   at origin)      │  │  (bottom of mesh flush with table) │   │   │
│  │  └───────────────────┘  └────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     PER VIEW (36 camera poses)                   │   │
│  │                                                                  │   │
│  │   ┌────────────────┐                                             │   │
│  │   │ Sample camera  │  view-sphere: random azimuth, elevation     │   │
│  │   │ pose (OpenGL)  │  distance ∈ [0.55, 0.85]m                   │   │
│  │   └───────┬────────┘                                             │   │
│  │           │                                                      │   │
│  │           v                                                      │   │
│  │   ┌────────────────┐     ┌───────────────────────┐               │   │
│  │   │  Render depth  │────>│  Back-project to PC   │               │   │
│  │   │  (pyrender)    │     │  using RealSense      │               │   │
│  │   │  640×480       │     │  intrinsics (fx,fy,..)│               │   │
│  │   └────────────────┘     └──────────┬────────────┘               │   │
│  │                                     │                            │   │
│  │                                     v                            │   │
│  │                          ┌──────────────────────┐                │   │
│  │                          │  Regularize to N pts │                │   │
│  │                          │  (sub/oversample)    │                │   │
│  │                          └──────────┬───────────┘                │   │
│  │                                     │                            │   │
│  │     ┌───────────────────────────────┼─────────────────────┐      │   │
│  │     │            COORDINATE ALIGNMENT                     │      │   │
│  │     │                               │                     │      │   │
│  │     │   ┌───────────────────┐       v                     │      │   │
│  │     │   │  Compute w2c      │  ┌────────────────────┐     │      │   │
│  │     │   │  (OpenGL→OpenCV   │  │  Transform PC to   │     │      │   │
│  │     │   │   + invert)       │  │  world frame; mark │     │      │   │
│  │     │   └────────┬──────────┘  │  z > table as      │     │      │   │
│  │     │            │             │  object_mask       │     │      │   │
│  │     │            │             └────────┬───────────┘     │      │   │
│  │     │            │                      │                 │      │   │
│  │     │            v                      v                 │      │   │
│  │     │   ┌────────────────────────────────────────────┐    │      │   │
│  │     │   │  Mean-centre PC:  pc -= mean(pc)           │    │      │   │
│  │     │   └────────────────────────┬───────────────────┘    │      │   │
│  │     │                            │                        │      │   │
│  │     └────────────────────────────┼────────────────────────┘      │   │
│  │                                  │                               │   │
│  │     ┌────────────────────────────┼───────────────────────────┐   │   │
│  │     │         GRASP LABELLING    │                           │   │   │
│  │     │                            v                           │   │   │
│  │     │   ┌──────────────────────────────────────────────┐     │   │   │
│  │     │   │  Grasps → camera frame:                      │     │   │   │
│  │     │   │    G_cam = w2c @ obj_pose @ G_local          │     │   │   │
│  │     │   │    G_cam.trans -= pc_mean                    │     │   │   │
│  │     │   └───────────────────────┬──────────────────────┘     │   │   │
│  │     │                           │                            │   │   │
│  │     │                           v                            │   │   │
│  │     │   ┌──────────────────────────────────────────────┐     │   │   │
│  │     │   │  Project TCP → nearest mesh vertex           │     │   │   │
│  │     │   │  (mesh verts also in centred camera frame)   │     │   │   │
│  │     │   │  This moves the label centre from the wrist  │     │   │   │
│  │     │   │  onto the actual object surface              │     │   │   │
│  │     │   └───────────────────────┬──────────────────────┘     │   │   │
│  │     │                           │                            │   │   │
│  │     │                           v                            │   │   │
│  │     │   ┌──────────────────────────────────────────────┐     │   │   │
│  │     │   │  KDTree (object pts only):                   │     │   │   │
│  │     │   │    radius query r=0.02m around each          │     │   │   │
│  │     │   │    surface centre → assign per-point labels  │     │   │   │
│  │     │   │    (confidence, approach_dir, base_dir, w)   │     │   │   │
│  │     │   └──────────────────────────────────────────────┘     │   │   │
│  │     │                                                        │   │   │
│  │     └────────────────────────────────────────────────────────┘   │   │
│  │                                                                  │   │
│  │                           ┌──────────────┐                       │   │
│  │                           │  Save .npz   │                       │   │
│  │                           └──────────────┘                       │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Output structure

```
data/out/
├── train/
│   ├── Mug/
│   │   ├── <mesh_hash_1>/
│   │   │   ├── 000.npz
│   │   │   ├── 001.npz
│   │   │   └── ...   (360 views)
│   │   ├── <mesh_hash_2>/
│   │   └── ...               (up to 10 meshes)
│   ├── Bowl/
│   └── ...                    (15 categories)
└── test/
    ├── Mug/
    │   ├── <mesh_hash_train11>/
    │   └── <mesh_hash_train12>/    (2 held-out meshes per category)
    ├── Bowl/
    └── ...
```

Rendering CLI:

```bash
# Full render (180 meshes x 360 views - substantial compute + disk)
python data/generate_data.py

# Only train meshes
python data/generate_data.py --splits train

# Only one category
python data/generate_data.py --category Mug

# Single mesh (debug)
python data/generate_data.py --mesh_hash 2997f21fa426e18a6ab1a25d0e8f3590
```

## .npz keys per file

| Key             | Shape    | Dtype   | Description                                 |
|-----------------|----------|---------|---------------------------------------------|
| `depth`         | (H, W)  | float32 | Depth image in metres                       |
| `points`        | (N, 3)  | float32 | Mean-centred point cloud (camera frame)     |
| `confidence`    | (N,)    | float32 | Per-point grasp score (0 or 1)              |
| `approach_dirs` | (N, 3)  | float32 | Gripper approach direction (z-axis of pose) |
| `base_dirs`     | (N, 3)  | float32 | Gripper closing direction (x-axis of pose)  |
| `widths`        | (N,)    | float32 | Grasp width (0.08m for Panda)               |
| `camera_pose`   | (4, 4)  | float64 | Camera extrinsic (OpenCV, mean-centred)     |

## Voxel visualization tools

Both voxel visualization scripts open interactive Open3D windows. If GLFW
fails on Linux Wayland, use the XWayland wrapper shown in `SETUP.md`.

### Synthetic / explanatory PTv3 voxel views (`voxel_viz.py`)

`voxel_viz.py` visualizes a generated `.npz` point cloud without loading a
trained checkpoint. It is useful for understanding the voxel grid, pooling
stages, sparse CPE behavior, and serialization order used by the PTv3-style
backbone helpers.

```bash
# Default: pooling view for the example Mug sample
python3 voxel_viz.py

# Pooling view for a specific generated sample
python3 voxel_viz.py data/out/train/Mug/2997f21fa426e18a6ab1a25d0e8f3590/000.npz

# Show all available visualization modes one after another
python3 voxel_viz.py <sample.npz> --mode all
```

Main modes:

| Mode        | What it shows |
|-------------|---------------|
| `pooling`   | Input voxelization, then repeated bit-shift voxel pooling stages. |
| `sparse`    | SparseCPE feature changes on the fixed occupied voxel grid. |
| `z`         | Morton/Z-order serialization path through occupied voxels. |
| `tz`        | Transposed Morton serialization using rotated coordinate order. |
| `hilbert`   | 3-D Hilbert serialization path through occupied voxels. |
| `thilbert` | Transposed Hilbert serialization. |
| `all`       | Runs pooling, sparse, and every serialization view. |

Useful options:

```bash
# Change base voxel size in metres
python3 voxel_viz.py <sample.npz> --grid-size 0.01

# Change how many pooling windows are shown
python3 voxel_viz.py <sample.npz> --mode pooling --stages 3

# Color sparse view by different feature statistics
python3 voxel_viz.py <sample.npz> --mode sparse --sparse-color delta_norm
python3 voxel_viz.py <sample.npz> --mode sparse --sparse-color after_norm
python3 voxel_viz.py <sample.npz> --mode sparse --sparse-color channel --feature-channel 0
python3 voxel_viz.py <sample.npz> --mode sparse --sparse-color signed_delta --feature-channel 0

# Force device for sparse mode; sparse3d requires spconv
python3 voxel_viz.py <sample.npz> --mode sparse --device cpu
python3 voxel_viz.py <sample.npz> --mode sparse --device cuda

# Reduce dense serialization path lines
python3 voxel_viz.py <sample.npz> --mode hilbert --curve-line-step 4

# Set a minimum Hilbert precision; 0 auto-selects from voxel span
python3 voxel_viz.py <sample.npz> --mode hilbert --hilbert-bits 10
```

### Real inference voxel views (`inference_voxel_viz.py`)

`inference_voxel_viz.py` loads a trained checkpoint, runs the same point-cloud
preprocessing and model forward pass used by `inference.py`, and records voxel
locations from each actual PTv3 `VoxelPoolDown` layer. Use this when you want
to see the voxel size and location changes that happen during model inference.

```bash
python3 inference_voxel_viz.py \
  --ckpt checkpoints/best.pt \
  --points data/out/train/Mug/2997f21fa426e18a6ab1a25d0e8f3590/000.npz
```

The script displays the input voxelization first, then each encoder pooling
stage from the real forward pass. It also prints the voxel size and occupied
voxel count for each window.

Useful options:

```bash
# Use a different checkpoint or input cloud
python3 inference_voxel_viz.py --ckpt checkpoints/last.pt --points <sample.npz>

# Force CPU/GPU selection
python3 inference_voxel_viz.py --ckpt checkpoints/best.pt --points <sample.npz> --device cpu
python3 inference_voxel_viz.py --ckpt checkpoints/best.pt --points <sample.npz> --device cuda

# Override checkpoint fallback settings if config is missing
python3 inference_voxel_viz.py --ckpt <ckpt.pt> --points <sample.npz> --cpe-mode knn
python3 inference_voxel_viz.py --ckpt <ckpt.pt> --points <sample.npz> --num-points 4096

# Color voxels and optionally overlay pooled point centers
python3 inference_voxel_viz.py --ckpt checkpoints/best.pt --points <sample.npz> --color feature_norm
python3 inference_voxel_viz.py --ckpt checkpoints/best.pt --points <sample.npz> --color height
python3 inference_voxel_viz.py --ckpt checkpoints/best.pt --points <sample.npz> --show-points

# Draw fewer voxel cubes per window for faster rendering
python3 inference_voxel_viz.py --ckpt checkpoints/best.pt --points <sample.npz> --max-voxels 3000

# Make the random point sampling reproducible
python3 inference_voxel_viz.py --ckpt checkpoints/best.pt --points <sample.npz> --seed 0
```

Notes:

- `inference_voxel_viz.py` only supports PTv3 checkpoints. A PointNet++ (`pn2`)
  checkpoint has no voxel-pooling layers to hook.
- The input `.npz` can be any generated file containing a `points` array. The
  loader also accepts `.npy`, `.ply`, `.pcd`, `.xyz`, and `.txt` point clouds.
- `--color feature_norm` colors pooled voxels by the feature norm after each
  pooling layer. The input voxelization falls back to height coloring because
  no model features exist yet.

## Key design decisions

- **No translation re-scaling on grasps**: ACRONYM stores grasp transforms
  already in the scaled object frame (real-world metres). Only the
  mesh-mean shift is applied.
- **Object bottom flush with table**: `obj_z = table_z/2 - mesh.bounds[0][2]`
  ensures no clipping through the table surface.
- **Surface projection**: Grasp wrist (TCP) is projected onto the nearest mesh
  vertex before label assignment, since the TCP is ~10cm from the object.
- **Object mask**: Points are classified as object vs table by checking their
  world-frame z coordinate. The KDTree is built from object points only,
  preventing grasp labels from leaking onto the table.

## Training-time dataset selection

`CGNDataset` (see `data/dataset.py`) reads `manifest.json` + the
budget JSON and filters the file list accordingly:

- `split="train"`: mesh entries with `split=="train"` and
  `rank <= train_objects_per_category`. A random fraction of the
  **views** is held back as `split="val"` (same meshes, different
  renders) for monitoring training.
- `split="test"`: mesh entries with `split=="test"` - completely
  unseen meshes, used to measure novel-instance generalization.

Training CLI:

```bash
# Use the active preset from training_budgets.json (5_per_cat by default)
python train.py

# Explicit preset
python train.py --budget_preset 2_per_cat

# Numeric override (wins over preset)
python train.py --train_objects_per_category 1
```
