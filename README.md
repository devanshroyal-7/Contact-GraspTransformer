# Contact-GraspNet (Transformer) — Inference

See [`SETUP.md`](./SETUP.md) for environment setup, data generation, training,
and visualization. This file documents **running inference** with a trained
checkpoint and generating the ACRONYM-style `.h5` + metadata `.json` that a
physics simulator can consume.

## TL;DR

```bash
python inference.py \
  --ckpt checkpoints/best.pt \
  --points data/out/train/Camera/<mesh_hash>/001.npz \
  --top-k 100 --score-thresh 0.5
```

Produces:

```
out/<Category>_<mesh_hash>_<scale>.h5     # ACRONYM-layout grasps
out/<Category>_<mesh_hash>_<scale>.json   # mesh + run metadata
```

## How the object is chosen

Inference reads **one point cloud** (`--points`). That cloud *is* the object —
the model has no category/hash input. Whatever you pass in is what the network
labels per-point.

`--category`, `--mesh-hash`, `--manifest`, and `--acronym-root` only affect the
**metadata** written into `out/<stem>.json` (and the filename stem). They let a
downstream sim locate the matching mesh for visualization. If the input path
follows `data/out/<split>/<Category>/<mesh_hash>/<view>.npz`, those values are
inferred from the path and cross-referenced with `manifest.json` automatically.

## CLI reference

| Flag | Default | Purpose |
|---|---|---|
| `--ckpt` | required | Checkpoint from `train.py` (`.pt`). |
| `--points` | required | `.npy`/`.npz`/`.ply`/`.pcd`/`.xyz`. `.npz` reads a `points` key if present. |
| `--out-dir` | `out` | Directory that will hold `<stem>.h5` + `<stem>.json`. Created if missing. |
| `--run-name` | auto | Override the output stem. Default is `<Category>_<mesh_hash>_<scale>` when resolvable, else `<points-basename>_pred`. |
| `--manifest` | `data/acronym/manifest.json` | ACRONYM manifest for `scale` / `mesh_path` / `grasp_file` lookup. |
| `--acronym-root` | `data/acronym` | Root used to make the mesh path absolute in the JSON. |
| `--category` / `--mesh-hash` | `None` | Force identity instead of inferring from the path. Metadata only. |
| `--also-npz` | off | Additionally write `<stem>.npz` with all grasp arrays. |
| `--backbone` | `ptv3` | Fallback only if the checkpoint has no embedded `config`. |
| `--num-points` | `4096` | Fallback N if `config` is missing. |
| `--cpe-mode` | auto | PTv3 CPE mode override; normally auto-detected from the weights. |
| `--score-thresh` | `0.5` | Drop grasps with confidence below this. |
| `--top-k` | `100` | Keep only the top-K by score. |
| `--nms-radius` | `0.02` | Greedy NMS on grasp position (metres). `0` disables. |
| `--device` | auto | `cuda` / `cpu`. Defaults to CUDA if available. |
| `--seed` | `0` | Reproducible point sampling. |

## Output files

### `out/<stem>.h5` (ACRONYM layout)

The same groups used by `data/generate_data.py`, `data/visualizer.py`, and
`data/viz_test.py`:

- `grasps/transforms` — `(K, 4, 4)` float32 SE(3), **panda_hand** convention,
  in the **input point cloud's frame** (see "Frames" below).
- `grasps/qualities/flex/object_in_gripper` — `(K,)` uint8, set from
  `scores >= 0.5`.
- `grasps/widths` — `(K,)` float32 target gripper opening (metres).
- Additional (non-ACRONYM) datasets for convenience:
  `grasps/scores`, `grasps/positions`, `grasps/quaternions` (xyzw),
  `grasps/contacts`.
- `object` group attrs when resolvable: `file`, `scale`, `category`,
  `mesh_hash`.

### `out/<stem>.json` (sidecar)

```jsonc
{
  "run_name": "Camera_155ffb08..._0.0007...",
  "h5": "out/Camera_155ffb08..._0.0007....h5",
  "points": "/abs/path/.../001.npz",
  "ckpt": "/abs/path/checkpoints/best.pt",
  "frame": "input_point_cloud",
  "num_grasps": 100,
  "score_thresh": 0.5,
  "top_k": 100,
  "nms_radius": 0.02,
  "mesh": {
    "category": "Camera",
    "mesh_hash": "155ffb08...",
    "scale": 0.0007...,
    "mesh_path": "meshes/Camera/155ffb08....obj",
    "grasp_file": "Camera_155ffb08..._0.0007....h5",
    "acronym_root": "/abs/path/data/acronym",
    "mesh_path_abs": "/abs/path/data/acronym/meshes/Camera/155ffb08....obj"
  }
}
```

This is what a sim should read to locate both the grasps (`h5`) and the mesh
(`mesh.mesh_path_abs` at `mesh.scale`).

## Frames (important for sim)

- Output `grasps/transforms` are in the **same frame as the input cloud**.
- If `--points` is a training sample rendered by `data/generate_data.py`, that
  frame is the **camera frame** used for that view (the `.npz` also stores
  `camera_pose` if you need to transform back to world).
- If `--points` is an arbitrary depth reading, the frame is whatever that
  reading is already in.

To overlay on the mesh in a simulator you must place the mesh in the same
frame as the cloud. For training samples, the generation pipeline centres the
mesh at `mesh_mean` and scales by `manifest.scale` before placing it at
`obj_pose` (world). The saved `.json` gives you the information to reconstruct
that placement.

## Examples

### 1. Training sample (auto-infers mesh identity)

```bash
python inference.py \
  --ckpt checkpoints/best.pt \
  --points data/out/train/Camera/155ffb08fba5df33f0c6f578f0594c3/001.npz \
  --top-k 200 --score-thresh 0.3 --nms-radius 0.02
```

Outputs:

```
out/Camera_155ffb08fba5df33f0c6f578f0594c3_0.0007923...h5
out/Camera_155ffb08fba5df33f0c6f578f0594c3_0.0007923...json
```

### 2. Arbitrary `.npy` cloud (no mesh metadata)

```bash
python inference.py \
  --ckpt checkpoints/best.pt \
  --points scans/scene.npy \
  --run-name scene_run1
```

Outputs `out/scene_run1.h5` + `out/scene_run1.json` with an empty `mesh`
section — the grasps themselves are still correct, there's just nothing to
overlay them on.

### 3. Force a specific mesh identity (e.g. custom cloud of a known object)

```bash
python inference.py \
  --ckpt checkpoints/best.pt \
  --points scans/mug_scan.ply \
  --category Mug --mesh-hash 2997f21fa426e18a6ab1a25d0e8f3590 \
  --manifest data/acronym/manifest.json
```

The sidecar will then point to that mesh at its manifest `scale`.

## Prerequisites

- A trained checkpoint at `checkpoints/best.pt` (or wherever you point
  `--ckpt`). `train.py` saves `best.pt` + `last.pt` into `checkpoints/`.
- `h5py` installed (already in `requirements.txt`).
- For `.ply` / `.pcd` inputs, `open3d` (optional, installed by `requirements.txt`).
- For training-sample inputs, the ACRONYM subset at `data/acronym/` with
  `manifest.json` is needed only to populate the `.json` sidecar; inference
  itself does not need it.
