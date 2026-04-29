# Contact-GraspNet With Point Transformer V3

Refer to [SETUP.md](SETUP.md) for environment setup.

This project trains and evaluates a point-cloud grasp detection model for
6-DoF robotic manipulation. It adapts the Contact-GraspNet prediction heads to
two interchangeable backbones:

- **PointNet++ (`pn2`)** as a compact baseline for dense point-cloud features.
- **Point Transformer V3 (`ptv3`)** with voxel pooling, space-filling-curve
  serialization, windowed attention, and configurable conditional positional
  encoding.

The end-to-end pipeline covers ACRONYM subset preparation, synthetic depth and
point-cloud rendering, per-point grasp label generation, training, inference,
and interactive visualization of both grasp labels and PTv3 voxel behavior.

## Project Highlights

- Generates Contact-GraspNet-style training samples from an ACRONYM object
  subset across 15 everyday object categories.
- Trains shared CGN heads for grasp confidence, approach/base directions, and
  gripper width.
- Supports both PointNet++ and PTv3 backbones from the same `ContactGraspNet`
  wrapper.
- Exports inference results as ACRONYM-layout `.h5` files plus JSON sidecars
  that downstream simulators can use to recover mesh metadata.
- Includes Open3D visualization tools for rendered samples, grasp labels,
  synthetic PTv3 voxel stages, and real checkpoint voxel pooling.

## Quick Start

Follow the full environment instructions in [`SETUP.md`](./SETUP.md).

```bash
conda create -n idlsproj python=3.9 -y
conda activate idlsproj
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

On a CPU-only machine, install the CPU PyTorch wheel instead of the CUDA wheel.
On a headless server, set `PYOPENGL_PLATFORM=egl` before running rendering
scripts. See [`SETUP.md`](./SETUP.md) for verification commands and Linux
Wayland display notes.

## Common Workflows

### Generate Training Data

The data pipeline renders ACRONYM meshes from multiple camera views, back-projects
depth to point clouds, and assigns per-point grasp labels.

```bash
# Full dataset generation
python data/generate_data.py

# Generate one category for faster iteration
python data/generate_data.py --category Mug

# Quick debug render
python data/generate_data.py --category Mug --n_views 5 --n_points 4096
```

Detailed data documentation lives in [`data.md`](./data.md), including the
ACRONYM subset layout, `manifest.json` schema, output `.npz` keys, coordinate
frames, and training budget presets.

### Train a Model

```bash
# Default PTv3 training
python train.py --data_dir data/out --backbone ptv3 --epochs 10

# PointNet++ baseline
python train.py --data_dir data/out --backbone pn2 --epochs 10

# Use a named object-budget preset
python train.py --budget_preset 2_per_cat
```

Training saves `best.pt` and `last.pt` checkpoints under `checkpoints/` by
default. Hyper-parameter sweeps are configured in
[`sweep_config.yaml`](./sweep_config.yaml), and architecture details are
documented in [`model.md`](./model.md).

### Run Inference

```bash
python inference.py \
  --ckpt checkpoints/best.pt \
  --points data/out/train/Camera/<mesh_hash>/001.npz \
  --top-k 100 \
  --score-thresh 0.5
```

Inference reads one point cloud (`.npz`, `.npy`, `.ply`, `.pcd`, `.xyz`, or
`.txt`) and returns ranked Panda-hand grasp poses in the same frame as that
cloud. Generated-sample paths can automatically provide category and mesh
metadata through `data/acronym/manifest.json`.

Typical outputs:

```text
out/<Category>_<mesh_hash>_<scale>.h5
out/<Category>_<mesh_hash>_<scale>.json
```

The `.h5` uses the ACRONYM grasp layout, while the `.json` sidecar records the
checkpoint, point-cloud source, frame, score settings, and mesh path/scale when
available.

### MuJoCo Grasp Validation

Use MuJoCo validation to compare whether dataset labels or model-predicted
grasps physically lift the target object. The recommended comparison path is to
run both checkpoints on the same generated `.npz` view:

```bash
# PointNet++ baseline
python -m eval.visualize_grasp \
  --source pred_cgn \
  --checkpoint <pointnetpp_checkpoint.pt> \
  --view_npz data/out/test/Mug/40f9a6cc6b2c3b3a78060a3a3a55e18f/000.npz \
  --start_delay_s 0

# PTv3 model
python -m eval.visualize_grasp \
  --source pred_ptv3 \
  --checkpoint <ptv3_checkpoint.pt> \
  --view_npz data/out/test/Mug/40f9a6cc6b2c3b3a78060a3a3a55e18f/000.npz \
  --start_delay_s 0
```

The `.npz` supplies the point cloud and frame information; the matching MuJoCo
mesh is resolved from `manifest.json`. Success is based on target-object lift.
Add `--no_viewer --skip_preview` when running headless batches.
See [`SETUP.md`](./SETUP.md#grasp-visualization--mujoco-execution) for dataset
label replay, raw ACRONYM H5 replay, and exported prediction replay commands.

### Visualize Data And Voxels

```bash
# Rendered depth / point cloud / grasp labels
python data/visualizer.py data/out/train/Mug/<mesh_hash>/000.npz
python data/visualizer.py data/out/train/Mug/<mesh_hash>/001.npz --mode grasps

# Synthetic PTv3 voxel and serialization views
python voxel_viz.py data/out/train/Mug/<mesh_hash>/000.npz --mode all

# Real voxel pooling from a trained PTv3 checkpoint
python inference_voxel_viz.py \
  --ckpt checkpoints/best.pt \
  --points data/out/train/Mug/<mesh_hash>/000.npz
```

See the visualization section in [`data.md`](./data.md#voxel-visualization-tools)
for modes, options, and display troubleshooting.

## Repository Guide

| Path | Purpose |
|---|---|
| [`SETUP.md`](./SETUP.md) | Environment setup, installation, verification, and quick commands. |
| [`data.md`](./data.md) | ACRONYM subset, data generation, output schemas, visualization, and training-data selection. |
| [`model.md`](./model.md) | PointNet++, PTv3, CGN heads, and training hyper-parameter documentation. |
| [`train.py`](./train.py) | Main training entry point with checkpointing and W&B logging. |
| [`inference.py`](./inference.py) | Point-cloud-to-grasp inference CLI and programmatic predictor. |
| [`data/generate_data.py`](./data/generate_data.py) | Synthetic render and label generation pipeline. |
| [`data/dataset.py`](./data/dataset.py) | Dataset loader and train/val/test object-budget filtering. |
| [`data/visualizer.py`](./data/visualizer.py) | Open3D visualization for generated `.npz` samples. |
| [`voxel_viz.py`](./voxel_viz.py) | Explanatory PTv3 voxelization, pooling, CPE, and serialization views. |
| [`inference_voxel_viz.py`](./inference_voxel_viz.py) | Checkpoint-backed PTv3 voxel-pooling visualization. |
| [`models/`](./models) | ContactGraspNet wrapper, backbones, and prediction heads. |
| [`loss.py`](./loss.py) | CGN training losses for confidence, pose directions, and width. |
| [`requirements.txt`](./requirements.txt) | Python dependencies. |

## Data Layout

The expected local ACRONYM subset is:

```text
data/acronym/
├── manifest.json
├── training_budgets.json
├── meshes/<Category>/*.obj
└── grasps/*.h5
```

Generated samples are written under `data/out/`:

```text
data/out/
├── train/<Category>/<mesh_hash>/<view>.npz
└── test/<Category>/<mesh_hash>/<view>.npz
```

Each `.npz` contains the rendered depth image, regularized point cloud,
per-point grasp labels, widths, and camera pose. The active training budget is
controlled by `data/acronym/training_budgets.json` or by `train.py` CLI flags.

## Inference Output And Frames

All inference grasp transforms are emitted in the **same coordinate frame as the
input point cloud**. For generated training samples, that is the saved camera
frame for the view. For arbitrary sensor scans, it is whatever frame the scan
already uses.

The exported `.h5` contains:

- `grasps/transforms`: `(K, 4, 4)` SE(3) transforms in Panda-hand convention.
- `grasps/qualities/flex/object_in_gripper`: binary success labels from scores.
- `grasps/widths`: target gripper widths in metres.
- Convenience arrays such as `scores`, `positions`, `quaternions`, and
  `contacts`.

Use the JSON sidecar to locate the matching mesh and scale in a simulator when
the input came from the generated ACRONYM-style dataset.

## References

- [`CGN.pdf`](./CGN.pdf): Contact-GraspNet reference paper/material.
- [`ptv3.pdf`](./ptv3.pdf): Point Transformer V3 reference paper/material.
- [`model.md`](./model.md): Local architecture notes and code references.
- [`data.md`](./data.md): Local data-generation and visualization notes.
