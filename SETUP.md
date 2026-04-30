# Environment Setup

## 1) Create Environment

```bash
conda create -n idlsproj python=3.9 -y
conda activate idlsproj
```

## 2) Install PyTorch

```bash
# CUDA 12.8 example
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# CPU-only fallback
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 3) Install Project Dependencies

```bash
python -m pip install -U pip
pip install -r requirements.txt
pip install obj2mjcf robot_descriptions
```

## 4) Verify

```bash
python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available())"
python -c "import trimesh, open3d, h5py; print('geometry imports OK')"
python -c "import mujoco; print('MuJoCo OK')"
```

## Data + Training Quick Commands

```bash
# Generate data
python data/generate_data.py

# Generate only Mug
python data/generate_data.py --category Mug

# Quick test (fewer views/points)
python data/generate_data.py --category Mug --n_views 5 --n_points 4096

# Example rendered Mug mesh hash
# 2997f21fa426e18a6ab1a25d0e8f3590

# Visualize a single rendered view (depth + point cloud)
python data/visualizer.py data/out/train/Mug/2997f21fa426e18a6ab1a25d0e8f3590/000.npz

# Depth-only grid of all views for one Mug mesh
python data/visualizer.py data/out/train/Mug/2997f21fa426e18a6ab1a25d0e8f3590/ --mode depth --grid

# Point cloud coloured by grasp confidence
python data/visualizer.py data/out/train/Mug/2997f21fa426e18a6ab1a25d0e8f3590/001.npz --mode grasps

# Train
python train.py --data_dir data/out --backbone ptv3 --epochs 10
```

> **Linux Wayland note:** If `open3d` / GLFW fails to create a window, run the
> visualizer through XWayland:
>
> ```bash
> env WAYLAND_DISPLAY= XDG_SESSION_TYPE=x11 GDK_BACKEND=x11 DISPLAY=:0 \
> python data/visualizer.py data/out/train/Mug/2997f21fa426e18a6ab1a25d0e8f3590/000.npz
> ```

## Grasp Visualization + MuJoCo Execution

Use `eval.visualize_grasp` for MuJoCo lift validation. MuJoCo always simulates
the matching object mesh; the `--source` flag only decides where the grasp comes
from.

```bash
python -m eval.visualize_grasp ...
```

### Recommended Workflow

```bash
# 1) Visual sanity check for generated dataset labels from one data/out view.
# The .npz path identifies the object; the matching mesh is found automatically.
python -m eval.visualize_grasp \
  --source labels \
  --view_npz data/out/test/Mug/40f9a6cc6b2c3b3a78060a3a3a55e18f/000.npz \
  --start_delay_s 0 \
  --top_k 1

# 2) Batch metric check for the same dataset labels.
python -m eval.visualize_grasp \
  --source labels \
  --view_npz data/out/test/Mug/40f9a6cc6b2c3b3a78060a3a3a55e18f/000.npz \
  --no_viewer \
  --skip_preview \
  --start_delay_s 0 \
  --top_k 10

# 3) Evaluate a PointNet++ checkpoint on the same view.
python -m eval.visualize_grasp \
  --source pred_cgn \
  --checkpoint models/checkpoints/best_pn2.pt \
  --view_npz data/out/test/Mug/40f9a6cc6b2c3b3a78060a3a3a55e18f/000.npz \
  --start_delay_s 0 \
  --top_k 5

# 4) Evaluate a PTv3 checkpoint on the same view.
python -m eval.visualize_grasp \
  --source pred_ptv3 \
  --checkpoint models/checkpoints/best_ptv3.pt \
  --view_npz data/out/test/Mug/40f9a6cc6b2c3b3a78060a3a3a55e18f/000.npz \
  --start_delay_s 0 \
  --top_k 5

# Optional: show GT labels and model predictions side-by-side in Trimesh.
python -m eval.visualize_grasp \
  --source pred_cgn \
  --checkpoint <pointnetpp_checkpoint.pt> \
  --view_npz data/out/test/Mug/40f9a6cc6b2c3b3a78060a3a3a55e18f/000.npz \
  --start_delay_s 0 \
  --top_k 5 \
  --compare_labels_preview \
  --preview_all_grasps
```

Use `--no_viewer --skip_preview` only for headless batch runs after the visual
path looks correct.

### Source Modes

| Source | Grasp comes from | Main use |
|---|---|---|
| `labels` | Generated `data/out/*.npz` labels | Validate the dataset-to-MuJoCo pipeline. |
| `pred_cgn` | PointNet++ checkpoint predictions from an `.npz` point cloud | Evaluate PointNet++ model grasps. |
| `pred_ptv3` | PTv3 checkpoint predictions from an `.npz` point cloud | Evaluate PTv3 model grasps. |
| `dataset_h5` | Raw ACRONYM `.h5` grasps | Low-level simulator sanity check. |
| `model_h5` | Exported model `.h5` + `.json` predictions | Replay saved predictions without loading a checkpoint. |

### Specific Object Selection

Prefer passing an explicit `--view_npz`:

```bash
python -m eval.visualize_grasp \
  --source labels \
  --view_npz data/out/<split>/<Category>/<mesh_hash>/<view>.npz \
  --start_delay_s 0 \
  --top_k 1
```

The path encodes the split, category, and mesh hash, so the script can resolve
the matching `data/acronym/meshes/.../*.obj` and scale from `manifest.json`.

You can also auto-select by split/category/hash:

```bash
python -m eval.visualize_grasp \
  --source labels \
  --split test \
  --category Mug \
  --mesh_hash 40f9a6cc6b2c3b3a78060a3a3a55e18f \
  --view_index 0 \
  --start_delay_s 0 \
  --top_k 1
```

### Raw H5 And Exported Prediction Replay

These modes are optional for normal model comparison.

```bash
# Raw ACRONYM H5 replay: simulator sanity check.
python -m eval.visualize_grasp \
  --source dataset_h5 \
  --grasp_h5 data/acronym/grasps/Mug_2997f21fa426e18a6ab1a25d0e8f3590_0.01929277648152453.h5 \
  --grasp_index 50 \
  --top_k 1

# Exported model prediction replay.
python -m eval.visualize_grasp \
  --source model_h5 \
  --grasp_h5 /path/to/model_grasps.h5 \
  --grasp_json /path/to/model_grasps.json \
  --top_k 1
```

### Success Metric And Notes

- Success is physical target-object lift: `object_lift_m >= 0.03`.
- Pose error is diagnostic only; hand motion alone is not success.
- The MuJoCo executor reaches/reorients the gripper, pauses open for `--pre_close_pause_s` seconds, then closes, pauses, and lifts.
- `labels` mode ranks generated candidates because `.npz` label confidence is usually binary.
- `--label_conf_thresh` defaults to `0.5` and only controls generated GT/label selection, not model predictions.
- `pred_cgn` and `pred_ptv3` rank all model-predicted point grasps by confidence.
- `--top_k` previews ranked candidates together in Trimesh, then runs one MuJoCo trial per grasp.
- `--compare_labels_preview` shows GT labels on the left and model predictions on the right. MuJoCo still executes the selected model grasps for `pred_cgn`/`pred_ptv3`.
- Add `--preview_all_grasps` to draw a capped translucent-orange background set; selected GT top-k is green and selected model top-k is blue.
- `--preview_all_limit` defaults to `40` per side. Increase it for denser context, or use `0` only when you really want every valid candidate.
- PTv3 checkpoint loading uses the checkpoint's saved `cpe_mode` by default. Sparse3D PTv3 checkpoints require `spconv` in the evaluation environment.
- `dataset_h5` and `model_h5` use H5 object mass when present.
- `model_h5` requires non-empty `grasps/transforms`, `grasps/widths`, and `grasps/scores`.
