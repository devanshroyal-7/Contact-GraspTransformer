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

Entrypoint:

```bash
python -m eval.visualize_grasp ...
```

Pipeline:

```text
grasp source -> top-1 grasp selection -> blocking Trimesh preview -> MuJoCo -> startup wait -> approach -> exact grasp -> close -> lift
```

### Common commands

```bash
# Dataset grasp
python -m eval.visualize_grasp --source labels --split test --top_k 1

# Specific object/view
python -m eval.visualize_grasp \
  --source labels \
  --split test \
  --category Pencil \
  --mesh_hash 3537584773badbfde015d304bda1df1d \
  --view_index 0

# Headless run
python -m eval.visualize_grasp \
  --source labels \
  --split test \
  --category Pencil \
  --mesh_hash 3537584773badbfde015d304bda1df1d \
  --no_viewer

# Repo-trained CGN checkpoint
python -m eval.visualize_grasp \
  --source pred_cgn \
  --split test \
  --checkpoint <repo_trained_cgn_checkpoint.pt>

# Repo-trained PTv3 checkpoint
python -m eval.visualize_grasp \
  --source pred_ptv3 \
  --split test \
  --checkpoint <repo_trained_ptv3_checkpoint.pt>
```

### Flag reference

- `--source {labels,pred_cgn,pred_ptv3,dataset_h5,model_h5}`: grasp source.
- `--source dataset_h5`: replay ACRONYM ground-truth grasps directly from a dataset `.h5`.
- `--source model_h5`: replay model-exported grasps from `.h5` plus export metadata `.json`.
- `--split {train,test}`: split used when auto-selecting a generated view.
- `--category`: optional category filter for view selection.
- `--mesh_hash`: optional mesh filter for view selection.
- `--view_index`: index inside the filtered view list.
- `--view_npz`: explicit generated per-view `.npz` path.
- `--checkpoint`: required for `pred_cgn` and `pred_ptv3`.
- `--grasp_h5`: ACRONYM or model-exported grasp `.h5` for `dataset_h5` / `model_h5`.
- `--grasp_json`: model-export metadata JSON for `model_h5`.
- `--grasp_index`: starting grasp index/rank for labels and H5 modes.
- `--top_k`: number of ranked label/H5 grasps to simulate from `--grasp_index`.
- `--h5_hand_depth_offset_m`: calibration offset for H5 marker-derived Panda hand poses along the hand approach axis. The default is `0.03`.
- `--only_successful_dataset_grasps` / `--no-only_successful_dataset_grasps`: filter ACRONYM H5 grasps by `object_in_gripper`.
- `--device`: torch device for model-backed sources.
- `--mesh_path`: optional explicit mesh path. If omitted, the ACRONYM manifest resolves it.
- `--mesh_scale`: explicit mesh scale when using `--mesh_path`.
- `--no_viewer`: run MuJoCo headless.
- `--start_delay_s`: initial wait before arm approach begins (default: `5.0`).
- `--hold_viewer`: keep the MuJoCo viewer open after execution.
- `--show_viewer_ui`: show MuJoCo command/UI panes (default view is clean with panes hidden).
- `--max_steps`: hard cap on simulation steps.
- `--trimesh-preview {cgt,acronym}`: `cgt` shows the parallel wireframe, `acronym` shows NVlabs marker.

### Behavior notes

- Trimesh preview is blocking by default. Close the window to continue into MuJoCo.
- Trimesh preview shows the top-1 grasp wireframe by default (`cgt` parallel-jaw frame).
- The previewed grasp is the same grasp MuJoCo retargets to.
- MuJoCo executes one selected grasp per trial; batch runs use the ranked candidates from the source without physics-based re-ranking.
- `labels` mode ranks generated `data/out` candidates instead of using storage-order `argmax`; confidence is primary, then wider generated widths, then tabletop-friendly approach direction.
- H5 and label modes can evaluate multiple grasps with `--top_k`; each trial rebuilds the scene and reports a batch success rate.
- MuJoCo retargets directly from the grasp's contact point, approach direction, base direction, and width.
- `dataset_h5` retargets from ACRONYM object-local marker poses; `model_h5` retargets exported marker poses in the JSON-declared `input_point_cloud` frame.
- MuJoCo uses damped least-squares IK per phase only; no motion planner or HDF5 marker frame is used for execution.
- MuJoCo waits 5 seconds by default before starting approach motion (`--start_delay_s` to tune).
- MuJoCo first moves to a simple approach pose, then retargets the exact grasp pose, closes the gripper to the predicted width, and lifts.
- MuJoCo now waits for gripper closure before lift and pauses for 1 second after close before lifting.
- Close and lift phases use faster default timings than approach/reach, so post-retarget grasping is more responsive.
- Success is based on physical target-object lift, not hand motion. A grasp succeeds only when the target object rises by at least 3 cm.
- H5 replay applies a default `0.03m` hand-depth calibration so ACRONYM marker-derived poses place the Menagerie Panda fingertip pads at the expected contact depth.
- `dataset_h5` and `model_h5` use `object/mass` from the H5 when present; otherwise the scene falls back to the configured mesh density.
- Arm execution is physics-enabled by default: no kinematic pass-through branch is used in the normal pipeline.
- The default scene mounts the Panda at tabletop height and places the object in a reachable tabletop zone.
- The target object is dynamic by default; use `--static_object` only when you explicitly want a pinned debug object.
- `pred_cgn` and `pred_ptv3` expect checkpoints trained in this repo. External checkpoints from different architectures are not compatible.
- `model_h5` fails fast when the H5 contains zero grasps; re-export with a lower score threshold or a top-1 fallback before simulation.

### H5 simulation examples

```bash
# Replay successful ACRONYM dataset grasps directly from the dataset H5
python -m eval.visualize_grasp \
  --source dataset_h5 \
  --grasp_h5 data/acronym/grasps/Mug_2997f21fa426e18a6ab1a25d0e8f3590_0.01929277648152453.h5 \
  --grasp_index 50 \
  --no_viewer \
  --skip_preview \
  --top_k 1

# Replay model-exported grasps from an H5/JSON pair
python -m eval.visualize_grasp \
  --source model_h5 \
  --grasp_h5 /path/to/model_grasps.h5 \
  --grasp_json /path/to/model_grasps.json \
  --no_viewer \
  --skip_preview \
  --top_k 10
```

### Object selection

If you run:

```bash
python -m eval.visualize_grasp --source labels --split test
```

the script scans `data/out/test/*/*/*.npz`, sorts the matches, and uses `--view_index 0` by default.
Use `--category`, `--mesh_hash`, and `--view_index` together when you want a specific test object/view.
