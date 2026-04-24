# Environment Setup

## 1. Create the conda environment

```bash
conda create -n idlsproj python=3.9 -y
conda activate idlsproj
```

## 2. Install PyTorch (pick your CUDA version)

```bash
# CUDA 12.8 (adjust for your driver — check with `nvidia-smi`)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# CPU-only (if no GPU)
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 3. Install the rest

```bash
pip install -r requirements.txt
```

> **Note:** `pyrender` needs OpenGL. On a headless server set `export PYOPENGL_PLATFORM=egl`
> before running any rendering scripts. On a desktop with a display this is not needed.

## 4. Verify

```bash
python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available())"
python -c "import pyrender, trimesh, open3d, h5py; print('All imports OK')"
```

## Quick commands

```bash
# Generate depth + point cloud + grasp labels for all objects (36 views each)
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
