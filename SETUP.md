# Setup

## Environment

```bash
conda create -n cgt python=3.11 -y
conda activate cgt
```

For an NVIDIA GPU, install the PyTorch build that matches your driver before the project dependencies.
For CUDA 12.8:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

For CPU-only PyTorch, install the CPU wheel before the project dependencies:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Install the full project dependencies:

```bash
python -m pip install -r requirements.txt -r requirements-data.txt -r requirements-viz.txt -r requirements-eval.txt
```

Dependencies use four files:

| File | Purpose |
| --- | --- |
| `requirements.txt` | Training and prediction |
| `requirements-data.txt` | Data generation, mesh loading, and H5 export |
| `requirements-viz.txt` | Open3D viewers and plots |
| `requirements-eval.txt` | MuJoCo evaluation, with the data dependencies |

For training on existing samples, install only `requirements.txt`.
For CLI inference with H5 export, also install `requirements-data.txt`.
For sample visualization, install the core, data, and visualization dependencies.

Training examples use `--cpe_mode knn` and disable W&B logging.
For `--cpe_mode sparse3d`, install a `spconv` wheel compatible with your CUDA environment.
Checkpoints retain their positional encoding choice and require the same backend for inference.

## Dataset

Obtain the ACRONYM grasp files and their matching ShapeNet meshes separately.
The subset builder expects this source layout:

```text
/path/to/acronym/
├── meshes/<Category>/<mesh_hash>.obj
└── grasps/<Category>_<mesh_hash>_<scale>.h5
```

Prepare a dedicated project subset:

```bash
python data/acronym/build_acronym_subset.py \
  --src /path/to/acronym --dst data/acronym
```

The builder selects 10 training meshes and 2 test meshes per category.
It writes `manifest.json` and removes destination assets outside the selected subset.
The committed manifest contains paths and metadata, not the mesh or grasp files.

Generate samples:

```bash
python data/generate_data.py
```

For a smaller run, use `--category Mug --n_views 5`.
Generated samples appear under `data/out/<split>/<Category>/<mesh_hash>/`.
Training budgets are defined in `data/acronym/training_budgets.json`.

## Rendering and Simulation

For headless data rendering, set EGL before running the generator:

```bash
PYOPENGL_PLATFORM=egl python data/generate_data.py
```

Open3D and interactive MuJoCo viewers require a graphical display.
For headless MuJoCo evaluation, use `--no_viewer --skip_preview`.
The first evaluation run can download the Panda model through `robot_descriptions`.

Evaluate generated labels with `--source labels` and a `--view_npz` path.
Evaluate model predictions with `--source pred_cgn` or `--source pred_ptv3` and `--checkpoint`.
All three modes require the matching object mesh and `data/acronym/manifest.json`.
