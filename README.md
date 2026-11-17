# Contact-GraspTransformer

[Project page](https://devanshroyal-7.github.io/Contact-Grasp-Transformer/)

Point-cloud grasp detection with Contact-GraspNet heads and interchangeable
PointNet++ and Point Transformer V3 (PTv3)-based backbones. The project extends the
[Contact-GraspNet baseline](https://github.com/dineshvarunshankar/Contact-GraspNet)
with a PTv3-based backbone, [ACRONYM](https://arxiv.org/abs/2011.09584) data generation, and MuJoCo grasp evaluation.

The pipeline renders ACRONYM meshes, assigns per-point grasp labels, trains a
model, and predicts 6-DoF Panda gripper poses. MuJoCo evaluates whether each
predicted grasp lifts the target object.

## Setup

```bash
git clone https://github.com/devanshroyal-7/Contact-GraspTransformer.git
cd Contact-GraspTransformer
conda create -n cgt python=3.11 -y
conda activate cgt
python -m pip install -r requirements.txt -r requirements-data.txt -r requirements-viz.txt -r requirements-eval.txt
```

See [SETUP.md](SETUP.md) for GPU installation and dataset preparation.
Meshes, grasp files, generated samples, and trained checkpoints are not included.

## Run

Run commands from the repository root after preparing the dataset.

Generate the training and test views:

```bash
python data/generate_data.py
```

Train either backbone with separate checkpoint directories:

```bash
python train.py --backbone pn2 --wandb_mode disabled
python train.py --backbone ptv3 --cpe_mode knn --wandb_mode disabled
```

Training saves `best.pt` and `last.pt` under `checkpoints/<backbone>/<run_folder>/`.
Each run folder includes the configuration and timestamp.

The PTv3 command uses k-NN positional encoding. Sparse 3D encoding requires `spconv`.

Predict grasps from a point cloud:

```bash
python inference_cli.py --ckpt "checkpoints/ptv3/<run_folder>/best.pt" \
  --points path/to/view.npz --top-k 100
```

Inference writes H5 predictions and JSON metadata to `out/`.
Grasp poses use the input point cloud's coordinate frame.

Evaluate a checkpoint on a generated test view:

```bash
python -m eval.visualize_grasp --source pred_ptv3 \
  --checkpoint "checkpoints/ptv3/<run_folder>/best.pt" \
  --view_npz path/to/generated/test/view.npz --top_k 5 --start_delay_s 0
```

For PointNet++, use `--source pred_cgn` and its checkpoint.
For a headless run, add `--no_viewer --skip_preview`.

## Code

- `data/`: subset preparation, rendering, labels, and dataset loading.
- `models/`, `loss.py`, `train.py`: backbones, prediction heads, losses, and training.
- `inference.py`, `inference_cli.py`: grasp prediction and export.
- `viz/`: sample, voxel, and serialization visualization.
- `eval/`: Panda simulation and object-lift evaluation.
- [data.md](data.md) and [model.md](model.md): data and architecture details.

## References

- [Contact-GraspNet: Efficient 6-DoF Grasp Generation in Cluttered Scenes](https://arxiv.org/abs/2103.14127), Sundermeyer et al., ICRA 2021.
- [Point Transformer V3: Simpler, Faster, Stronger](https://arxiv.org/abs/2312.10035), Wu et al., CVPR 2024.
- [ACRONYM: A Large-Scale Grasp Dataset Based on Simulation](https://arxiv.org/abs/2011.09584), Eppner, Mousavian, and Fox, ICRA 2021.
