# Data

The pipeline renders ACRONYM objects and assigns grasp labels to sampled depth points.
Mesh assets and grasp files must be supplied separately. See [SETUP.md](SETUP.md).

## Source subset

```text
data/acronym/
├── manifest.json
├── training_budgets.json
├── meshes/<Category>/*.obj
└── grasps/*.h5
```

Build the subset from an existing ACRONYM directory:

```bash
python data/acronym/build_acronym_subset.py --src /path/to/acronym
```

The default destination is the script's directory, `data/acronym`.
Use `--dst` to change it or `--dry_run` to inspect the selection.
The builder copies selected assets and removes stale assets from the destination.

The subset contains 10 training objects and two test objects per category.
It selects objects in filename order, with one scale per mesh hash.
When multiple scales exist, it selects the first scale string in lexicographic order.

Categories: Mug, Bowl, Bottle, Cup, Pan, Book, Vase, CellPhone, Pencil,
ToyFigure, Knife, FoodItem, Camera, SodaCan, and WineBottle.

Each manifest entry contains:

| Field | Meaning |
| --- | --- |
| `category` | Object category |
| `mesh_hash` | Mesh identifier |
| `mesh_path` | OBJ path relative to the ACRONYM root |
| `grasp_file` | H5 filename under `grasps/` |
| `scale` | Mesh scale |
| `split` | `train` or `test` |
| `rank` | Selection order within the category and split |

## Generate views

Run from the repository root:

```bash
python data/generate_data.py
```

The defaults are 360 views per object and 4,096 points per view.
For a smaller generation run:

```bash
python data/generate_data.py --category Mug --n_views 5
```

The generator scales and centers each mesh, places it on a table, and renders depth.
It back-projects depth, crops the point cloud, samples points, and subtracts their mean.
Successful grasps provide confidence, direction, and width labels near projected surface centers.
Table points receive no positive grasp labels.

Output paths use this layout:

```text
data/out/<train|test>/<Category>/<mesh_hash>/<view_index>.npz
```

Each sample contains:

| Array | Shape | Meaning |
| --- | --- | --- |
| `depth` | `(H, W)` | Rendered depth in meters |
| `points` | `(N, 3)` | Points in the centered camera frame |
| `confidence` | `(N,)` | Binary grasp labels |
| `approach_dirs` | `(N, 3)` | Assigned approach directions |
| `base_dirs` | `(N, 3)` | Assigned baseline directions |
| `widths` | `(N,)` | Assigned gripper widths in meters |
| `camera_pose` | `(4, 4)` | Transform from world to centered camera coordinates |

## Training and validation

`training_budgets.json` sets the number of training objects per category.
Available presets select 1, 2, 5, or 10 objects. The active preset selects five.
The command-line flags take precedence in this order:

1. `--train_objects_per_category`
2. `--budget_preset`
3. The JSON `active_preset`

The loader reserves 20% of selected training views for validation, using seed 42.
Training and validation can contain views of the same objects.
The test split uses separate objects and is not limited by the training budget.

The loader resamples points and labels together to the requested point count.
Training augmentation applies a random 3D rotation and Gaussian position noise with a 0.001-meter standard deviation.

## Inspect a sample

```bash
python viz/dataset_visualizer.py path/to/view.npz --mode grasps
python viz/voxel_viz.py path/to/view.npz --mode all
```

These viewers require a graphical display.
