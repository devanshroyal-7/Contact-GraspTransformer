# Data Generation Pipeline

## Source data (ACRONYM)

```
data/acronym/
├── manifest.json            5 objects: Mug, Bowl, Bottle, Cup, Pan
├── meshes/<Category>/*.obj  raw ShapeNet meshes
└── grasps/*.h5              grasp transforms + success labels
```

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
├── Mug/
│   ├── 000.npz
│   ├── 001.npz
│   └── ...          (36 views)
├── Bowl/
├── Bottle/
├── Cup/
└── Pan/
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
