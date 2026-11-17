# Model

[ContactGraspNet](models/model.py) combines a point-cloud backbone with shared grasp prediction heads.
Choose `pn2` or `ptv3` with `train.py --backbone`.
Both accept `(B, N, 3)` point coordinates and produce `(B, 64, N)` features.
Here, `B` is the batch size and `N` is the point count.

## Shared heads

[CGNHeads](models/cgn_heads.py) uses four pointwise prediction branches.
Each branch contains a 1×1 convolution, batch normalization, ReLU, and an output convolution.

| Output | Shape | Meaning |
| --- | --- | --- |
| `confidence_logits` | `(B, N)` | Grasp confidence before sigmoid |
| `confidence` | `(B, N)` | Grasp confidence after sigmoid |
| `base_dirs` | `(B, N, 3)` | Normalized baseline direction |
| `approach_dirs` | `(B, N, 3)` | Approach direction orthogonal to the baseline |
| `width_bin_logits` | `(B, N, 10)` | Width classification logits |
| `widths` | `(B, N)` | Selected bin centers in meters |

Gram–Schmidt orthogonalization produces the two direction vectors.
Ten equal width bins cover 0 to 0.08 meters.
The predicted width is the center of the bin with the largest logit.

## PointNet++ backbone

[SimplePointNet2](models/backbone_pn2.py) has four set-abstraction stages and four feature-propagation stages.
Set abstraction uses farthest-point sampling, radius grouping, and max pooling.

| Encoder stage | Sampled points | Radius (m) | Neighbors | MLP channels |
| --- | ---: | ---: | ---: | --- |
| 1 | 512 | 0.1 | 32 | 32, 32, 64 |
| 2 | 128 | 0.2 | 32 | 64, 64, 128 |
| 3 | 32 | 0.4 | 32 | 128, 128, 256 |
| 4 | 8 | 0.8 | 32 | 256, 256, 512 |

Feature propagation interpolates from three neighbors with inverse-distance weights.
From the deepest stage to the input resolution, its MLP channels are
`256, 256`, `256, 128`, `128, 64`, and `64, 64, 64`.

## PTv3-based backbone

[PTv3Wrapper](models/backbone_ptv3.py) is a custom transformer implementation with serialized attention and voxel pooling.
It quantizes coordinates, serializes points into attention windows, and combines encoder features through decoder skip connections.

| Setting | Default |
| --- | --- |
| Grid size | 0.01 m |
| Attention window | 256 points |
| Encoder channels | 32, 64, 128, 256 |
| Attention heads | 4, 4, 8, 8 |
| Encoder blocks per stage | 2, 2, 2, 2 |
| Decoder blocks per stage | 2, 2, 2 |
| Pooling strides | 2, 2, 2 |
| Maximum drop-path rate | 0.3 |

Serialization uses Z-order, transposed Z-order, Hilbert, and transposed Hilbert orders.
Training shuffles the order sequence. Evaluation uses a fixed sequence.
Each block applies conditional positional encoding, windowed attention, and an MLP with residual connections.
Padding masks exclude unused tokens from attention.
Voxel pooling averages features. The decoder gathers features through the pooling map and adds skip features.

The `--cpe_mode` flag selects positional encoding:

| Mode | Operation | Dependency |
| --- | --- | --- |
| `knn` | Spatial aggregation over 16 neighbors | PyTorch |
| `conv1d` | Depthwise convolution over serialized points | PyTorch |
| `sparse3d` | Sparse 3D convolution | `spconv` |

The README command selects `knn`. The training script defaults to `sparse3d`.

## Loss and training

[loss.py](loss.py) combines three terms:

- Binary cross-entropy on the 128 hardest confidence predictions per scene, or all points if fewer exist.
- Confidence-weighted distance between corresponding predicted and assigned gripper keypoints.
- Weighted binary cross-entropy on one-hot width bins.

Keypoint and width losses use positive labeled points.
The default weights are 1 for confidence, 10 for keypoint distance, and 1 for width classification.

[train.py](train.py) defaults to 10 epochs, batch size 4, and 4,096 points per sample.
It uses Adam with learning rate 0.001, no weight decay, and no learning-rate schedule.
Training requires generated samples. See [data.md](data.md).
Checkpoints include model configuration and weights.
Training saves `best.pt` and `last.pt` under `checkpoints/<backbone>/<run_folder>/`.
The `--checkpoint_dir` flag changes the root directory. Each run folder includes the configuration and timestamp.

## Inference

[inference_cli.py](inference_cli.py) calls the [prediction library](inference.py), which loads the checkpoint configuration and predicts per-point grasps.
It filters by confidence, limits the candidate count, and applies position-based non-maximum suppression.
Outputs include poses, positions, quaternions, widths, scores, and contact points.
Exported poses use the input point cloud's coordinate frame.

See [README.md](README.md) for training, inference, and MuJoCo evaluation commands.
