"""Visualization tools for generated samples, PTv3 voxels, and SFC diagrams.

Ownership:
- ``viz/dataset_visualizer.py`` — Open3D/matplotlib views of generated ``.npz``
  samples and ACRONYM GT grasps (dataset artifacts).
- ``viz/voxel_viz.py`` — synthetic / explanatory PTv3 voxelization views.
- ``viz/inference_voxel_viz.py`` — checkpoint-backed PTv3 voxel pooling views.
- ``viz/space_f_curves.py`` — space-filling-curve diagram generator.

Run modules as ``python -m viz.<module> ...`` from the repo root (or
``python viz/<module>.py``, which bootstraps the repo root only in script mode).
"""
