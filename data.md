flowchart TD
    A["Grasp poses from HDF5<br/>(mesh-local frame)"] --> B["Scale translations<br/>by object/scale"]
    B --> C["Subtract mesh centroid<br/>(same as mesh.vertices -= mean)"]
    C --> D["Apply obj_pose<br/>(place on table)"]
    D --> E["Transform to camera frame<br/>(OpenGL-to-OpenCV + inv)"]
    E --> F["Subtract pc_mean<br/>(same mean-centring as point cloud)"]
    F --> G["KDTree: assign per-point labels"]
