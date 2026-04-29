"""Scene construction for grasp visualization/execution.

Design goals:
- Menagerie-style environment visuals (skybox + checkerboard + directional light).
- Stable table/object placement for grasp debugging.
- Deterministic Panda initialization (home keyframe + matching controls).
"""

from __future__ import annotations

import hashlib
import os
import subprocess

import mujoco
import numpy as np
import trimesh


def _get_panda_xml_path() -> str:
    """Resolve the Panda MJCF path (robot_descriptions cache first)."""
    try:
        import robot_descriptions.panda_mj_description as panda_desc
        from robot_descriptions.loaders.mujoco import load_robot_description

        load_robot_description("panda_mj_description")
        return panda_desc.MJCF_PATH
    except Exception:
        pass

    candidates = [
        "mujoco_menagerie/franka_emika_panda/panda.xml",
        os.path.expanduser("~/.mujoco/menagerie/franka_emika_panda/panda.xml"),
        os.path.expanduser("~/.cache/robot_descriptions/mujoco_menagerie/franka_emika_panda/panda.xml"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)

    raise FileNotFoundError(
        "Cannot find Menagerie Panda MJCF. Install `robot_descriptions` "
        "or clone https://github.com/google-deepmind/mujoco_menagerie."
    )


CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


def _mesh_hash(mesh_path: str, scale: float) -> str:
    h = hashlib.md5()
    h.update(mesh_path.encode())
    h.update(f"{scale:.10f}".encode())
    return h.hexdigest()[:16]


def _decompose_mesh(obj_path: str, scale: float, cache_dir: str = CACHE_DIR) -> str:
    """Scale, center and convex-decompose mesh; return cache directory."""
    key = _mesh_hash(obj_path, scale)
    out_dir = os.path.join(cache_dir, key)

    if os.path.isdir(out_dir) and any(f.endswith(".xml") for f in os.listdir(out_dir)):
        return out_dir

    os.makedirs(out_dir, exist_ok=True)

    mesh = trimesh.load(obj_path, force="mesh")
    mesh.apply_scale(scale)
    mesh.vertices -= mesh.vertices.mean(axis=0)
    mesh.export(os.path.join(out_dir, "object.obj"))

    try:
        subprocess.run(
            [
                "obj2mjcf",
                "--obj-dir",
                out_dir,
                "--save-mjcf",
                "--decompose",
                "--overwrite",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("obj2mjcf not found — pip install obj2mjcf") from exc
    except subprocess.CalledProcessError as exc:
        print(
            f"Warning: obj2mjcf failed ({exc.stderr.strip()[:80]}), "
            "falling back to convex hull."
        )
        mesh.convex_hull.export(os.path.join(out_dir, "object_col.obj"))
        with open(os.path.join(out_dir, "object.xml"), "w", encoding="utf-8") as f:
            f.write("<mujoco/>")

    return out_dir


def _find_collision_meshes(cache_dir: str) -> list[str]:
    files = [
        f
        for f in sorted(os.listdir(cache_dir))
        if f.endswith(".obj") and ("collision" in f or "decomp" in f)
    ]
    if not files:
        files = [
            f for f in sorted(os.listdir(cache_dir)) if f.endswith(".obj") and f != "object.obj"
        ]
    return files or ["object.obj"]


# Geometry contract:
# - World floor follows Menagerie `scene.xml`: checker plane at z=0.
# - The table surface remains at z=TABLE_TOP_Z to stay aligned with data generation.
TABLE_DIMS = (1.2, 1.6, 0.6)
TABLE_TOP_Z = TABLE_DIMS[2] * 0.5
WORLD_FLOOR_Z = 0.0
# Place Panda on the tabletop near the back edge (not outside the table footprint).
ROBOT_TABLE_EDGE_INSET_M = 0.08
EVAL_OBJECT_OFFSET_XY = (-0.18, 0.0)
TABLETOP_THICKNESS_M = 0.04
TABLE_LEG_WIDTH_M = 0.10
TABLE_LEG_INSET_X_M = 0.12
TABLE_LEG_INSET_Y_M = 0.18
TABLE_APRON_THICKNESS_M = 0.03
TABLE_APRON_HEIGHT_M = 0.08
# Slight sink into floor plane to avoid visual "floating" from z-fighting.
TABLE_LEG_FLOOR_SINK_M = 0.01

TABLE_RGBA = (0.32, 0.32, 0.34, 1.0)
OBJECT_FRICTION = (1.2, 0.05, 0.01)
OBJECT_DENSITY = 150.0
OBJECT_FREEJOINT_DAMPING = (3.0, 3.0, 3.0, 0.3, 0.3, 0.3)
SETTLE_STEPS = 200
# Free joint + collision so the arm interacts physically (set False for deterministic static eval).
OBJECT_DYNAMIC = True


def _add_menagerie_environment(scene: mujoco.MjSpec, wb, *, floor_z: float) -> str:
    """Apply Menagerie-style visuals and checkerboard floor."""
    scene.stat.center = [0.3, 0.0, 0.4]
    scene.stat.extent = 1.0

    scene.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    scene.visual.headlight.ambient = [0.3, 0.3, 0.3]
    scene.visual.headlight.specular = [0.0, 0.0, 0.0]
    scene.visual.rgba.haze = [0.15, 0.25, 0.35, 1.0]
    scene.visual.global_.azimuth = 120
    scene.visual.global_.elevation = -20

    scene.add_texture(
        name="skybox",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.3, 0.5, 0.7],
        rgb2=[0.0, 0.0, 0.0],
        width=512,
        height=3072,
    )
    tex = scene.add_texture(
        name="groundplane",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        mark=mujoco.mjtMark.mjMARK_EDGE,
        rgb1=[0.2, 0.3, 0.4],
        rgb2=[0.1, 0.2, 0.3],
        markrgb=[0.8, 0.8, 0.8],
        width=300,
        height=300,
    )
    mat = scene.add_material(
        name="groundplane",
        textures=[tex.name],
        texuniform=True,
        texrepeat=[5.0, 5.0],
        reflectance=0.2,
    )

    wb.add_light(
        pos=[0.0, 0.0, 1.5],
        dir=[0.0, 0.0, -1.0],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    )
    wb.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0.0, 0.0, 0.05],
        pos=[0.0, 0.0, float(floor_z)],
        material=mat.name,
        contype=1,
        conaffinity=1,
    )
    return mat.name


def _add_table(wb, table_dims: tuple[float, float, float]) -> None:
    """Add a tabletop with aprons and four legs resting on the world floor."""
    sx = table_dims[0] * 0.5
    sy = table_dims[1] * 0.5
    table_top_z = table_dims[2] * 0.5
    tabletop_half_z = TABLETOP_THICKNESS_M * 0.5
    tabletop_center_z = table_top_z - tabletop_half_z
    leg_height = max(tabletop_center_z - (WORLD_FLOOR_Z - TABLE_LEG_FLOOR_SINK_M), 2e-3)
    leg_half_z = leg_height * 0.5
    leg_center_z = (tabletop_center_z + (WORLD_FLOOR_Z - TABLE_LEG_FLOOR_SINK_M)) * 0.5
    leg_half_xy = TABLE_LEG_WIDTH_M * 0.5
    leg_x = max(sx - TABLE_LEG_INSET_X_M, leg_half_xy)
    leg_y = max(sy - TABLE_LEG_INSET_Y_M, leg_half_xy)

    table_body = wb.add_body(name="table", pos=[0.0, 0.0, 0.0])
    table_body.add_geom(
        name="tabletop",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.0, 0.0, tabletop_center_z],
        size=[sx, sy, tabletop_half_z],
        rgba=list(TABLE_RGBA),
        friction=[1.0, 0.005, 0.001],
    )
    for idx, (lx, ly) in enumerate(
        [
            (leg_x, leg_y),
            (leg_x, -leg_y),
            (-leg_x, leg_y),
            (-leg_x, -leg_y),
        ]
    ):
        table_body.add_geom(
            name=f"table_leg_{idx}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[lx, ly, leg_center_z],
            size=[leg_half_xy, leg_half_xy, leg_half_z],
            rgba=list(TABLE_RGBA),
            friction=[1.0, 0.005, 0.001],
        )

    apron_half_z = min(TABLE_APRON_HEIGHT_M * 0.5, max(leg_half_z - 0.02, 0.02))
    apron_half_t = TABLE_APRON_THICKNESS_M * 0.5
    apron_center_z = tabletop_center_z - tabletop_half_z - apron_half_z
    apron_specs = [
        ("table_apron_y_pos", [0.0, sy - apron_half_t, apron_center_z], [sx, apron_half_t, apron_half_z]),
        ("table_apron_y_neg", [0.0, -sy + apron_half_t, apron_center_z], [sx, apron_half_t, apron_half_z]),
        ("table_apron_x_pos", [sx - apron_half_t, 0.0, apron_center_z], [apron_half_t, sy, apron_half_z]),
        ("table_apron_x_neg", [-sx + apron_half_t, 0.0, apron_center_z], [apron_half_t, sy, apron_half_z]),
    ]
    for name, pos, size in apron_specs:
        table_body.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos,
            size=size,
            rgba=list(TABLE_RGBA),
            friction=[1.0, 0.005, 0.001],
        )


def _apply_home_keyframe(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for name in ("panda_home", "home"):
        kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
        if kid >= 0:
            data.qpos[:] = model.key_qpos[kid, :]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            return


def _set_panda_home_controls(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Set position controls to current home qpos to avoid startup transients."""
    if model.nu <= 0:
        return

    for prefix in ("panda_", ""):
        joint_names = [f"{prefix}joint{i}" for i in range(1, 8)]
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
        if all(j >= 0 for j in jids):
            qids = [model.jnt_qposadr[j] for j in jids]
            break
    else:
        return

    for ai in range(model.nu):
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, ai) or ""
        for ji, jname in enumerate(joint_names):
            if jname in aname:
                data.ctrl[ai] = float(data.qpos[qids[ji]])
                break

    for ai in range(model.nu):
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, ai) or ""
        if "actuator8" in aname:
            data.ctrl[ai] = float(model.actuator_ctrlrange[ai, 1])
            break


def _stabilize_object_freejoint(model: mujoco.MjModel) -> None:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_joint")
    if jid < 0 or int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_FREE):
        return
    adr = int(model.jnt_dofadr[jid])
    model.dof_damping[adr : adr + 6] = np.asarray(OBJECT_FREEJOINT_DAMPING, dtype=np.float64)


def _boost_gripper_force(model: mujoco.MjModel) -> None:
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name and "actuator8" in name:
            model.actuator_gainprm[i][0] *= 50.0
            model.actuator_biasprm[i][1] *= 50.0
            model.actuator_biasprm[i][2] *= 5.0
            model.actuator_forcerange[i][0] = -500.0
            model.actuator_forcerange[i][1] = 500.0


def _finalize_model(model: mujoco.MjModel) -> mujoco.MjData:
    _boost_gripper_force(model)
    _stabilize_object_freejoint(model)
    data = mujoco.MjData(model)
    _apply_home_keyframe(model, data)
    _set_panda_home_controls(model, data)
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    return data


def build_scene(
    mesh_obj_path: str,
    mesh_scale: float,
    panda_xml_path: str | None = None,
    table_dims: tuple[float, float, float] = TABLE_DIMS,
    object_offset_xy: tuple[float, float] | None = None,
    object_dynamic: bool | None = None,
    object_mass_kg: float | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData, float, tuple[float, float], str]:
    """Compose scene with Panda + table + free mesh object."""
    if panda_xml_path is None:
        panda_xml_path = _get_panda_xml_path()

    cache_dir = _decompose_mesh(mesh_obj_path, mesh_scale)
    col_meshes = _find_collision_meshes(cache_dir)
    visual_path = os.path.join(cache_dir, "object.obj")

    obj_mesh = trimesh.load(visual_path, force="mesh")
    table_top_z = table_dims[2] * 0.5
    obj_z = float(table_top_z - obj_mesh.bounds[0, 2])
    ox, oy = object_offset_xy if object_offset_xy is not None else EVAL_OBJECT_OFFSET_XY

    panda_x = -(table_dims[0] * 0.5 - ROBOT_TABLE_EDGE_INSET_M)
    panda_z = table_top_z

    scene = mujoco.MjSpec()
    scene.option.timestep = 0.002
    scene.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    scene.option.gravity = [0.0, 0.0, -9.81]
    scene.option.noslip_iterations = 3
    scene.option.impratio = 10

    wb = scene.worldbody
    _add_menagerie_environment(scene, wb, floor_z=WORLD_FLOOR_Z)

    wb.add_camera(
        name="eval_cam",
        pos=[1.35, -1.35, 0.95],
        xyaxes=[0.72, 0.72, 0.0, -0.34, 0.34, 0.88],
    )

    _add_table(wb, table_dims)

    do_dyn = object_dynamic if object_dynamic is not None else OBJECT_DYNAMIC
    obj_body = wb.add_body(name="target_object", pos=[float(ox), float(oy), obj_z])
    if do_dyn:
        obj_body.add_freejoint(name="object_joint")
    obj_body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="object_visual",
        contype=0,
        conaffinity=0,
        group=2,
        density=0.0,
        rgba=[0.2, 0.6, 0.8, 1.0],
    )
    for i in range(len(col_meshes)):
        geom_kwargs = {
            "type": mujoco.mjtGeom.mjGEOM_MESH,
            "meshname": f"object_col_{i}",
            "group": 3,
            "rgba": [0.8, 0.3, 0.3, 1.0],
            "friction": list(OBJECT_FRICTION),
            "solimp": [0.95, 0.99, 0.001, 0.5, 2.0],
            "solref": [0.004, 1.0],
        }
        if object_mass_kg is not None and object_mass_kg > 0:
            geom_kwargs["mass"] = float(object_mass_kg) / max(len(col_meshes), 1)
        else:
            geom_kwargs["density"] = OBJECT_DENSITY
        obj_body.add_geom(
            **geom_kwargs
        )

    scene.add_mesh(name="object_visual", file=visual_path)
    for i, cf in enumerate(col_meshes):
        scene.add_mesh(name=f"object_col_{i}", file=os.path.join(cache_dir, cf))

    panda_spec = mujoco.MjSpec.from_file(panda_xml_path)
    attach_frame = wb.add_frame()
    attach_frame.pos = [panda_x, 0.0, panda_z]
    scene.attach(panda_spec, prefix="panda_", suffix="", frame=attach_frame)

    model = scene.compile()
    data = _finalize_model(model)
    return model, data, obj_z, (float(ox), float(oy)), visual_path


def build_simple_scene(
    panda_xml_path: str | None = None,
    box_size: tuple[float, float, float] = (0.03, 0.03, 0.05),
    object_offset_xy: tuple[float, float] | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData, float, tuple[float, float], str]:
    """Build a simple scene with a primitive free box object."""
    if panda_xml_path is None:
        panda_xml_path = _get_panda_xml_path()

    td0, td1, td2 = TABLE_DIMS
    table_top_z = td2 * 0.5
    obj_z = table_top_z + box_size[2]
    ox, oy = object_offset_xy if object_offset_xy is not None else EVAL_OBJECT_OFFSET_XY

    panda_x = -(td0 * 0.5 - ROBOT_TABLE_EDGE_INSET_M)
    panda_z = table_top_z

    scene = mujoco.MjSpec()
    scene.option.timestep = 0.002
    scene.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    scene.option.gravity = [0.0, 0.0, -9.81]
    scene.option.noslip_iterations = 3
    scene.option.impratio = 10

    wb = scene.worldbody
    _add_menagerie_environment(scene, wb, floor_z=WORLD_FLOOR_Z)

    wb.add_camera(
        name="eval_cam",
        pos=[1.35, -1.35, 1.1],
        xyaxes=[0.72, 0.72, 0.0, -0.34, 0.34, 0.88],
    )

    _add_table(wb, TABLE_DIMS)

    obj_body = wb.add_body(name="target_object", pos=[float(ox), float(oy), obj_z])
    if OBJECT_DYNAMIC:
        obj_body.add_freejoint(name="object_joint")
    obj_body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(box_size),
        mass=0.2,
        rgba=[0.8, 0.3, 0.3, 1.0],
        friction=list(OBJECT_FRICTION),
    )

    panda_spec = mujoco.MjSpec.from_file(panda_xml_path)
    attach_frame = wb.add_frame()
    attach_frame.pos = [panda_x, 0.0, panda_z]
    scene.attach(panda_spec, prefix="panda_", suffix="", frame=attach_frame)

    model = scene.compile()
    data = _finalize_model(model)
    return model, data, obj_z, (float(ox), float(oy)), ""
