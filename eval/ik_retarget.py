"""Direct top-1 IK retargeting and execution for MuJoCo Panda."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from eval.visualize_types import GraspSpec, PoseError, RetargetPlan, validate_pose_se3


PANDA_MAX_WIDTH_M = 0.08
PANDA_CTRL_MAX = 255.0


class ExecPhase(Enum):
    START_DELAY = "start_delay"
    APPROACH = "approach"
    REACH_GRASP = "reach_grasp"
    CLOSE = "close"
    POST_CLOSE_PAUSE = "post_close_pause"
    LIFT = "lift"
    DONE = "done"


@dataclass
class RetargetConfig:
    """Direct IK retargeting policy for one top-1 grasp execution path."""

    contact_to_wrist_m: float = 0.070
    approach_offset_m: float = 0.10
    approach_lift_m: float = 0.06
    lift_height_m: float = 0.15
    cartesian_waypoints: int = 8

    dt: float = 0.002
    start_delay_s: float = 5.0
    phase_duration_s: float = 0.9
    lift_phase_duration_s: float = 0.24
    approach_timeout_s: float = 8.0
    reach_timeout_s: float = 1.8
    close_min_time_s: float = 0.35
    close_timeout_s: float = 5.0
    post_close_pause_s: float = 1.0
    lift_timeout_s: float = 0.75
    success_lift_margin_m: float = 0.03
    success_grasp_pos_tol_m: float = 0.18
    success_grasp_rot_tol_deg: float = 35.0
    track_pos_tol_m: float = 1.2e-2
    track_rot_tol_deg: float = 10.0
    approach_pos_tol_m: float = 1.5e-2
    close_width_tol_m: float = 3.0e-3
    close_stall_delta_m: float = 5.0e-5
    close_stall_steps: int = 120
    close_to_zero_width: bool = True

    ik_max_steps: int = 800
    ik_pos_tol_m: float = 2.5e-3
    ik_rot_tol_rad: float = 0.06
    ik_seeds: int = 8
    ik_seed: int = 0
    ik_allow_best_effort: bool = True

    width_sim_scale: float = 1.12
    extra_opening_m: float = 0.014
    min_open_width_m: float = 0.048
    gripper_ctrl_slack: float = 32.0


@dataclass
class ExecResult:
    success: bool = False
    phase_reached: str = "none"
    pose_error_m: float = 0.0
    pose_error_deg: float = 0.0
    object_z_initial: float = 0.0
    object_z_final: float = 0.0
    lift_height_m: float = 0.0
    object_lift_m: float = 0.0
    error: str | None = None
    failure_reason: str | None = None


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _clip_width(width_m: float) -> float:
    return float(np.clip(width_m, 0.0, PANDA_MAX_WIDTH_M))


def width_to_ctrl(width_m: float) -> float:
    return float(_clip_width(width_m) / PANDA_MAX_WIDTH_M * PANDA_CTRL_MAX)


def _smoothstep(alpha: float) -> float:
    a = float(np.clip(alpha, 0.0, 1.0))
    return a * a * (3.0 - 2.0 * a)


def _interpolate_pose(T0: np.ndarray, T1: np.ndarray, alpha: float) -> np.ndarray:
    a = float(np.clip(alpha, 0.0, 1.0))
    P0 = np.asarray(T0[:3, 3], dtype=np.float64)
    P1 = np.asarray(T1[:3, 3], dtype=np.float64)
    R0 = np.asarray(T0[:3, :3], dtype=np.float64)
    R1 = np.asarray(T1[:3, :3], dtype=np.float64)

    dR = R1 @ R0.T
    dRV = R.from_matrix(dR).as_rotvec()
    Ri = R.from_rotvec(a * dRV).as_matrix() @ R0

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Ri
    T[:3, 3] = (1.0 - a) * P0 + a * P1
    return T


def contact_to_hand_pose(
    contact_pose: np.ndarray,
    width_m: float,
    *,
    contact_to_wrist_m: float = 0.066,
) -> np.ndarray:
    """Convert the top-1 CGT contact frame into the Panda hand frame."""
    T_contact = np.asarray(contact_pose, dtype=np.float64)
    contact = np.asarray(T_contact[:3, 3], dtype=np.float64)
    baseline = _normalize(np.asarray(T_contact[:3, 1], dtype=np.float64))
    approach = _normalize(np.asarray(T_contact[:3, 2], dtype=np.float64))

    baseline = _normalize(baseline - np.dot(baseline, approach) * approach)
    x_axis = _normalize(np.cross(baseline, approach))
    y_axis = baseline
    z_axis = approach

    T_hand = np.eye(4, dtype=np.float64)
    T_hand[:3, 0] = x_axis
    T_hand[:3, 1] = y_axis
    T_hand[:3, 2] = z_axis
    _ = _clip_width(width_m)
    T_hand[:3, 3] = contact - float(contact_to_wrist_m) * approach
    return validate_pose_se3(T_hand, "hand_pose")


def marker_to_hand_pose(marker_pose: np.ndarray) -> np.ndarray:
    """Convert an ACRONYM marker frame (x=baseline, z=approach) to Panda hand frame."""
    T_marker = validate_pose_se3(marker_pose, "marker_pose")
    Rz_neg90 = np.array(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    T_hand = T_marker.copy()
    T_hand[:3, :3] = T_marker[:3, :3] @ Rz_neg90
    return validate_pose_se3(T_hand, "hand_pose")


def build_retarget_plan_from_hand_pose(
    grasp_pose: np.ndarray,
    *,
    cfg: RetargetConfig | None = None,
) -> RetargetPlan:
    """Build simple approach and lift targets from an explicit Panda hand pose."""
    c = cfg or RetargetConfig()
    T_grasp = validate_pose_se3(grasp_pose, "grasp_pose")
    approach = _normalize(T_grasp[:3, 2])

    T_approach = T_grasp.copy()
    T_approach[:3, 3] = (
        T_grasp[:3, 3]
        - c.approach_offset_m * approach
        + np.array([0.0, 0.0, c.approach_lift_m], dtype=np.float64)
    )

    T_lift = T_grasp.copy()
    T_lift[:3, 3] = T_grasp[:3, 3] + np.array([0.0, 0.0, c.lift_height_m], dtype=np.float64)
    return RetargetPlan(approach_pose=T_approach, grasp_pose=T_grasp, lift_pose=T_lift)


def build_retarget_plan(
    grasp: GraspSpec,
    *,
    cfg: RetargetConfig | None = None,
) -> RetargetPlan:
    """Convert the selected top-1 contact-frame grasp into Panda hand targets."""
    c = cfg or RetargetConfig()
    T_grasp = contact_to_hand_pose(
        grasp.contact_pose_SE3,
        grasp.width_m,
        contact_to_wrist_m=c.contact_to_wrist_m,
    )
    return build_retarget_plan_from_hand_pose(T_grasp, cfg=c)


def compute_pose_error(target_pose: np.ndarray, live_pose: np.ndarray) -> PoseError:
    dp = float(np.linalg.norm(target_pose[:3, 3] - live_pose[:3, 3]))
    R_t = target_pose[:3, :3]
    R_l = live_pose[:3, :3]

    R_err = R_t @ R_l.T
    d0 = float(np.rad2deg(np.linalg.norm(R.from_matrix(R_err).as_rotvec())))

    R_t_alt = R_t @ np.diag([-1.0, -1.0, 1.0])
    R_err_alt = R_t_alt @ R_l.T
    d1 = float(np.rad2deg(np.linalg.norm(R.from_matrix(R_err_alt).as_rotvec())))
    ddeg = min(d0, d1)
    return PoseError(target_vs_live_translation_m=dp, target_vs_live_rotation_deg=ddeg)


def _discover_panda(model: mujoco.MjModel) -> dict:
    for prefix in ("panda_", ""):
        joint_names = [f"{prefix}joint{i}" for i in range(1, 8)]
        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
        if all(j >= 0 for j in joint_ids):
            break
    else:
        raise RuntimeError("Cannot find Panda arm joints")

    qpos_ids = [model.jnt_qposadr[j] for j in joint_ids]
    dof_ids = [model.jnt_dofadr[j] for j in joint_ids]
    q_lo = np.array([model.jnt_range[j, 0] for j in joint_ids], dtype=np.float64)
    q_hi = np.array([model.jnt_range[j, 1] for j in joint_ids], dtype=np.float64)

    act_ids: list[int] = []
    for joint_idx, joint_name in enumerate(joint_names, start=1):
        actuator_id = -1
        for ai in range(model.nu):
            actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, ai) or ""
            if joint_name in actuator_name or actuator_name.endswith(f"actuator{joint_idx}"):
                actuator_id = ai
                break
        if actuator_id < 0:
            raise RuntimeError(f"Cannot find actuator for {joint_name}")
        act_ids.append(actuator_id)

    gripper_act_id = -1
    for ai in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, ai) or ""
        if "actuator8" in actuator_name:
            gripper_act_id = ai
            break
    if gripper_act_id < 0:
        raise RuntimeError("Cannot find Panda gripper actuator")

    ee_body_id = -1
    for name in ("panda_hand", "hand"):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            ee_body_id = body_id
            break
    if ee_body_id < 0:
        raise RuntimeError("Cannot find Panda hand body")

    finger_joint_names = ("panda_finger_joint1", "panda_finger_joint2")
    finger_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in finger_joint_names]
    if not all(j >= 0 for j in finger_joint_ids):
        raise RuntimeError("Cannot find Panda finger joints")
    finger_qpos_ids = [int(model.jnt_qposadr[j]) for j in finger_joint_ids]

    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    return {
        "joint_ids": joint_ids,
        "qpos_ids": qpos_ids,
        "dof_ids": dof_ids,
        "q_lo": q_lo,
        "q_hi": q_hi,
        "act_ids": act_ids,
        "gripper_act_id": gripper_act_id,
        "ee_body_id": ee_body_id,
        "finger_qpos_ids": finger_qpos_ids,
        "object_body_id": object_body_id,
    }


def _home_arm_q(model: mujoco.MjModel, panda: dict) -> np.ndarray | None:
    for name in ("panda_home", "home"):
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
        if key_id >= 0:
            return np.array(
                [model.key_qpos[key_id, qid] for qid in panda["qpos_ids"]],
                dtype=np.float64,
            )
    return None


def _ik_pose_score(pos_err_m: float, rot_err_rad: float) -> float:
    # Couple translational and rotational miss into one comparable scalar.
    # 1 rad orientation miss is treated similarly to ~4 cm translation miss.
    return float(pos_err_m + 0.04 * rot_err_rad)


def _solve_ik_once(
    model: mujoco.MjModel,
    data_ref: mujoco.MjData,
    target_pose: np.ndarray,
    panda: dict,
    cfg: RetargetConfig,
    seed_q: np.ndarray,
) -> tuple[np.ndarray | None, float, float]:
    d = mujoco.MjData(model)
    d.qpos[:] = data_ref.qpos[:]
    d.qvel[:] = 0.0
    for i, qid in enumerate(panda["qpos_ids"]):
        d.qpos[qid] = float(seed_q[i])

    target_pos = target_pose[:3, 3]
    target_R = target_pose[:3, :3]
    ee_body_id = panda["ee_body_id"]

    best_q: np.ndarray | None = d.qpos[panda["qpos_ids"]].copy()
    best_pos_err = float("inf")
    best_rot_err = float("inf")
    best_score = float("inf")

    for _ in range(cfg.ik_max_steps):
        mujoco.mj_forward(model, d)
        cur_pos = np.asarray(d.xpos[ee_body_id], dtype=np.float64)
        cur_R = np.asarray(d.xmat[ee_body_id], dtype=np.float64).reshape(3, 3)

        err_pos = target_pos - cur_pos
        R_err = target_R @ cur_R.T
        rv = R.from_matrix(R_err).as_rotvec()
        target_R_alt = target_R @ np.diag([-1.0, -1.0, 1.0])
        rv_alt = R.from_matrix(target_R_alt @ cur_R.T).as_rotvec()
        err_rot = rv_alt if float(np.linalg.norm(rv_alt)) < float(np.linalg.norm(rv)) else rv
        pos_err = float(np.linalg.norm(err_pos))
        rot_err = float(np.linalg.norm(err_rot))
        score = _ik_pose_score(pos_err, rot_err)
        if score < best_score:
            best_score = score
            best_pos_err = pos_err
            best_rot_err = rot_err
            best_q = d.qpos[panda["qpos_ids"]].copy()

        if (
            pos_err < cfg.ik_pos_tol_m
            and rot_err < cfg.ik_rot_tol_rad
        ):
            return d.qpos[panda["qpos_ids"]].copy(), pos_err, rot_err

        err = np.concatenate([err_pos, err_rot])
        e = float(np.linalg.norm(err))
        if e > 0.04:
            err *= 0.04 / max(e, 1e-9)

        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacBody(model, d, jacp, jacr, ee_body_id)
        J = np.vstack([jacp[:, panda["dof_ids"]], jacr[:, panda["dof_ids"]]])

        lam = 1e-4 * (1.0 + 5.0 * e)
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), err)
        for i, qid in enumerate(panda["qpos_ids"]):
            d.qpos[qid] += dq[i]
        for i, joint_id in enumerate(panda["joint_ids"]):
            lo, hi = model.jnt_range[joint_id]
            d.qpos[panda["qpos_ids"][i]] = np.clip(d.qpos[panda["qpos_ids"][i]], lo, hi)

    return best_q, best_pos_err, best_rot_err


def solve_ik_dls(
    model: mujoco.MjModel,
    data_ref: mujoco.MjData,
    target_pose: np.ndarray,
    panda: dict,
    cfg: RetargetConfig,
    *,
    preferred_seed: np.ndarray | None = None,
) -> np.ndarray | None:
    """Solve one full-pose IK target with a small deterministic seed set."""
    seeds: list[np.ndarray] = []
    if preferred_seed is not None:
        seeds.append(np.asarray(preferred_seed, dtype=np.float64))
    seeds.extend(
        [
            np.asarray(data_ref.qpos[panda["qpos_ids"]], dtype=np.float64),
            0.5 * (panda["q_lo"] + panda["q_hi"]),
        ]
    )
    q_home = _home_arm_q(model, panda)
    if q_home is not None:
        seeds.append(q_home)

    rng = np.random.default_rng(cfg.ik_seed)
    for _ in range(max(cfg.ik_seeds - len(seeds), 0)):
        seeds.append(rng.uniform(panda["q_lo"], panda["q_hi"]))

    best_q: np.ndarray | None = None
    best_score = float("inf")
    seen: list[np.ndarray] = []
    for seed in seeds:
        if any(float(np.linalg.norm(seed - prev)) < 1e-4 for prev in seen):
            continue
        seen.append(seed)
        q, pos_err, rot_err = _solve_ik_once(model, data_ref, target_pose, panda, cfg, seed)
        if q is None:
            continue
        if pos_err < cfg.ik_pos_tol_m and rot_err < cfg.ik_rot_tol_rad:
            return q
        score = _ik_pose_score(pos_err, rot_err)
        if score < best_score:
            best_score = score
            best_q = q
    if cfg.ik_allow_best_effort:
        return best_q
    return None


class SimpleIKGraspExecutor:
    """Direct top-1 executor: exact IK per phase, no feasibility filtering."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        plan: RetargetPlan,
        grasp_width_m: float,
        cfg: RetargetConfig | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self.plan = plan
        self.cfg = cfg or RetargetConfig()
        self.cfg.dt = float(model.opt.timestep)
        self.panda = _discover_panda(model)

        self.phase = ExecPhase.START_DELAY if self.cfg.start_delay_s > 0.0 else ExecPhase.APPROACH
        self.phase_timer = 0.0
        self.result = ExecResult(phase_reached=self.phase.value)

        self._gripper_open_ctrl = self._compute_open_ctrl(grasp_width_m)
        self._gripper_target_width_m = (
            0.0 if self.cfg.close_to_zero_width else _clip_width(grasp_width_m)
        )
        self._gripper_close_ctrl = self._compute_close_ctrl(self._gripper_target_width_m)
        self._gripper_ctrl = self._gripper_open_ctrl

        self._active_target_pose: np.ndarray | None = None
        self._segment_start_pose: np.ndarray | None = None
        self._segment_start_q: np.ndarray | None = None
        self._segment_target_q: np.ndarray | None = None
        self._segment_waypoints_q: np.ndarray | None = None
        self._segment_duration_s: float = self.cfg.phase_duration_s
        self._hold_q: np.ndarray | None = None
        self._hand_z_at_grasp: float | None = None
        self._close_stall_steps = 0
        self._last_close_width_m: float | None = None

        object_body_id = self.panda["object_body_id"]
        if object_body_id >= 0:
            mujoco.mj_forward(self.model, self.data)
            self.result.object_z_initial = float(self.data.xpos[object_body_id][2])

        if self.phase == ExecPhase.APPROACH and not self._start_approach_segment():
            self.phase = ExecPhase.DONE

    def _start_approach_segment(self) -> bool:
        if not self._begin_segment(self.plan.approach_pose, self.cfg.phase_duration_s):
            self.result.error = "Failed to initialize approach IK"
            self.result.failure_reason = self.result.error
            return False
        return True

    def _compute_open_ctrl(self, width_m: float) -> float:
        w_cmd = max(
            _clip_width(width_m) * self.cfg.width_sim_scale + self.cfg.extra_opening_m,
            self.cfg.min_open_width_m,
        )
        return min(width_to_ctrl(w_cmd) + self.cfg.gripper_ctrl_slack, PANDA_CTRL_MAX)

    def _compute_close_ctrl(self, width_m: float) -> float:
        return width_to_ctrl(_clip_width(width_m))

    def _current_arm_q(self) -> np.ndarray:
        return np.asarray(self.data.qpos[self.panda["qpos_ids"]], dtype=np.float64).copy()

    def _apply_arm_ctrl(self, q: np.ndarray) -> None:
        for i, actuator_id in enumerate(self.panda["act_ids"]):
            self.data.ctrl[actuator_id] = float(q[i])
        self.data.ctrl[self.panda["gripper_act_id"]] = float(self._gripper_ctrl)

    def current_hand_pose(self) -> np.ndarray:
        body_id = self.panda["ee_body_id"]
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = np.asarray(self.data.xpos[body_id], dtype=np.float64)
        T[:3, :3] = np.asarray(self.data.xmat[body_id], dtype=np.float64).reshape(3, 3)
        return T

    def current_gripper_width_m(self) -> float:
        qids = self.panda["finger_qpos_ids"]
        return float(sum(float(self.data.qpos[qid]) for qid in qids))

    def current_target_pose(self) -> np.ndarray:
        if self._active_target_pose is not None:
            return self._active_target_pose
        return self.plan.grasp_pose

    def _pose_is_close(self, target_pose: np.ndarray) -> bool:
        err = compute_pose_error(target_pose, self.current_hand_pose())
        return (
            err.target_vs_live_translation_m <= self.cfg.track_pos_tol_m
            and err.target_vs_live_rotation_deg <= self.cfg.track_rot_tol_deg
        )

    def _approach_is_close(self, target_pose: np.ndarray) -> bool:
        live = self.current_hand_pose()
        dp = float(np.linalg.norm(target_pose[:3, 3] - live[:3, 3]))
        return dp <= self.cfg.approach_pos_tol_m

    def _begin_segment(self, target_pose: np.ndarray, duration_s: float) -> bool:
        q_start = self._current_arm_q()
        T_start = self.current_hand_pose()
        T_target = np.asarray(target_pose, dtype=np.float64).copy()

        n_waypoints = max(int(self.cfg.cartesian_waypoints), 1)
        waypoint_q: list[np.ndarray] = [q_start.copy()]
        q_seed = q_start.copy()
        for i in range(1, n_waypoints + 1):
            a = float(i) / float(n_waypoints)
            T_wp = _interpolate_pose(T_start, T_target, a)
            q_wp = solve_ik_dls(
                self.model,
                self.data,
                T_wp,
                self.panda,
                self.cfg,
                preferred_seed=q_seed,
            )
            if q_wp is None:
                return False
            waypoint_q.append(q_wp.copy())
            q_seed = q_wp.copy()

        self._active_target_pose = T_target
        self._segment_start_pose = T_start
        self._segment_start_q = q_start
        self._segment_waypoints_q = np.asarray(waypoint_q, dtype=np.float64)
        self._segment_target_q = waypoint_q[-1].copy()
        self._segment_duration_s = max(float(duration_s), 1e-3)
        self.phase_timer = 0.0
        return True

    def _segment_command_q(self) -> np.ndarray:
        if self._segment_start_q is None or self._segment_target_q is None:
            return self._current_arm_q()
        alpha = _smoothstep(self.phase_timer / self._segment_duration_s)
        if self._segment_waypoints_q is not None and len(self._segment_waypoints_q) >= 2:
            wps = self._segment_waypoints_q
            u = alpha * float(len(wps) - 1)
            i = min(int(np.floor(u)), len(wps) - 2)
            beta = u - float(i)
            return (1.0 - beta) * wps[i] + beta * wps[i + 1]
        return self._segment_start_q + alpha * (self._segment_target_q - self._segment_start_q)

    def _hold_step(self) -> None:
        if self._hold_q is None:
            self._hold_q = self._current_arm_q()
        self._apply_arm_ctrl(self._hold_q)

    def _close_complete(self) -> bool:
        width_m = self.current_gripper_width_m()
        if width_m <= self._gripper_target_width_m + self.cfg.close_width_tol_m:
            return True
        if self.phase_timer < self.cfg.close_min_time_s:
            self._last_close_width_m = width_m
            self._close_stall_steps = 0
            return False
        if self._last_close_width_m is not None:
            stalled = abs(width_m - self._last_close_width_m) <= self.cfg.close_stall_delta_m
            if stalled:
                self._close_stall_steps += 1
            else:
                self._close_stall_steps = 0
        self._last_close_width_m = width_m
        return self._close_stall_steps >= self.cfg.close_stall_steps

    def _finish(self) -> None:
        # Keep the grasp-pose error measured at REACH->CLOSE transition.
        # Recomputing at the end of lift would include intentional lift displacement.
        if self.result.pose_error_m <= 0.0 and self.result.pose_error_deg <= 0.0:
            err = compute_pose_error(self.plan.grasp_pose, self.current_hand_pose())
            self.result.pose_error_m = err.target_vs_live_translation_m
            self.result.pose_error_deg = err.target_vs_live_rotation_deg

        object_body_id = self.panda["object_body_id"]
        if object_body_id >= 0:
            self.result.object_z_final = float(self.data.xpos[object_body_id][2])
            self.result.object_lift_m = float(
                self.result.object_z_final - self.result.object_z_initial
            )
        hand_now = self.current_hand_pose()
        z0 = (
            self._hand_z_at_grasp
            if self._hand_z_at_grasp is not None
            else float(self.plan.grasp_pose[2, 3])
        )
        self.result.lift_height_m = float(hand_now[2, 3] - z0)
        if self.result.error is not None:
            self.result.success = False
            self.result.failure_reason = self.result.error
        elif object_body_id < 0:
            self.result.success = False
            self.result.failure_reason = "target_object_not_found"
        elif self.result.object_lift_m < self.cfg.success_lift_margin_m:
            self.result.success = False
            self.result.failure_reason = "object_not_lifted"
        else:
            self.result.success = True
            self.result.failure_reason = None
        self.result.phase_reached = ExecPhase.DONE.value

    def force_finish(self, error: str | None = None) -> None:
        """Finalize metrics even when the caller stops execution early."""
        if error is not None:
            self.result.error = error
            self.result.failure_reason = error
        self.phase = ExecPhase.DONE
        self._finish()

    def step(self) -> ExecPhase:
        if self.phase == ExecPhase.DONE:
            return self.phase

        if self.phase == ExecPhase.START_DELAY:
            self._gripper_ctrl = self._gripper_open_ctrl
            self._hold_step()
        elif self.phase == ExecPhase.APPROACH:
            self._gripper_ctrl = self._gripper_open_ctrl
            self._apply_arm_ctrl(self._segment_command_q())
        elif self.phase == ExecPhase.REACH_GRASP:
            self._gripper_ctrl = self._gripper_open_ctrl
            self._apply_arm_ctrl(self._segment_command_q())
        elif self.phase == ExecPhase.CLOSE:
            self._gripper_ctrl = self._gripper_close_ctrl
            self._hold_step()
        elif self.phase == ExecPhase.POST_CLOSE_PAUSE:
            self._gripper_ctrl = self._gripper_close_ctrl
            self._hold_step()
        elif self.phase == ExecPhase.LIFT:
            self._gripper_ctrl = self._gripper_close_ctrl
            self._apply_arm_ctrl(self._segment_command_q())

        mujoco.mj_step(self.model, self.data)
        self.phase_timer += self.cfg.dt

        if self.phase == ExecPhase.START_DELAY:
            if self.phase_timer >= self.cfg.start_delay_s:
                self.phase = ExecPhase.APPROACH
                self.result.phase_reached = self.phase.value
                if not self._start_approach_segment():
                    self.phase = ExecPhase.DONE

        elif self.phase == ExecPhase.APPROACH:
            if self._active_target_pose is not None and self._approach_is_close(self._active_target_pose):
                self.phase = ExecPhase.REACH_GRASP
                self.result.phase_reached = self.phase.value
                if not self._begin_segment(self.plan.grasp_pose, self.cfg.phase_duration_s):
                    self.result.error = "Failed to solve grasp IK"
                    self.result.failure_reason = self.result.error
                    self.phase = ExecPhase.DONE
            elif self.phase_timer >= self.cfg.approach_timeout_s:
                self.phase = ExecPhase.REACH_GRASP
                self.result.phase_reached = self.phase.value
                if not self._begin_segment(self.plan.grasp_pose, self.cfg.phase_duration_s):
                    self.result.error = "Failed to solve grasp IK"
                    self.result.failure_reason = self.result.error
                    self.phase = ExecPhase.DONE

        elif self.phase == ExecPhase.REACH_GRASP:
            if self._active_target_pose is not None and self._pose_is_close(self._active_target_pose):
                err = compute_pose_error(self.plan.grasp_pose, self.current_hand_pose())
                self.result.pose_error_m = err.target_vs_live_translation_m
                self.result.pose_error_deg = err.target_vs_live_rotation_deg
                self._hand_z_at_grasp = float(self.current_hand_pose()[2, 3])
                self._hold_q = self._current_arm_q()
                self._close_stall_steps = 0
                self._last_close_width_m = self.current_gripper_width_m()
                self.phase = ExecPhase.CLOSE
                self.phase_timer = 0.0
                self.result.phase_reached = self.phase.value
            elif self.phase_timer >= self.cfg.reach_timeout_s:
                err = compute_pose_error(self.plan.grasp_pose, self.current_hand_pose())
                self.result.pose_error_m = err.target_vs_live_translation_m
                self.result.pose_error_deg = err.target_vs_live_rotation_deg
                self._hand_z_at_grasp = float(self.current_hand_pose()[2, 3])
                self._hold_q = self._current_arm_q()
                self._close_stall_steps = 0
                self._last_close_width_m = self.current_gripper_width_m()
                self.phase = ExecPhase.CLOSE
                self.phase_timer = 0.0
                self.result.phase_reached = self.phase.value

        elif self.phase == ExecPhase.CLOSE:
            if self._close_complete():
                self.phase = ExecPhase.POST_CLOSE_PAUSE
                self.result.phase_reached = self.phase.value
                self._hold_q = self._current_arm_q()
                self.phase_timer = 0.0
            elif self.phase_timer >= self.cfg.close_timeout_s:
                # Max-effort fallback: proceed after timeout even if not at zero width.
                self.phase = ExecPhase.POST_CLOSE_PAUSE
                self.result.phase_reached = self.phase.value
                self._hold_q = self._current_arm_q()
                self.phase_timer = 0.0

        elif self.phase == ExecPhase.POST_CLOSE_PAUSE:
            if self.phase_timer >= self.cfg.post_close_pause_s:
                self.phase = ExecPhase.LIFT
                self.result.phase_reached = self.phase.value
                if not self._begin_segment(self.plan.lift_pose, self.cfg.lift_phase_duration_s):
                    self.result.error = "Failed to solve lift IK"
                    self.result.failure_reason = self.result.error
                    self.phase = ExecPhase.DONE

        elif self.phase == ExecPhase.LIFT:
            if self._active_target_pose is not None and self._pose_is_close(self._active_target_pose):
                self.phase = ExecPhase.DONE
                self._finish()
            elif self.phase_timer >= self.cfg.lift_timeout_s:
                self.phase = ExecPhase.DONE
                self._finish()

        if self.phase == ExecPhase.DONE and self.result.phase_reached != ExecPhase.DONE.value:
            err = compute_pose_error(self.plan.grasp_pose, self.current_hand_pose())
            self.result.pose_error_m = err.target_vs_live_translation_m
            self.result.pose_error_deg = err.target_vs_live_rotation_deg
        if self.phase == ExecPhase.DONE and self.result.error is not None and not self.result.failure_reason:
            self.result.failure_reason = self.result.error
        return self.phase
