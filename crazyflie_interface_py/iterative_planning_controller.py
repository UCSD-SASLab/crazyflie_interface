#!/usr/bin/env python3
from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import rclpy
import rowan
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from example_interfaces.msg import Float32MultiArray
from std_msgs.msg import Bool, ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

from crazyflie_interface_py.template_controller import TemplateController

MODEL_ROOT = pathlib.Path(get_package_share_directory("crazyflie_interface")) / "models"

try:
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import jax_dataclasses as jdc
    from rraa_rl.collector import Collector
    from rraa_rl.load_ckpt import load_ckpt
    from rraa_rl.rollout_utils import extract_rollouts_eval
    from scipy.interpolate import CubicSpline, make_smoothing_spline
except ImportError as exc:  # pragma: no cover - exercised only in missing-dependency environments
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class IterativePlanningControllerBase(TemplateController):
    """Shared plumbing for iterative rollout controllers in crazyflie_interface.

    These controllers continuously:
    1. read the current robot state from `cf_interface/state`
    2. overwrite the relevant positions in an rraa-rl environment state
    3. run a short simulated rollout from that observed state
    4. smooth the short horizon with splines
    5. publish the next full-state setpoint directly to `cf_interface/control_full_state`

    The controller can also ignore live robot positions and instead advance an internal
    ghost copy of the controlled robots, similar to 20d_drone_controller_ghost.py.
    """

    controller_rate_hz: float = 10.0
    rollout_horizon_steps: int = 20
    n_smooth_pts_per_step: int = 2
    planning_seed: int = 1
    flight_height_m: float = 0.6
    plan_eval_dt_s: float | None = None
    simulate_controlled_positions: bool = False
    smooth_coef: float = 1e-4
    dog_as_flying_agent: bool = True
    publish_dog_plan: bool = False
    expected_robot_count: int = 0

    run_path: str = ""
    agent_radius_real_m: float = 0.11
    center_shift_m: np.ndarray = np.zeros(2)
    vel_max_real_mps: float = 1.0

    def __init__(self, node_name: str):
        self.control_publisher_topic = "cf_interface/control_full_state"
        super().__init__(
            node_name,
            controller_rate=self.controller_rate_hz,
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                "Missing Python dependencies for iterative rollout controller. "
                "Install rraa-rl and its dependencies, including jax."
            ) from _IMPORT_ERROR

        self._ros_parameters = self._param_to_dict(self._parameters)
        robots = self._ros_parameters.get("robots", {})
        self.nbr_robots = len(robots)
        robot_names = list(robots.keys())
        if self.nbr_robots == 0 and self.expected_robot_count > 0:
            self.nbr_robots = self.expected_robot_count
            robot_names = [f"sim_cf_{idx:02d}" for idx in range(self.nbr_robots)]
            self.get_logger().warning(
                "No 'robots' ROS parameter provided; falling back to expected sim robot count "
                f"{self.expected_robot_count}."
            )
        self.get_logger().info(f"Robots: {robot_names}")
        self.get_logger().info(f"Number of robots: {self.nbr_robots}")

        self.create_subscription(Bool, "cf_interface/flight_status", self.flight_status_callback, 1)
        self.in_flight = False

        self._loaded = False
        self.iteration = 0
        self.last_yaws = None
        self.ghost_positions_m = None
        self.ghost_velocities_mps = None
        self.path_pubs = []
        self.marker_pub = self.create_publisher(MarkerArray, "controller_plan_markers", 1)
        self.dog_plan_pubs = []
        self.dog_path_pubs = []

        self.start_controller()

    def _param_to_dict(self, param_ros):
        tree = {}
        for item in param_ros:
            t = tree
            for part in item.split("."):
                if part == item.split(".")[-1]:
                    t = t.setdefault(part, param_ros[item].value)
                else:
                    t = t.setdefault(part, {})
        return tree

    def flight_status_callback(self, msg: Bool):
        self.in_flight = bool(msg.data)

    def _ensure_loaded(self):
        if self._loaded:
            return

        run_path = pathlib.Path(self.run_path)
        self.get_logger().info(f"Loading rollout checkpoint from {run_path}")
        _run, self.agent, self.env, _cfg_dict = load_ckpt(run_path, step=None)
        self.collector = Collector.create(
            key=jr.PRNGKey(1234),
            env=self.env,
            cfg=Collector.Cfg(n_envs=1, auto_reset=False, ignore_trunc=True),
        )
        self.template_state = self.env.get_eval_states(1, root_only=True)

        cfg = self.env.base.cfg
        self.sim_to_m = float(self.agent_radius_real_m / cfg.agent_radius)
        vel_max_sim_per_simtime = float(max(cfg.vel_maxs))
        vel_max_m_per_simtime = vel_max_sim_per_simtime * self.sim_to_m
        self.s_per_simtime = vel_max_m_per_simtime / self.vel_max_real_mps
        self.dt_s = float(cfg.dt * self.s_per_simtime)
        if self.plan_eval_dt_s is None:
            self.plan_eval_dt_s = 1.0 / self.controller_rate

        self._init_shadow_from_template()
        controlled = self._controlled_slots()
        if len(controlled) != self.nbr_robots:
            raise ValueError(
                f"Controller maps {len(controlled)} planned trajectories to robots, "
                f"but crazyflie_interface has {self.nbr_robots} enabled robots."
            )

        self.last_yaws = np.zeros(self.nbr_robots, dtype=float)
        self.path_pubs = [
            self.create_publisher(Path, f"controller_plan_path_{ii:02d}", 1) for ii in range(self.nbr_robots)
        ]
        dog_slots = self._dog_slots()
        self.dog_plan_pubs = [
            self.create_publisher(Float32MultiArray, f"dog_plan_{ii:02d}", 1) for ii in range(len(dog_slots))
        ]
        self.dog_path_pubs = [
            self.create_publisher(Path, f"dog_plan_path_{ii:02d}", 1) for ii in range(len(dog_slots))
        ]
        self._loaded = True
        self.get_logger().info(f"Loaded rollout checkpoint from {run_path}")

    def _reshape_states(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=float).reshape(self.nbr_robots, -1)

    def _current_positions_velocities(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        states = self._reshape_states(state)
        live_pos = states[:, 0:3]
        live_vel = states[:, 3:6]

        if self.ghost_positions_m is None:
            self.ghost_positions_m = live_pos.copy()
            self.ghost_velocities_mps = live_vel.copy()

        if self.simulate_controlled_positions:
            pos = self.ghost_positions_m.copy()
            vel = self.ghost_velocities_mps.copy()
        else:
            pos = live_pos
            vel = live_vel

        return pos, vel

    def _m_to_sim_xy(self, pos_m_xy: np.ndarray) -> np.ndarray:
        return (pos_m_xy - self.center_shift_m) / self.sim_to_m

    def _sim_to_m_xy(self, pos_sim_xy: np.ndarray) -> np.ndarray:
        return pos_sim_xy * self.sim_to_m + self.center_shift_m

    def _rollout_once(self, plan_state: Any):
        Tb_rollout, _info_collect = self.agent.collect_eval_with_states(
            self.collector,
            plan_state,
            self.rollout_horizon_steps,
            temporal_transitions=True,
        )
        Tb_rollout = jax.device_get(Tb_rollout)
        bT_rollout = Tb_rollout.switch01()
        b_trajs = extract_rollouts_eval(bT_rollout)
        if len(b_trajs) == 0:
            raise RuntimeError("No rollout returned from collect_eval_with_states.")
        return b_trajs[0]

    def _smooth_path(self, T_time: np.ndarray, T_pos: np.ndarray):
        if len(T_time) < 2:
            raise ValueError("Need at least two knot points for smoothing.")

        T_fine = np.linspace(
            0.0,
            float(T_time[-1]),
            num=(len(T_time) - 1) * self.n_smooth_pts_per_step + 1,
        )
        T_pos_linterp = np.stack(
            [np.interp(T_fine, T_time, T_pos[:, ii]) for ii in range(T_pos.shape[1])],
            axis=1,
        )
        if len(T_fine) <= 3:
            spline = CubicSpline(T_fine, T_pos_linterp, axis=0)
            return T_fine, spline

        if len(T_fine) > 5:
            lam = self.smooth_coef * len(T_time)
            spl_x = make_smoothing_spline(T_fine, T_pos_linterp[:, 0], lam=lam)
            spl_y = make_smoothing_spline(T_fine, T_pos_linterp[:, 1], lam=lam)
            T_pos_smooth = np.stack([spl_x(T_fine), spl_y(T_fine)], axis=1)
        else:
            T_pos_smooth = T_pos_linterp

        spline = CubicSpline(T_fine, T_pos_smooth, axis=0)
        return T_fine, spline

    def _make_fullstate_cmd(
        self,
        robot_idx: int,
        pos_xy_m: np.ndarray,
        vel_xy_mps: np.ndarray,
        acc_xy_mps2: np.ndarray,
        yaw_override: float | None = None,
    ) -> np.ndarray:
        cmd = np.zeros(16, dtype=float)
        cmd[0:3] = np.array([pos_xy_m[0], pos_xy_m[1], self.flight_height_m])
        cmd[3:6] = np.array([vel_xy_mps[0], vel_xy_mps[1], 0.0])
        cmd[6:10] = np.array([0.0, 0.0, 0.0, 1.0])
        cmd[10:13] = np.array([0.0, 0.0, 0.0])
        cmd[13:16] = np.array([acc_xy_mps2[0], acc_xy_mps2[1], 0.0])

        speed = float(np.linalg.norm(vel_xy_mps))
        yaw = yaw_override
        if yaw is None and speed > 1e-4:
            yaw = float(np.arctan2(vel_xy_mps[1], vel_xy_mps[0]))
        if yaw is not None:
            self.last_yaws[robot_idx] = yaw
            quat_wxyz = rowan.from_euler(0.0, 0.0, yaw, "xyz")
            cmd[6:10] = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        return cmd

    def _hover_cmds(self, state: np.ndarray) -> np.ndarray:
        states = self._reshape_states(state)
        u = np.zeros((self.nbr_robots, 16), dtype=float)
        for ii in range(self.nbr_robots):
            u[ii, 0:3] = np.array([states[ii, 0], states[ii, 1], self.flight_height_m])
            u[ii, 6:10] = np.array([0.0, 0.0, 0.0, 1.0])
        return u

    def __call__(self, state):
        self._ensure_loaded()
        if not self.in_flight:
            return self._hover_cmds(state).flatten()

        controlled_pos_m, controlled_vel_mps = self._current_positions_velocities(state)
        self._apply_controlled_observations(controlled_pos_m, controlled_vel_mps)

        plan_state = self._make_plan_state()
        if self.iteration == 0:
            self.get_logger().info("Running first rollout from observed state")
        traj = self._rollout_once(plan_state)
        if self.iteration == 0:
            self.get_logger().info("Building first command set from rollout")
        cmd, viz_trajs, dog_plans = self._build_commands_from_traj(traj)
        self._publish_rviz(viz_trajs, cmd)
        self._publish_dog_plans(dog_plans)
        self._advance_shadow_from_traj(traj)

        if self.simulate_controlled_positions:
            self.ghost_positions_m = cmd[:, 0:3].copy()
            self.ghost_velocities_mps = cmd[:, 3:6].copy()

        self.iteration += 1
        if self.iteration % 20 == 0:
            xy = cmd[:, 0:2]
            self.get_logger().info(f"Iterative replanning iteration {self.iteration}, targets={xy}")
        return cmd.flatten()

    def _publish_rviz(self, viz_trajs: list[np.ndarray], cmd: np.ndarray):
        stamp = self.get_clock().now().to_msg()
        header = Header(stamp=stamp, frame_id="world")
        marker_array = MarkerArray()

        for robot_idx, T_pos in enumerate(viz_trajs):
            path_msg = Path(header=header)
            poses = []
            for pos in T_pos:
                pose = PoseStamped()
                pose.header = header
                pose.pose.position.x = float(pos[0])
                pose.pose.position.y = float(pos[1])
                pose.pose.position.z = float(self.flight_height_m)
                pose.pose.orientation.w = 1.0
                poses.append(pose)
            path_msg.poses = poses
            self.path_pubs[robot_idx].publish(path_msg)

            marker = Marker()
            marker.header = header
            marker.ns = "controller_cmd"
            marker.id = robot_idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(cmd[robot_idx, 0])
            marker.pose.position.y = float(cmd[robot_idx, 1])
            marker.pose.position.z = float(cmd[robot_idx, 2])
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.10
            marker.scale.y = 0.10
            marker.scale.z = 0.10
            marker.color = ColorRGBA(r=0.15, g=0.80, b=0.25, a=0.90)
            marker_array.markers.append(marker)

            if self.simulate_controlled_positions and self.ghost_positions_m is not None:
                ghost_marker = Marker()
                ghost_marker.header = header
                ghost_marker.ns = "controller_ghost"
                ghost_marker.id = robot_idx + self.nbr_robots
                ghost_marker.type = Marker.SPHERE
                ghost_marker.action = Marker.ADD
                ghost_marker.pose.position.x = float(self.ghost_positions_m[robot_idx, 0])
                ghost_marker.pose.position.y = float(self.ghost_positions_m[robot_idx, 1])
                ghost_marker.pose.position.z = float(self.ghost_positions_m[robot_idx, 2])
                ghost_marker.pose.orientation.w = 1.0
                ghost_marker.scale.x = 0.07
                ghost_marker.scale.y = 0.07
                ghost_marker.scale.z = 0.07
                ghost_marker.color = ColorRGBA(r=0.90, g=0.30, b=0.20, a=0.80)
                marker_array.markers.append(ghost_marker)

        self.marker_pub.publish(marker_array)

    def _publish_dog_plans(self, dog_plans: list[dict[str, np.ndarray]]):
        if not self.publish_dog_plan:
            return

        stamp = self.get_clock().now().to_msg()
        header = Header(stamp=stamp, frame_id="world")
        for ii, dog_plan in enumerate(dog_plans):
            T_pos = dog_plan["T_pos"]
            T_yaw = dog_plan["T_yaw"]
            delta_t = float(dog_plan["delta_t"])
            n_steps = int(len(T_pos))

            msg = Float32MultiArray()
            flat = np.concatenate(
                [
                    np.array([delta_t, float(n_steps)], dtype=float),
                    np.column_stack([T_pos, T_yaw[:, None]]).reshape(-1),
                ]
            )
            msg.data = flat.tolist()
            self.dog_plan_pubs[ii].publish(msg)

            path_msg = Path(header=header)
            poses = []
            for pos, yaw in zip(T_pos, T_yaw):
                quat_wxyz = rowan.from_euler(0.0, 0.0, float(yaw), "xyz")
                pose = PoseStamped()
                pose.header = header
                pose.pose.position.x = float(pos[0])
                pose.pose.position.y = float(pos[1])
                pose.pose.position.z = float(self.flight_height_m)
                pose.pose.orientation.x = float(quat_wxyz[1])
                pose.pose.orientation.y = float(quat_wxyz[2])
                pose.pose.orientation.z = float(quat_wxyz[3])
                pose.pose.orientation.w = float(quat_wxyz[0])
                poses.append(pose)
            path_msg.poses = poses
            self.dog_path_pubs[ii].publish(path_msg)

    def _controlled_slots(self) -> list[tuple[str, int]]:
        raise NotImplementedError

    def _init_shadow_from_template(self):
        raise NotImplementedError

    def _apply_controlled_observations(self, positions_m: np.ndarray, velocities_mps: np.ndarray):
        raise NotImplementedError

    def _make_plan_state(self):
        raise NotImplementedError

    def _build_commands_from_traj(self, traj) -> tuple[np.ndarray, list[np.ndarray]]:
        raise NotImplementedError

    def _advance_shadow_from_traj(self, traj):
        raise NotImplementedError

    def _dog_slots(self) -> list[int]:
        return []

    @staticmethod
    def _yaw_from_tangent(T_pos: np.ndarray) -> np.ndarray:
        if len(T_pos) < 2:
            return np.zeros((len(T_pos),), dtype=float)
        T_vel = np.gradient(T_pos, axis=0)
        return np.arctan2(T_vel[:, 1], T_vel[:, 0])


class HerdController(IterativePlanningControllerBase):
    run_path = str(MODEL_ROOT / "20260128-134905_total_hardware")
    expected_robot_count = 5
    agent_radius_real_m = 0.11
    center_shift_m = np.array([0.24, 0.01], dtype=float)
    vel_max_real_mps = 1.1
    smooth_coef = 1e-4
    use_herd_agents = True
    cf_herder_idx: list[int] = [0, 1]
    dog_herder_idx: list[int] = []
    dog_face_nearest_herd: bool = True

    def __init__(self, node_name: str = "herd_controller"):
        super().__init__(node_name=node_name)

    def _controlled_slots(self) -> list[tuple[str, int]]:
        slots: list[tuple[str, int]] = []
        if self.use_herd_agents:
            slots.extend(("herd", idx) for idx in range(self.env.base.cfg.n_herd))
        slots.extend(("herder", idx) for idx in self.cf_herder_idx)
        if self.dog_as_flying_agent:
            slots.extend(("dog", idx) for idx in self.dog_herder_idx)
        return slots

    def _dog_slots(self) -> list[int]:
        return list(self.dog_herder_idx)

    def _init_shadow_from_template(self):
        base = self.template_state.base
        self.shadow_temporal_node_idx = int(np.asarray(self.template_state.temporal_node_idx)[0])
        self.shadow_herd_state = np.array(np.asarray(base.herd_state)[0], copy=True)
        self.shadow_herder_state = np.array(np.asarray(base.herder_state)[0], copy=True)

    def _apply_controlled_observations(self, positions_m: np.ndarray, velocities_mps: np.ndarray):
        for robot_idx, (kind, idx) in enumerate(self._controlled_slots()):
            pos_sim = self._m_to_sim_xy(positions_m[robot_idx, :2])
            vel_sim = velocities_mps[robot_idx, :2] * self.s_per_simtime / self.sim_to_m
            if kind == "herd":
                self.shadow_herd_state[idx, :2] = pos_sim
            elif kind in ("herder", "dog"):
                self.shadow_herder_state[idx, :2] = pos_sim
                self.shadow_herder_state[idx, 2:4] = vel_sim
            else:
                raise ValueError(f"Unknown herd controlled kind: {kind}")

    def _make_plan_state(self):
        base_state = jdc.replace(
            self.template_state.base,
            herd_state=jnp.asarray(self.shadow_herd_state[None, ...]),
            herder_state=jnp.asarray(self.shadow_herder_state[None, ...]),
        )
        return jdc.replace(
            self.template_state,
            temporal_node_idx=jnp.asarray([self.shadow_temporal_node_idx], dtype=jnp.int32),
            base=base_state,
        )

    def _build_commands_from_traj(self, traj) -> tuple[np.ndarray, list[np.ndarray], list[dict[str, np.ndarray]]]:
        future_herd = np.asarray(traj.state_next.base.herd_state)
        future_herder = np.asarray(traj.state_next.base.herder_state[:, :, :2])

        curr_herd = self.shadow_herd_state[:, :2]
        curr_herder = self.shadow_herder_state[:, :2]
        herd_knots = np.concatenate([curr_herd[None, ...], future_herd], axis=0)
        herder_knots = np.concatenate([curr_herder[None, ...], future_herder], axis=0)

        T_time_s = np.arange(herd_knots.shape[0], dtype=float) * self.dt_s
        t_eval = min(self.plan_eval_dt_s, T_time_s[-1])

        cmd = np.zeros((self.nbr_robots, 16), dtype=float)
        viz_trajs: list[np.ndarray] = []
        smooth_herd_paths = []
        for herd_idx in range(herd_knots.shape[1]):
            T_pos_m = self._sim_to_m_xy(herd_knots[:, herd_idx])
            T_fine, spline = self._smooth_path(T_time_s, T_pos_m)
            smooth_herd_paths.append(np.asarray(spline(T_fine)))

        dog_plans: list[dict[str, np.ndarray]] = []
        for robot_idx, (kind, idx) in enumerate(self._controlled_slots()):
            T_pos_sim = herd_knots[:, idx] if kind == "herd" else herder_knots[:, idx]
            T_pos_m = self._sim_to_m_xy(T_pos_sim)
            T_fine, spline = self._smooth_path(T_time_s, T_pos_m)
            T_pos_smooth = np.asarray(spline(T_fine))
            idx_eval = int(np.clip(np.searchsorted(T_fine, t_eval, side="left"), 0, len(T_fine) - 1))
            pos_xy = T_pos_smooth[idx_eval]
            vel_xy = np.asarray(spline.derivative(1)(t_eval))
            acc_xy = np.asarray(spline.derivative(2)(t_eval))
            yaw_override = None
            if kind == "dog":
                if self.dog_face_nearest_herd and len(smooth_herd_paths) > 0:
                    herd_stack = np.stack(smooth_herd_paths, axis=1)
                    T_dist = np.linalg.norm(herd_stack - T_pos_smooth[:, None, :], axis=-1)
                    T_closest = np.argmin(T_dist, axis=1)
                    T_target = herd_stack[np.arange(len(T_fine)), T_closest]
                    T_diff = T_target - T_pos_smooth
                    T_yaw = np.arctan2(T_diff[:, 1], T_diff[:, 0])
                else:
                    T_yaw = self._yaw_from_tangent(T_pos_smooth)
                yaw_override = float(T_yaw[idx_eval])
                dog_plans.append({"T_pos": T_pos_smooth, "T_yaw": T_yaw, "delta_t": float(T_fine[1] - T_fine[0])})
            cmd[robot_idx] = self._make_fullstate_cmd(robot_idx, pos_xy, vel_xy, acc_xy, yaw_override=yaw_override)
            viz_trajs.append(T_pos_smooth)

        if self.publish_dog_plan and not self.dog_as_flying_agent:
            for herder_idx in self.dog_herder_idx:
                T_pos_m = self._sim_to_m_xy(herder_knots[:, herder_idx])
                T_fine, spline = self._smooth_path(T_time_s, T_pos_m)
                T_pos_smooth = np.asarray(spline(T_fine))
                if self.dog_face_nearest_herd and len(smooth_herd_paths) > 0:
                    herd_stack = np.stack(smooth_herd_paths, axis=1)
                    T_dist = np.linalg.norm(herd_stack - T_pos_smooth[:, None, :], axis=-1)
                    T_closest = np.argmin(T_dist, axis=1)
                    T_target = herd_stack[np.arange(len(T_fine)), T_closest]
                    T_diff = T_target - T_pos_smooth
                    T_yaw = np.arctan2(T_diff[:, 1], T_diff[:, 0])
                else:
                    T_yaw = self._yaw_from_tangent(T_pos_smooth)
                dog_plans.append({"T_pos": T_pos_smooth, "T_yaw": T_yaw, "delta_t": float(T_fine[1] - T_fine[0])})

        return cmd, viz_trajs, dog_plans

    def _advance_shadow_from_traj(self, traj):
        future_herd = np.asarray(traj.state_next.base.herd_state)
        future_herder = np.asarray(traj.state_next.base.herder_state)
        if len(future_herd) > 0:
            self.shadow_herd_state = np.array(future_herd[0], copy=True)
        if len(future_herder) > 0:
            self.shadow_herder_state = np.array(future_herder[0], copy=True)
        temporal_idx = np.asarray(getattr(traj, "temporal_node_idx", np.array([self.shadow_temporal_node_idx])))
        if temporal_idx.size > 0:
            use_idx = min(1, temporal_idx.size - 1)
            self.shadow_temporal_node_idx = int(temporal_idx[use_idx])


class DeliveryController(IterativePlanningControllerBase):
    run_path = str(MODEL_ROOT / "20260129-154526_baker_reset-v4")
    expected_robot_count = 3
    agent_radius_real_m = 0.112
    center_shift_m = np.array([0.24, 0.01], dtype=float) + np.array([0.18, 0.10], dtype=float)
    vel_max_real_mps = 0.6
    smooth_coef = 2e-5
    cf_herder_idx: list[int] = [0, 1]
    dog_herder_idx: list[int] = []

    def __init__(self, node_name: str = "delivery_controller"):
        super().__init__(node_name=node_name)

    def _controlled_slots(self) -> list[tuple[str, int]]:
        slots = [("herder", idx) for idx in self.cf_herder_idx]
        if self.dog_as_flying_agent:
            slots.extend(("dog", idx) for idx in self.dog_herder_idx)
        return slots

    def _dog_slots(self) -> list[int]:
        return list(self.dog_herder_idx)

    def _init_shadow_from_template(self):
        base = self.template_state.base
        self.shadow_temporal_node_idx = int(np.asarray(self.template_state.temporal_node_idx)[0])
        self.shadow_herd_state = np.array(np.asarray(base.herd_state)[0], copy=True)
        self.shadow_herder_state = np.array(np.asarray(base.herder_state)[0], copy=True)
        self.shadow_centers = np.array(np.asarray(base.centers)[0], copy=True)

    def _apply_controlled_observations(self, positions_m: np.ndarray, velocities_mps: np.ndarray):
        for robot_idx, (_kind, idx) in enumerate(self._controlled_slots()):
            pos_sim = self._m_to_sim_xy(positions_m[robot_idx, :2])
            vel_sim = velocities_mps[robot_idx, :2] * self.s_per_simtime / self.sim_to_m
            self.shadow_herder_state[idx, :2] = pos_sim
            self.shadow_herder_state[idx, 2:4] = vel_sim

    def _make_plan_state(self):
        base_state = jdc.replace(
            self.template_state.base,
            herd_state=jnp.asarray(self.shadow_herd_state[None, ...]),
            herder_state=jnp.asarray(self.shadow_herder_state[None, ...]),
            centers=jnp.asarray(self.shadow_centers[None, ...]),
        )
        return jdc.replace(
            self.template_state,
            temporal_node_idx=jnp.asarray([self.shadow_temporal_node_idx], dtype=jnp.int32),
            base=base_state,
        )

    def _build_commands_from_traj(self, traj) -> tuple[np.ndarray, list[np.ndarray], list[dict[str, np.ndarray]]]:
        future_herder = np.asarray(traj.state_next.base.herder_state[:, :, :2])
        curr_herder = self.shadow_herder_state[:, :2]
        herder_knots = np.concatenate([curr_herder[None, ...], future_herder], axis=0)

        T_time_s = np.arange(herder_knots.shape[0], dtype=float) * self.dt_s
        t_eval = min(self.plan_eval_dt_s, T_time_s[-1])

        cmd = np.zeros((self.nbr_robots, 16), dtype=float)
        viz_trajs: list[np.ndarray] = []
        dog_plans: list[dict[str, np.ndarray]] = []
        for robot_idx, (_kind, idx) in enumerate(self._controlled_slots()):
            T_pos_m = self._sim_to_m_xy(herder_knots[:, idx])
            T_fine, spline = self._smooth_path(T_time_s, T_pos_m)
            T_pos_smooth = np.asarray(spline(T_fine))
            idx_eval = int(np.clip(np.searchsorted(T_fine, t_eval, side="left"), 0, len(T_fine) - 1))
            pos_xy = T_pos_smooth[idx_eval]
            vel_xy = np.asarray(spline.derivative(1)(t_eval))
            acc_xy = np.asarray(spline.derivative(2)(t_eval))
            yaw_override = None
            if self._controlled_slots()[robot_idx][0] == "dog":
                T_yaw = self._yaw_from_tangent(T_pos_smooth)
                yaw_override = float(T_yaw[idx_eval])
                dog_plans.append({"T_pos": T_pos_smooth, "T_yaw": T_yaw, "delta_t": float(T_fine[1] - T_fine[0])})
            cmd[robot_idx] = self._make_fullstate_cmd(robot_idx, pos_xy, vel_xy, acc_xy, yaw_override=yaw_override)
            viz_trajs.append(T_pos_smooth)

        if self.publish_dog_plan and not self.dog_as_flying_agent:
            for herder_idx in self.dog_herder_idx:
                T_pos_m = self._sim_to_m_xy(herder_knots[:, herder_idx])
                T_fine, spline = self._smooth_path(T_time_s, T_pos_m)
                T_pos_smooth = np.asarray(spline(T_fine))
                T_yaw = self._yaw_from_tangent(T_pos_smooth)
                dog_plans.append({"T_pos": T_pos_smooth, "T_yaw": T_yaw, "delta_t": float(T_fine[1] - T_fine[0])})

        return cmd, viz_trajs, dog_plans

    def _advance_shadow_from_traj(self, traj):
        future_herd = np.asarray(traj.state_next.base.herd_state)
        future_herder = np.asarray(traj.state_next.base.herder_state)
        future_centers = np.asarray(traj.state_next.base.centers[:, :2, :2])
        if len(future_herd) > 0:
            self.shadow_herd_state = np.array(future_herd[0], copy=True)
        if len(future_herder) > 0:
            self.shadow_herder_state = np.array(future_herder[0], copy=True)
        if len(future_centers) > 0:
            self.shadow_centers = np.array(future_centers[0], copy=True)
        temporal_idx = np.asarray(getattr(traj, "temporal_node_idx", np.array([self.shadow_temporal_node_idx])))
        if temporal_idx.size > 0:
            use_idx = min(1, temporal_idx.size - 1)
            self.shadow_temporal_node_idx = int(temporal_idx[use_idx])
