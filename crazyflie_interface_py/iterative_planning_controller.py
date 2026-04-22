#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import time
from typing import Any

import numpy as np
import rclpy
import rowan
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from example_interfaces.msg import Float32MultiArray
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

from crazyflie_interface_py.template_controller import TemplateController

MODEL_ROOT = pathlib.Path(get_package_share_directory("crazyflie_interface")) / "models"

try:
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import jax.tree_util as jtu
    import jax_dataclasses as jdc
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Circle, Rectangle
    from rraa_rl.collector import Collector
    from rraa_rl.evaluate_dag import evaluate_dag
    from rraa_rl.load_ckpt import load_ckpt
    from rraa_rl.rollout_utils import extract_rollouts_eval
    from rraa_rl.src.env.general_task.env import get_rules
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
    save_debug_plot_on_first_rollout: bool = True
    use_accel_feedforward: bool = False
    max_speed_command_mps: float = 0.6
    max_accel_command_mps2: float = 0.8
    save_reference_rollout_plot: bool = True
    save_reference_rollout_gif: bool = False
    save_sim_overlay_plot: bool = True
    use_real_eval_state: bool = False
    real_eval_sample_multiplier: int = 8
    reference_seed: int = 1
    stage_to_reference_start: bool = True
    stage_acceptance_radius_m: float = 0.10
    startup_delay_s: float = 5.0
    use_sim_time: bool = True

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
        self._apply_scalar_parameter_overrides()
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
        viz_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.env_marker_pub = self.create_publisher(MarkerArray, "controller_env_markers", viz_qos)
        self.dog_plan_pubs = []
        self.dog_path_pubs = []
        self.reference_path_pubs = []
        self.sim_history_path_pubs = []
        self.debug_env_default_paths_m = None
        self.debug_reference_paths_m = None
        self.debug_env_default_predicates = None
        self.debug_reference_predicates = None
        self.debug_sim_history_m = []
        self.reference_start_targets_m = None
        self._startup_release_time = None
        self._startup_status_counter = 0
        self.debug_observed_xyz_pub = self.create_publisher(
            Float32MultiArray, "controller_debug_observed_xyz", 1
        )
        self.debug_target_xyz_pub = self.create_publisher(
            Float32MultiArray, "controller_debug_target_xyz", 1
        )

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

    def _apply_scalar_parameter_overrides(self):
        for key, value in self._ros_parameters.items():
            if isinstance(value, dict):
                continue
            if hasattr(self, key):
                setattr(self, key, value)
                self.get_logger().info(f"Applied ROS parameter override: {key}={value!r}")

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
        self.template_state = self._get_template_state()

        cfg = self.env.base.cfg
        self.sim_to_m = float(self.agent_radius_real_m / cfg.agent_radius)
        vel_max_sim_per_simtime = float(max(cfg.vel_maxs))
        vel_max_m_per_simtime = vel_max_sim_per_simtime * self.sim_to_m
        self.s_per_simtime = vel_max_m_per_simtime / self.vel_max_real_mps
        self.dt_s = float(cfg.dt * self.s_per_simtime)
        if self.plan_eval_dt_s is None:
            self.plan_eval_dt_s = 1.0 / self.controller_rate

        self._init_shadow_from_template()
        self.shadow_herd_state_template = (
            None if getattr(self, "shadow_herd_state", None) is None else np.array(self.shadow_herd_state, copy=True)
        )
        self.shadow_herder_state_template = (
            None if getattr(self, "shadow_herder_state", None) is None else np.array(self.shadow_herder_state, copy=True)
        )
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
        self.reference_path_pubs = [
            self.create_publisher(Path, f"controller_reference_path_{ii:02d}", 1) for ii in range(self.nbr_robots)
        ]
        self.sim_history_path_pubs = [
            self.create_publisher(Path, f"controller_sim_history_path_{ii:02d}", 1) for ii in range(self.nbr_robots)
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
        self._publish_environment_rviz()

    def _get_template_state(self):
        reference_seed = int(max(self.reference_seed, 0))
        n_envs = max(1, reference_seed + 1)
        if self.use_real_eval_state:
            batch_state = self.env.get_real_eval_states(
                n_envs,
                self.real_eval_sample_multiplier * n_envs,
                root_only=True,
            )
        else:
            batch_state = self.env.get_eval_states(n_envs, root_only=True)
        select_idx = min(reference_seed, n_envs - 1)
        return jtu.tree_map(lambda x: x[select_idx : select_idx + 1], batch_state)

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

    def _rollout_once(self, plan_state: Any, rollout_steps: int | None = None):
        rollout_steps = rollout_steps or self.rollout_horizon_steps
        collect_opts = dict(temporal_transitions=True)
        Tb_rollout, _info_collect = self.agent.collect_eval_with_states(
            self.collector,
            plan_state,
            rollout_steps,
            # temporal_transitions=True,
            **collect_opts
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
        vel_xy_mps = np.asarray(vel_xy_mps, dtype=float)
        acc_xy_mps2 = np.asarray(acc_xy_mps2, dtype=float)
        speed = float(np.linalg.norm(vel_xy_mps))
        if speed > self.max_speed_command_mps > 0.0:
            vel_xy_mps = vel_xy_mps * (self.max_speed_command_mps / speed)
        acc_mag = float(np.linalg.norm(acc_xy_mps2))
        if acc_mag > self.max_accel_command_mps2 > 0.0:
            acc_xy_mps2 = acc_xy_mps2 * (self.max_accel_command_mps2 / acc_mag)
        if not self.use_accel_feedforward:
            acc_xy_mps2[:] = 0.0

        cmd[0:3] = np.array([pos_xy_m[0], pos_xy_m[1], self.flight_height_m])
        cmd[3:6] = np.array([vel_xy_mps[0], vel_xy_mps[1], 0.0])
        cmd[6:10] = np.array([0.0, 0.0, 0.0, 1.0])
        cmd[10:13] = np.array([0.0, 0.0, 0.0])
        cmd[13:16] = np.array([acc_xy_mps2[0], acc_xy_mps2[1], 0.0])

        yaw = yaw_override
        if yaw is None and speed > 1e-4:
            yaw = float(np.arctan2(vel_xy_mps[1], vel_xy_mps[0]))
        if yaw is not None:
            self.last_yaws[robot_idx] = yaw
            quat_wxyz = rowan.from_euler(0.0, 0.0, yaw, "xyz")
            cmd[6:10] = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        return cmd

    def _save_debug_rollout_plot(self, viz_trajs: list[np.ndarray], cmd: np.ndarray):
        if not self.save_debug_plot_on_first_rollout or _IMPORT_ERROR is not None:
            return
        out_path = pathlib.Path("/mounted_volume/tmp") / f"{self.get_name()}_first_rollout.png"
        fig, ax = plt.subplots(figsize=(7, 6))
        self._plot_environment(ax)
        for ii, T_pos in enumerate(viz_trajs):
            ax.plot(T_pos[:, 0], T_pos[:, 1], label=f"traj_{ii}")
            ax.scatter(cmd[ii, 0], cmd[ii, 1], s=30)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"{self.get_name()} first rollout")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        self.get_logger().info(f"Saved first rollout debug plot to {out_path}")

    def _save_reference_rollout_plot(
        self,
        reference_paths_m: list[np.ndarray],
        *,
        stem: str = "reference_rollout",
        title: str | None = None,
        herd_state: np.ndarray | None = None,
        herder_state: np.ndarray | None = None,
        temporal_node_idx: np.ndarray | None = None,
        predicates_next: dict[str, np.ndarray] | None = None,
    ):
        if not self.save_reference_rollout_plot or _IMPORT_ERROR is not None:
            return
        out_path = pathlib.Path("/mounted_volume/tmp") / f"{self.get_name()}_{stem}.png"
        fig, ax = plt.subplots(figsize=(7, 6))
        self._plot_environment(ax, herd_state=herd_state, herder_state=herder_state)
        for ii, T_pos in enumerate(reference_paths_m):
            ax.plot(T_pos[:, 0], T_pos[:, 1], label=f"ref_{ii}")
            ax.scatter(T_pos[0, 0], T_pos[0, 1], s=20, marker="o")
            ax.scatter(T_pos[-1, 0], T_pos[-1, 1], s=20, marker="x")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(title or f"{self.get_name()} full rraa-rl rollout")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        self.get_logger().info(f"Saved full rollout reference plot to {out_path}")

        self._save_reference_rollout_gif(
            reference_paths_m,
            stem=stem,
            title=title,
            herd_state=herd_state,
            herder_state=herder_state,
            temporal_node_idx=temporal_node_idx,
            predicates_next=predicates_next,
        )
        self._save_reference_profile_plot(reference_paths_m)

    def _save_reference_rollout_gif(
        self,
        reference_paths_m: list[np.ndarray],
        *,
        stem: str = "reference_rollout",
        title: str | None = None,
        herd_state: np.ndarray | None = None,
        herder_state: np.ndarray | None = None,
        temporal_node_idx: np.ndarray | None = None,
        predicates_next: dict[str, np.ndarray] | None = None,
    ):
        if not self.save_reference_rollout_gif or _IMPORT_ERROR is not None:
            return
        valid_paths = [np.asarray(T_pos) for T_pos in reference_paths_m if len(T_pos) >= 1]
        if len(valid_paths) == 0:
            self.get_logger().warning("Skipping reference rollout GIF because there are no valid paths.")
            return

        n_frames = max(len(T_pos) for T_pos in valid_paths)
        if n_frames < 2:
            self.get_logger().warning("Skipping reference rollout GIF because all reference paths are shorter than 2 samples.")
            return

        out_path = pathlib.Path("/mounted_volume/tmp") / f"{self.get_name()}_{stem}.gif"
        fig, ax = plt.subplots(figsize=(7, 6))
        self._plot_environment(ax, herd_state=herd_state, herder_state=herder_state, show_init_agents=False)

        line_artists = []
        point_artists = []
        disc_artists = []
        herd_disc_color = (1.0, 0.71, 0.72, 0.75)
        herder_disc_color = (0.20, 0.54, 0.74, 0.75)
        agent_radius = self._plot_agent_radius_m()
        for ii, T_pos in enumerate(valid_paths):
            (line,) = ax.plot([], [], lw=1.5, label=f"ref_{ii}")
            (point,) = ax.plot([], [], marker="o", ms=5, linestyle="None")
            kind = self._controlled_slots()[ii][0] if ii < len(self._controlled_slots()) else "herder"
            color = herd_disc_color if kind == "herd" else herder_disc_color
            disc = Circle((0.0, 0.0), agent_radius, facecolor=color, edgecolor="none", zorder=4)
            ax.add_patch(disc)
            line_artists.append((line, T_pos))
            point_artists.append((point, T_pos))
            disc_artists.append((disc, T_pos))

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(title or f"{self.get_name()} full rraa-rl rollout")
        ax.legend(loc="best", fontsize=8)
        node_text = ax.text(
            0.02,
            0.98,
            "",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"),
        )
        predicate_text_top = ax.text(
            0.02,
            0.085,
            "",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            family="monospace",
            color="#1f77b4",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.8, edgecolor="none"),
        )
        predicate_text_mid = ax.text(
            0.02,
            0.050,
            "",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            family="monospace",
            color="#d62728",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.8, edgecolor="none"),
        )
        predicate_text_bottom = ax.text(
            0.02,
            0.015,
            "",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            family="monospace",
            color="#d62728",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.8, edgecolor="none"),
        )
        fig.tight_layout()

        def _update(frame_idx: int):
            artists = []
            for line, T_pos in line_artists:
                last_idx = min(frame_idx + 1, len(T_pos))
                line.set_data(T_pos[:last_idx, 0], T_pos[:last_idx, 1])
                artists.append(line)
            for point, T_pos in point_artists:
                use_idx = min(frame_idx, len(T_pos) - 1)
                point.set_data([T_pos[use_idx, 0]], [T_pos[use_idx, 1]])
                artists.append(point)
            for disc, T_pos in disc_artists:
                use_idx = min(frame_idx, len(T_pos) - 1)
                disc.center = (float(T_pos[use_idx, 0]), float(T_pos[use_idx, 1]))
                artists.append(disc)
            if temporal_node_idx is not None and len(temporal_node_idx) > 0:
                node_idx = int(temporal_node_idx[min(frame_idx, len(temporal_node_idx) - 1)])
                node_text.set_text(f"temporal_node_idx={node_idx}")
            else:
                node_text.set_text("temporal_node_idx=<missing>")
            artists.append(node_text)
            if predicates_next:
                reach_keys = [
                    key for key in ("herd_gate_0", "herd_gate_1", "herd_herded") if key in predicates_next
                ]
                safety_keys = [
                    key for key in ("herder_oob", "herder_unsafe", "herder_collide_wall", "herder_collide") if key in predicates_next
                ]
                if not reach_keys and not safety_keys:
                    fallback = sorted(predicates_next.keys())[:6]
                    reach_keys = fallback[:3]
                    safety_keys = fallback[3:]

                reach_parts_top = []
                for key in reach_keys:
                    values = np.asarray(predicates_next[key]).reshape(-1)
                    use_idx = min(frame_idx, len(values) - 1)
                    reach_parts_top.append(f"{key}={float(values[use_idx]):+.2f}")

                safety_parts_top = []
                safety_parts_bottom = []
                split_idx = (len(safety_keys) + 1) // 2
                for idx_key, key in enumerate(safety_keys):
                    values = np.asarray(predicates_next[key]).reshape(-1)
                    use_idx = min(frame_idx, len(values) - 1)
                    target = safety_parts_top if idx_key < split_idx else safety_parts_bottom
                    target.append(f"{key}={float(values[use_idx]):+.2f}")

                predicate_text_top.set_text(" | ".join(reach_parts_top))
                predicate_text_mid.set_text(" | ".join(safety_parts_top))
                predicate_text_bottom.set_text(" | ".join(safety_parts_bottom))
            else:
                predicate_text_top.set_text("")
                predicate_text_mid.set_text("")
                predicate_text_bottom.set_text("")
            artists.append(predicate_text_top)
            artists.append(predicate_text_mid)
            artists.append(predicate_text_bottom)
            return artists

        anim = FuncAnimation(fig, _update, frames=n_frames, interval=max(int(1000 * self.dt_s), 50), blit=False)
        anim.save(out_path, writer=PillowWriter(fps=max(int(round(1.0 / max(self.dt_s, 1e-3))), 1)))
        plt.close(fig)
        self.get_logger().info(f"Saved full rollout reference GIF to {out_path}")

    def _save_reference_profile_plot(self, reference_paths_m: list[np.ndarray]):
        if _IMPORT_ERROR is not None or len(reference_paths_m) == 0:
            return
        valid_paths = [np.asarray(T_pos) for T_pos in reference_paths_m if len(T_pos) >= 2]
        if len(valid_paths) == 0:
            self.get_logger().warning("Skipping reference profile plot because all reference paths are shorter than 2 samples.")
            return
        out_path = pathlib.Path("/mounted_volume/tmp") / f"{self.get_name()}_reference_profiles.png"
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8), layout="constrained")
        for ii, T_pos in enumerate(valid_paths):
            T_time = np.arange(len(T_pos), dtype=float) * self.dt_s
            vel = np.gradient(T_pos, self.dt_s, axis=0)
            acc = np.gradient(vel, self.dt_s, axis=0)
            axes[0].plot(T_time, T_pos[:, 0], label=f"agent_{ii}")
            axes[1].plot(T_time, np.linalg.norm(vel, axis=1))
            axes[2].plot(T_time, np.linalg.norm(acc, axis=1))
        axes[0].set_ylabel("x [m]")
        axes[1].set_ylabel("|v| [m/s]")
        axes[2].set_ylabel("|a| [m/s²]")
        axes[2].set_xlabel("time [s]")
        axes[0].set_title(f"{self.get_name()} full rollout profiles")
        axes[1].axhline(self.max_speed_command_mps, color="red", ls="--", lw=1.0)
        axes[2].axhline(self.max_accel_command_mps2, color="red", ls="--", lw=1.0)
        axes[0].legend(loc="best", fontsize=8, ncol=2)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        self.get_logger().info(f"Saved full rollout profile plot to {out_path}")

    def _save_sim_overlay_plot(self):
        if not self.save_sim_overlay_plot or _IMPORT_ERROR is not None or self.debug_reference_paths_m is None:
            return
        if len(self.debug_sim_history_m) < 2:
            return
        out_path = pathlib.Path("/mounted_volume/tmp") / f"{self.get_name()}_sim_overlay.png"
        hist = np.stack(self.debug_sim_history_m, axis=0)  # (T, N, 2)
        fig, ax = plt.subplots(figsize=(7, 6))
        self._plot_environment(ax)
        for ii, T_pos in enumerate(self.debug_reference_paths_m):
            ax.plot(T_pos[:, 0], T_pos[:, 1], ls="--", lw=1.0, label=f"ref_{ii}")
        for ii in range(hist.shape[1]):
            ax.plot(hist[:, ii, 0], hist[:, ii, 1], lw=1.5, label=f"sim_{ii}")
            ax.scatter(hist[-1, ii, 0], hist[-1, ii, 1], s=20)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"{self.get_name()} sim vs rraa-rl reference")
        ax.legend(loc="best", fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        self.get_logger().info(f"Saved sim overlay plot to {out_path}")

    def _compute_reference_paths(self, plan_state: Any) -> tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]:
        full_steps = max(int(self.env.eval_T), int(self.rollout_horizon_steps))
        full_traj = self._rollout_once(plan_state, rollout_steps=full_steps)
        self._log_rollout_transition_diagnostics(full_traj, label="reference_observed")
        predicates_next = {
            key: np.asarray(val).reshape(-1) for key, val in jax.device_get(getattr(full_traj, "predicates_next")).items()
        }
        return self._extract_controlled_paths_m(full_traj), self._extract_temporal_node_idx(full_traj), predicates_next

    def _compute_env_default_reference_paths(self) -> tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray], Any]:
        full_steps = max(int(self.env.eval_T), int(self.rollout_horizon_steps))
        full_traj = self._rollout_once(self.template_state, rollout_steps=full_steps)
        self._log_rollout_transition_diagnostics(full_traj, label="reference_env_default")
        predicates_next = {
            key: np.asarray(val).reshape(-1) for key, val in jax.device_get(getattr(full_traj, "predicates_next")).items()
        }
        return self._extract_controlled_paths_m(full_traj), self._extract_temporal_node_idx(full_traj), predicates_next, self.template_state

    def _extract_temporal_node_idx(self, traj) -> np.ndarray:
        temporal_candidates = []
        state_now = getattr(traj, "state_now", None)
        if state_now is not None and hasattr(state_now, "temporal_node_idx"):
            temporal_candidates.append(("state_now.temporal_node_idx", np.asarray(state_now.temporal_node_idx)))
        if hasattr(traj, "temporal_node_idx"):
            temporal_candidates.append(("traj.temporal_node_idx", np.asarray(traj.temporal_node_idx)))

        if len(temporal_candidates) == 0:
            self.get_logger().warning("Rollout trajectory has no temporal_node_idx field.")
            return np.array([], dtype=int)

        source_name, temporal_idx = max(temporal_candidates, key=lambda item: item[1].size)
        temporal_idx = temporal_idx.astype(int, copy=False).reshape(-1)
        unique_nodes = np.unique(temporal_idx)
        self.get_logger().info(
            "Rollout temporal nodes: "
            f"source={source_name} unique={unique_nodes.tolist()} "
            f"len={len(temporal_idx)} "
            f"sequence_head={temporal_idx[:min(len(temporal_idx), 20)].tolist()}"
        )
        return temporal_idx

    def _log_rollout_transition_diagnostics(self, traj, label: str):
        try:
            obs_next = jax.device_get(getattr(traj, "obs_next"))
            predicates_next = jax.device_get(getattr(traj, "predicates_next"))
        except Exception as exc:
            self.get_logger().warning(f"Could not extract rollout transition diagnostics for {label}: {exc}")
            return

        try:
            Tt_reach_val = jax.device_get(jax.vmap(self.agent.get_t_reach_val)(obs_next, predicates_next))
        except Exception as exc:
            self.get_logger().warning(f"Could not compute reach values for {label}: {exc}")
            return

        temporal_idx = self._extract_temporal_node_idx(traj)
        if len(temporal_idx) == 0:
            return

        try:
            t_value_next = jax.device_get(
                jax.vmap(lambda obs: self.agent.network.select("critic")(obs, params=self.agent.network.params))(obs_next)
            )
        except Exception:
            try:
                t_value_next = jax.device_get(
                    jax.vmap(lambda obs: self.agent.network.select("critic")(obs, params=self.agent.network.params))(obs_next)
                )
            except Exception as exc:
                self.get_logger().warning(f"Could not compute critic values for {label}: {exc}")
                return

        try:
            temporal_idx_next = jax.device_get(
                jax.vmap(self.env.transition_temporal_node)(predicates_next, t_value_next, temporal_idx)
            )
        except Exception as exc:
            self.get_logger().warning(f"Could not compute temporal transitions for {label}: {exc}")
            return

        node_names = list(getattr(self.env, "temporal_node_names", []))
        self.get_logger().info(
            f"{label} temporal node names: {node_names}"
        )

        head_n = min(len(temporal_idx), 20)
        current_reach = Tt_reach_val[np.arange(len(temporal_idx)), temporal_idx]
        self.get_logger().info(
            f"{label} current-node reach head={np.round(current_reach[:head_n], 3).tolist()} "
            f"next_nodes_head={temporal_idx_next[:head_n].astype(int).tolist()}"
        )

        for node_idx in range(Tt_reach_val.shape[1]):
            self.get_logger().info(
                f"{label} reach[node={node_idx}] head={np.round(Tt_reach_val[:head_n, node_idx], 3).tolist()}"
            )

        for key in sorted(predicates_next.keys()):
            values = np.asarray(predicates_next[key]).reshape(-1)
            self.get_logger().info(
                f"{label} predicate[{key}] head={np.round(values[:head_n], 3).tolist()} "
                f"min={float(np.min(values)):.3f} max={float(np.max(values)):.3f}"
            )

        if "herd_gate_0" in predicates_next:
            gate0 = np.asarray(predicates_next["herd_gate_0"]).reshape(-1)
            pos_idx = np.flatnonzero(gate0 > 0.0)
            if len(pos_idx) == 0:
                self.get_logger().info(f"{label} herd_gate_0 never becomes positive.")
            else:
                self._log_positive_predicate_steps(
                    label=label,
                    predicate_name="herd_gate_0",
                    pos_idx=pos_idx,
                    predicates_next=predicates_next,
                    t_value_next=t_value_next,
                    Tt_reach_val=Tt_reach_val,
                    temporal_idx=temporal_idx,
                    temporal_idx_next=temporal_idx_next,
                    current_reach=current_reach,
                )

        if "herd_gate_1" in predicates_next:
            gate1 = np.asarray(predicates_next["herd_gate_1"]).reshape(-1)
            pos_idx = np.flatnonzero(gate1 > 0.0)
            if len(pos_idx) == 0:
                self.get_logger().info(f"{label} herd_gate_1 never becomes positive.")
            else:
                self._log_positive_predicate_steps(
                    label=label,
                    predicate_name="herd_gate_1",
                    pos_idx=pos_idx,
                    predicates_next=predicates_next,
                    t_value_next=t_value_next,
                    Tt_reach_val=Tt_reach_val,
                    temporal_idx=temporal_idx,
                    temporal_idx_next=temporal_idx_next,
                    current_reach=current_reach,
                )

    def _log_positive_predicate_steps(
        self,
        *,
        label: str,
        predicate_name: str,
        pos_idx: np.ndarray,
        predicates_next: dict[str, np.ndarray],
        t_value_next: np.ndarray,
        Tt_reach_val: np.ndarray,
        temporal_idx: np.ndarray,
        temporal_idx_next: np.ndarray,
        current_reach: np.ndarray,
    ):
        self.get_logger().info(
            f"{label} {predicate_name} positive indices head={pos_idx[:min(len(pos_idx), 20)].tolist()} count={len(pos_idx)}"
        )
        inspect_idx = pos_idx[: min(len(pos_idx), 8)]
        node_names = list(getattr(self.env, "temporal_node_names", []))
        for idx in inspect_idx:
            pred_t = {k: np.asarray(v).reshape(-1)[idx] for k, v in predicates_next.items()}
            t_value_t = np.asarray(t_value_next[idx])
            try:
                triggers = get_rules(self.env.temporal_nodes, self.env.dag_nodes, pred_t, t_value_t, which=np)
                trigger_summary = [
                    {
                        "parent": int(tr.parent),
                        "child": int(tr.child),
                        "condition": float(np.asarray(tr.condition)),
                    }
                    for tr in triggers
                ]
            except Exception as exc:
                trigger_summary = [{"error": str(exc)}]

            current_node = int(temporal_idx[idx])
            next_node = int(temporal_idx_next[idx])
            current_reach_val = float(current_reach[idx])
            critic_vals = np.round(np.asarray(t_value_t).reshape(-1), 3).tolist()
            reach_vals = np.round(np.asarray(Tt_reach_val[idx]).reshape(-1), 3).tolist()
            pred_focus = {
                k: float(pred_t[k])
                for k in sorted(pred_t.keys())
                if k in ("herd_gate_0", "herd_gate_1", "herd_herded", "herder_unsafe", "herder_oob")
            }
            node_name = node_names[current_node] if current_node < len(node_names) else str(current_node)
            self.get_logger().info(
                f"{label} {predicate_name}-positive step={int(idx)} node={current_node} ({node_name}) next={next_node} "
                f"current_reach={current_reach_val:.3f} "
                f"predicates={pred_focus} reach_vals={reach_vals} critic_vals={critic_vals} triggers={trigger_summary}"
            )
            if predicate_name in ("herd_gate_0", "herd_gate_1"):
                self._log_temporal_node_rule_details(
                    label=label,
                    current_node=current_node,
                    pred_t=pred_t,
                    t_value_t=t_value_t,
                )

    def _describe_dag_node(self, node_idx: int, depth: int = 0, max_depth: int = 3) -> str:
        node = self.env.dag_nodes[node_idx]
        prefix = "  " * depth
        if depth >= max_depth:
            return f"{prefix}{node_idx}: {type(node).__name__}"

        parts = [f"{prefix}{node_idx}: {type(node).__name__}"]
        args = getattr(node, "args", None)
        if args is not None:
            for child_idx in args:
                parts.append(self._describe_dag_node(int(child_idx), depth + 1, max_depth=max_depth))
        for attr_name in ("reach", "avoid", "arg"):
            if hasattr(node, attr_name):
                child_idx = int(getattr(node, attr_name))
                parts.append(f"{prefix}  {attr_name}->")
                parts.append(self._describe_dag_node(child_idx, depth + 1, max_depth=max_depth))
        if hasattr(node, "name"):
            parts.append(f"{prefix}  name={getattr(node, 'name')}")
        if hasattr(node, "value"):
            parts.append(f"{prefix}  value={getattr(node, 'value')}")
        return "\n".join(parts)

    def _log_temporal_node_rule_details(self, *, label: str, current_node: int, pred_t: dict[str, float], t_value_t: np.ndarray):
        dag_node_idx = int(self.env.temporal_nodes[current_node])
        dag_node = self.env.dag_nodes[dag_node_idx]
        self.get_logger().info(
            f"{label} temporal-node detail for node={current_node}: \n{self._describe_dag_node(dag_node_idx)}"
        )

        V_dict = {int(dag_id): np.asarray(t_value_t[temporal_idx]) for temporal_idx, dag_id in enumerate(self.env.temporal_nodes)}
        scratch = {}
        args = getattr(dag_node, "args", ())
        if args:
            child_vals = []
            for child_idx in args:
                val = float(np.asarray(evaluate_dag(self.env.dag_nodes, int(child_idx), pred_t, V_dict, scratch=scratch, which=np)))
                child_vals.append((int(child_idx), type(self.env.dag_nodes[int(child_idx)]).__name__, val))
            self.get_logger().info(f"{label} node={current_node} direct child values={child_vals}")

        for attr_name in ("reach", "avoid", "arg"):
            if hasattr(dag_node, attr_name):
                child_idx = int(getattr(dag_node, attr_name))
                val = float(np.asarray(evaluate_dag(self.env.dag_nodes, child_idx, pred_t, V_dict, scratch=scratch, which=np)))
                self.get_logger().info(
                    f"{label} node={current_node} {attr_name} child={child_idx} "
                    f"type={type(self.env.dag_nodes[child_idx]).__name__} value={val:.3f}"
                )

        if current_node == 0:
            self._log_root_branch_values(label=label, pred_t=pred_t, V_dict=V_dict)

    def _log_root_branch_values(self, *, label: str, pred_t: dict[str, float], V_dict: dict[int, np.ndarray]):
        scratch = {}
        focus_nodes = [9, 3, 2, 0, 8, 5, 4, 7, 6]
        values = []
        for node_idx in focus_nodes:
            if node_idx >= len(self.env.dag_nodes):
                continue
            try:
                val = float(
                    np.asarray(
                        evaluate_dag(
                            self.env.dag_nodes,
                            int(node_idx),
                            pred_t,
                            V_dict,
                            scratch=scratch,
                            which=np,
                        )
                    )
                )
            except Exception as exc:
                self.get_logger().warning(f"{label} could not evaluate root focus node {node_idx}: {exc}")
                continue
            values.append((node_idx, type(self.env.dag_nodes[node_idx]).__name__, val))

        self.get_logger().info(f"{label} root focus values={values}")

        for node_idx in (9, 3):
            if node_idx >= len(self.env.dag_nodes):
                continue
            node = self.env.dag_nodes[node_idx]
            args = getattr(node, "args", ())
            if not args:
                continue
            child_vals = []
            for child_idx in args:
                try:
                    val = float(
                        np.asarray(
                            evaluate_dag(
                                self.env.dag_nodes,
                                int(child_idx),
                                pred_t,
                                V_dict,
                                scratch=scratch,
                                which=np,
                            )
                        )
                    )
                except Exception as exc:
                    self.get_logger().warning(f"{label} could not evaluate root child node {child_idx}: {exc}")
                    continue
                child_vals.append((int(child_idx), type(self.env.dag_nodes[int(child_idx)]).__name__, val))
            self.get_logger().info(f"{label} root node={node_idx} child values={child_vals}")

    def _publish_path_series(self, pubs: list, paths_xy_m: list[np.ndarray], z: float):
        header = Header(frame_id="world")
        for pub, T_pos in zip(pubs, paths_xy_m):
            path_msg = Path(header=header)
            poses = []
            for pos in T_pos:
                pose = PoseStamped()
                pose.header = header
                pose.pose.position.x = float(pos[0])
                pose.pose.position.y = float(pos[1])
                pose.pose.position.z = float(z)
                pose.pose.orientation.w = 1.0
                poses.append(pose)
            path_msg.poses = poses
            pub.publish(path_msg)

    def _publish_environment_rviz(self):
        header = Header(frame_id="world")
        markers = self._build_environment_markers(header)
        if markers:
            self.env_marker_pub.publish(MarkerArray(markers=markers))

    def _build_environment_markers(self, header: Header) -> list[Marker]:
        return []

    def _plot_environment(
        self,
        ax,
        herd_state: np.ndarray | None = None,
        herder_state: np.ndarray | None = None,
        show_init_agents: bool = True,
    ):
        return None

    def _plot_agent_radius_m(self) -> float:
        return 0.05

    def _disc_marker(self, header: Header, marker_id: int, ns: str, xy: np.ndarray, radius: float, color: ColorRGBA) -> Marker:
        marker = Marker()
        marker.header = header
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = float(xy[0])
        marker.pose.position.y = float(xy[1])
        marker.pose.position.z = 0.01
        marker.pose.orientation.w = 1.0
        marker.scale.x = float(2.0 * radius)
        marker.scale.y = float(2.0 * radius)
        marker.scale.z = 0.02
        marker.color = color
        return marker

    def _box_marker(
        self, header: Header, marker_id: int, ns: str, center_xy: np.ndarray, size_xy: np.ndarray, color: ColorRGBA
    ) -> Marker:
        marker = Marker()
        marker.header = header
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = float(center_xy[0])
        marker.pose.position.y = float(center_xy[1])
        marker.pose.position.z = 0.01
        marker.pose.orientation.w = 1.0
        marker.scale.x = float(size_xy[0])
        marker.scale.y = float(size_xy[1])
        marker.scale.z = 0.02
        marker.color = color
        return marker

    def _hover_cmds(self, state: np.ndarray) -> np.ndarray:
        states = self._reshape_states(state)
        u = np.zeros((self.nbr_robots, 16), dtype=float)
        for ii in range(self.nbr_robots):
            u[ii, 0:3] = np.array([states[ii, 0], states[ii, 1], self.flight_height_m])
            u[ii, 6:10] = np.array([0.0, 0.0, 0.0, 1.0])
        return u

    def _initialize_reference_start_targets(self):
        if self.debug_env_default_paths_m is None or self.reference_start_targets_m is not None:
            return
        self.reference_start_targets_m = np.stack(
            [np.asarray(T_pos[0], dtype=float) for T_pos in self.debug_env_default_paths_m],
            axis=0,
        )
        self._startup_release_time = time.monotonic() + max(0.0, float(self.startup_delay_s))
        self.get_logger().info(
            "Initialized reference-start staging targets: "
            f"{np.round(self.reference_start_targets_m, 2)}"
        )
        self.get_logger().info(
            f"Holding at reference-start targets for {float(self.startup_delay_s):.1f} s before active control."
        )

    def _staging_cmds(self, controlled_pos_m: np.ndarray) -> np.ndarray:
        cmd = np.zeros((self.nbr_robots, 16), dtype=float)
        for ii in range(self.nbr_robots):
            target_xy = self.reference_start_targets_m[ii]
            cmd[ii] = self._make_fullstate_cmd(
                ii,
                target_xy,
                np.zeros(2, dtype=float),
                np.zeros(2, dtype=float),
                yaw_override=0.0,
            )
        self._publish_debug_xyz(controlled_pos_m, cmd)
        return cmd

    def _publish_debug_xyz(self, observed_pos_m: np.ndarray, cmd: np.ndarray):
        observed_msg = Float32MultiArray()
        observed_msg.data = np.asarray(observed_pos_m, dtype=float).reshape(-1).tolist()
        self.debug_observed_xyz_pub.publish(observed_msg)

        target_msg = Float32MultiArray()
        target_msg.data = np.asarray(cmd[:, 0:3], dtype=float).reshape(-1).tolist()
        self.debug_target_xyz_pub.publish(target_msg)

    def _log_stage_debug(self, observed_pos_m: np.ndarray, cmd: np.ndarray, max_err_m: float):
        observed_fmt = np.round(np.asarray(observed_pos_m, dtype=float), 2).tolist()
        target_fmt = np.round(np.asarray(cmd[:, 0:3], dtype=float), 2).tolist()
        self.get_logger().info(
            "Staging debug: "
            f"max_err={max_err_m:.3f} m "
            f"observed_xyz={observed_fmt} "
            f"target_xyz={target_fmt}",
            throttle_duration_sec=2.0,
        )

    def __call__(self, state):
        self._ensure_loaded()
        if not self.in_flight:
            return self._hover_cmds(state).flatten()

        controlled_pos_m, controlled_vel_mps = self._current_positions_velocities(state)
        self.debug_sim_history_m.append(controlled_pos_m[:, :2].copy())
        if self.iteration == 0 and self.save_reference_rollout_plot and self.debug_env_default_paths_m is None:
            self.debug_env_default_paths_m, env_default_temporal_idx, self.debug_env_default_predicates, template_state = self._compute_env_default_reference_paths()
            template_base = template_state.base
            template_herd = getattr(template_base, "herd_state", None)
            template_herder = getattr(template_base, "herder_state", None)
            template_herd = None if template_herd is None else np.asarray(template_herd)[0]
            template_herder = None if template_herder is None else np.asarray(template_herder)[0]
            self._save_reference_rollout_plot(
                self.debug_env_default_paths_m,
                stem="reference_env_default",
                title=f"{self.get_name()} full rraa-rl rollout (env default)",
                herd_state=template_herd,
                herder_state=template_herder,
                temporal_node_idx=env_default_temporal_idx,
                predicates_next=self.debug_env_default_predicates,
            )
            self._publish_path_series(self.reference_path_pubs, self.debug_env_default_paths_m, self.flight_height_m + 0.03)
            self._initialize_reference_start_targets()

        if self.stage_to_reference_start and self.reference_start_targets_m is not None:
            cmd = self._staging_cmds(controlled_pos_m)
            hold_paths = [np.vstack([controlled_pos_m[ii, :2], self.reference_start_targets_m[ii]]) for ii in range(self.nbr_robots)]
            self._publish_rviz(hold_paths, cmd)
            self._startup_status_counter += 1
            pos_err = np.linalg.norm(controlled_pos_m[:, :2] - self.reference_start_targets_m, axis=1)
            remaining_s = 0.0 if self._startup_release_time is None else max(
                0.0, self._startup_release_time - time.monotonic()
            )
            if self._startup_status_counter % 20 == 0:
                self.get_logger().info(
                    "Holding at reference-start targets before active control: "
                    f"max_err={float(np.max(pos_err)):.3f} m remaining={remaining_s:.1f} s"
                )
                self._log_stage_debug(controlled_pos_m, cmd, float(np.max(pos_err)))
            if remaining_s > 0.0:
                return cmd.flatten()
            self.get_logger().info("Startup delay elapsed; enabling active control.")
            self.stage_to_reference_start = False

        self._apply_controlled_observations(controlled_pos_m, controlled_vel_mps)

        plan_state = self._make_plan_state()
        if self.iteration == 0 and self.save_reference_rollout_plot and self.debug_reference_paths_m is None:
            self.debug_reference_paths_m, observed_temporal_idx, self.debug_reference_predicates = self._compute_reference_paths(plan_state)
            self._save_reference_rollout_plot(
                self.debug_reference_paths_m,
                stem="reference_rollout",
                title=f"{self.get_name()} full rraa-rl rollout (observed state)",
                temporal_node_idx=observed_temporal_idx,
                predicates_next=self.debug_reference_predicates,
            )
            self._publish_path_series(self.reference_path_pubs, self.debug_reference_paths_m, self.flight_height_m + 0.03)
        if self.iteration == 0:
            self.get_logger().info("Running first rollout from observed state")
        traj = self._rollout_once(plan_state)
        if self.iteration == 0:
            self.get_logger().info("Building first command set from rollout")
        cmd, viz_trajs, dog_plans = self._build_commands_from_traj(traj)
        if self.iteration == 0:
            self.get_logger().info(
                "First command stats: "
                f"speed_max={np.max(np.linalg.norm(cmd[:, 3:6], axis=1)):.3f} m/s, "
                f"acc_max={np.max(np.linalg.norm(cmd[:, 13:16], axis=1)):.3f} m/s^2"
            )
            self._save_debug_rollout_plot(viz_trajs, cmd)
        self._publish_rviz(viz_trajs, cmd)
        self._publish_dog_plans(dog_plans)
        self._advance_shadow_from_traj(traj)

        if self.simulate_controlled_positions:
            self.ghost_positions_m = cmd[:, 0:3].copy()
            self.ghost_velocities_mps = cmd[:, 3:6].copy()

        self.iteration += 1
        if self.iteration % 20 == 0:
            self._save_sim_overlay_plot()
            self._publish_path_series(
                self.sim_history_path_pubs,
                [np.asarray(self.debug_sim_history_m)[:, ii, :] for ii in range(self.nbr_robots)],
                self.flight_height_m + 0.06,
            )
        if self.iteration % 20 == 0:
            xy = cmd[:, 0:2]
            self.get_logger().info(f"Iterative replanning iteration {self.iteration}, targets={xy}")
        return cmd.flatten()

    def _publish_rviz(self, viz_trajs: list[np.ndarray], cmd: np.ndarray):
        header = Header(frame_id="world")
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

        header = Header(frame_id="world")
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

    def _extract_controlled_paths_m(self, traj) -> list[np.ndarray]:
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
    use_real_eval_state = True
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

    def _extract_controlled_paths_m(self, traj) -> list[np.ndarray]:
        herd_knots = np.asarray(traj.state_now.base.herd_state)
        herder_knots = np.asarray(traj.state_next.base.herder_state[:, :, :2])

        paths_m = []
        for kind, idx in self._controlled_slots():
            T_pos_sim = herd_knots[:, idx] if kind == "herd" else herder_knots[:, idx]
            paths_m.append(self._sim_to_m_xy(T_pos_sim))
        return paths_m

    def _build_environment_markers(self, header: Header) -> list[Marker]:
        env = self.env.base
        markers: list[Marker] = []
        wall_color = ColorRGBA(r=0.33, g=0.33, b=0.33, a=0.75)
        gate_color = ColorRGBA(r=0.47, g=0.60, b=0.47, a=0.35)
        herd_color = ColorRGBA(r=1.0, g=0.71, b=0.72, a=0.65)
        herder_color = ColorRGBA(r=0.20, g=0.54, b=0.74, a=0.65)
        marker_id = 0

        herded_center = self._sim_to_m_xy(np.asarray(env.herded_center))
        herded_radius = float(env.cfg.herded_radius * self.sim_to_m)
        markers.append(self._disc_marker(header, marker_id, "env", herded_center, herded_radius, gate_color))
        marker_id += 1

        gates = np.asarray(env.gates)
        for gate_xy in gates:
            markers.append(self._disc_marker(header, marker_id, "env", self._sim_to_m_xy(gate_xy), herded_radius, gate_color))
            marker_id += 1

        for wall in [env.wall_lower_aabb, env.wall_upper_aabb]:
            lo = self._sim_to_m_xy(np.asarray(wall.minpos))
            hi = self._sim_to_m_xy(np.asarray(wall.maxpos))
            center = 0.5 * (lo + hi)
            size = hi - lo
            markers.append(self._box_marker(header, marker_id, "env", center, size, wall_color))
            marker_id += 1

        halfsize = np.asarray(env.cfg.halfsize) * self.sim_to_m
        center = self.center_shift_m
        thick = 0.1
        bounds = [
            (center + np.array([halfsize[0] + thick, 0.0]), np.array([2 * thick, 2 * halfsize[1]])),
            (center - np.array([halfsize[0] + thick, 0.0]), np.array([2 * thick, 2 * halfsize[1]])),
            (center + np.array([0.0, halfsize[1] + thick]), np.array([2 * halfsize[0], 2 * thick])),
            (center - np.array([0.0, halfsize[1] + thick]), np.array([2 * halfsize[0], 2 * thick])),
        ]
        for c_xy, s_xy in bounds:
            markers.append(self._box_marker(header, marker_id, "env", c_xy, s_xy, wall_color))
            marker_id += 1

        agent_radius = float(env.cfg.agent_radius * self.sim_to_m)
        for ii in range(self.shadow_herd_state.shape[0]):
            markers.append(
                self._disc_marker(header, marker_id, "env_init", self._sim_to_m_xy(self.shadow_herd_state[ii, :2]), agent_radius, herd_color)
            )
            marker_id += 1
        for ii in range(self.shadow_herder_state.shape[0]):
            markers.append(
                self._disc_marker(
                    header, marker_id, "env_init", self._sim_to_m_xy(self.shadow_herder_state[ii, :2]), agent_radius, herder_color
                )
            )
            marker_id += 1

        return markers

    def _plot_environment(
        self,
        ax,
        herd_state: np.ndarray | None = None,
        herder_state: np.ndarray | None = None,
        show_init_agents: bool = True,
    ):
        env = self.env.base
        wall_color = (0.33, 0.33, 0.33, 0.25)
        gate_color = (0.47, 0.60, 0.47, 0.18)
        herd_color = (1.0, 0.71, 0.72, 0.45)
        herder_color = (0.20, 0.54, 0.74, 0.45)

        herded_center = self._sim_to_m_xy(np.asarray(env.herded_center))
        herded_radius = float(env.cfg.herded_radius * self.sim_to_m)
        ax.add_patch(Circle(herded_center, herded_radius, facecolor=gate_color, edgecolor="none"))

        gates = np.asarray(env.gates)
        for gate_xy in gates:
            ax.add_patch(Circle(self._sim_to_m_xy(gate_xy), herded_radius, facecolor=gate_color, edgecolor="none"))

        for wall in [env.wall_lower_aabb, env.wall_upper_aabb]:
            lo = self._sim_to_m_xy(np.asarray(wall.minpos))
            hi = self._sim_to_m_xy(np.asarray(wall.maxpos))
            size = hi - lo
            ax.add_patch(Rectangle(lo, size[0], size[1], facecolor=wall_color, edgecolor="none"))

        halfsize = np.asarray(env.cfg.halfsize) * self.sim_to_m
        center = self.center_shift_m
        thick = 0.1
        bounds = [
            (center + np.array([halfsize[0], -halfsize[1]]), np.array([thick, 2 * halfsize[1]])),
            (center + np.array([-halfsize[0] - thick, -halfsize[1]]), np.array([thick, 2 * halfsize[1]])),
            (center + np.array([-halfsize[0], halfsize[1]]), np.array([2 * halfsize[0], thick])),
            (center + np.array([-halfsize[0], -halfsize[1] - thick]), np.array([2 * halfsize[0], thick])),
        ]
        for lo, size in bounds:
            ax.add_patch(Rectangle(lo, size[0], size[1], facecolor=wall_color, edgecolor="none"))

        agent_radius = float(env.cfg.agent_radius * self.sim_to_m)
        if show_init_agents:
            herd_state = self.shadow_herd_state if herd_state is None else np.asarray(herd_state)
            herder_state = self.shadow_herder_state if herder_state is None else np.asarray(herder_state)
            for ii in range(herd_state.shape[0]):
                xy = self._sim_to_m_xy(herd_state[ii, :2])
                ax.add_patch(Circle(xy, agent_radius, facecolor=herd_color, edgecolor="none"))
            for ii in range(herder_state.shape[0]):
                xy = self._sim_to_m_xy(herder_state[ii, :2])
                ax.add_patch(Circle(xy, agent_radius, facecolor=herder_color, edgecolor="none"))

    def _plot_agent_radius_m(self) -> float:
        return float(self.env.base.cfg.agent_radius * self.sim_to_m)


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

    def _extract_controlled_paths_m(self, traj) -> list[np.ndarray]:
        future_herder = np.asarray(traj.state_next.base.herder_state[:, :, :2])
        curr_herder = self.shadow_herder_state[:, :2]
        herder_knots = np.concatenate([curr_herder[None, ...], future_herder], axis=0)

        paths_m = []
        for _kind, idx in self._controlled_slots():
            paths_m.append(self._sim_to_m_xy(herder_knots[:, idx]))
        return paths_m
