#!/usr/bin/env python3
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Path
from rclpy.duration import Duration
from rclpy.time import Time
from std_msgs.msg import Header

from crazyflie_interface.msg import CFTrajArray
from crazyflie_interface_py.template_controller import TemplateController
from crazyflie_interface_py.traj_manager import TrajManager
from crazyflie_interface_py.tracker_utils import (
    build_full_state_command,
    hold_current_positions,
    message_to_traj,
    solve_assignment,
    state_to_matrix,
)


def quat_from_yaw(yaw: float) -> Quaternion:
    half = 0.5 * yaw
    return Quaternion(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half))


def pose_stamped_from_position_yaw(stamp: Time, position: np.ndarray, yaw: float, frame_id: str = "world") -> PoseStamped:
    header = Header(stamp=stamp.to_msg(), frame_id=frame_id)
    pose = PoseStamped(header=header)
    pose.pose.position.x = float(position[0])
    pose.pose.position.y = float(position[1])
    pose.pose.position.z = float(position[2])
    pose.pose.orientation = quat_from_yaw(yaw)
    return pose


class RealmCFTrackerController(TemplateController):
    def __init__(self, node_name: str = "realmcf_tracker_controller"):
        self.control_publisher_topic = "cf_interface/control_full_state"
        super().__init__(
            node_name,
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
            controller_rate=20.0,
        )

        self._ros_parameters = self._param_to_dict(self._parameters)
        robots = self._ros_parameters.get("robots", {})
        self.nbr_robots = len(robots)
        self.height = float(self._ros_parameters.get("realmcf", {}).get("height", 0.5))
        self.override_height = bool(self._ros_parameters.get("realmcf", {}).get("override_height", True))
        self.viz_duration = float(self._ros_parameters.get("realmcf", {}).get("viz_duration", 3.0))
        self.frame_id = self._ros_parameters.get("realmcf", {}).get("world_frame", "world")

        cfg = TrajManager.Cfg()
        cfg.lookahead_dt = float(self._ros_parameters.get("realmcf", {}).get("lookahead_dt", cfg.lookahead_dt))
        cfg.initial_err_frac = float(
            self._ros_parameters.get("realmcf", {}).get("initial_err_frac", cfg.initial_err_frac)
        )
        cfg.err_decay_halflife = float(
            self._ros_parameters.get("realmcf", {}).get("err_decay_halflife", cfg.err_decay_halflife)
        )
        self._traj_cfg = cfg
        self.traj_managers = [TrajManager(cfg) for _ in range(self.nbr_robots)]

        self.first_step_time: Time | None = None
        self.cf_to_traj: list[int] | None = None
        self.initial_hold_state: np.ndarray | None = None

        self.create_subscription(CFTrajArray, "traj_input", self._traj_input_cb, 5)
        self.path_pubs = [self.create_publisher(Path, f"/Ag{i:02}/path", 10) for i in range(self.nbr_robots)]

        self.start_controller()
        self.get_logger().info(f"Tracker configured for {self.nbr_robots} robots")

    def _ensure_robot_count(self, n_robots: int):
        if self.nbr_robots == n_robots:
            return

        if self.nbr_robots != 0 and self.nbr_robots != n_robots:
            raise ValueError(f"Tracker already configured for {self.nbr_robots} robots, got {n_robots}")

        self.nbr_robots = n_robots
        self.traj_managers = [TrajManager(self._traj_cfg) for _ in range(self.nbr_robots)]
        self.path_pubs = [self.create_publisher(Path, f"/Ag{i:02}/path", 10) for i in range(self.nbr_robots)]
        self.get_logger().info(f"Inferred tracker robot count as {self.nbr_robots}")

    def _param_to_dict(self, param_ros):
        tree = {}
        for item in param_ros:
            t = tree
            parts = item.split(".")
            for part in parts[:-1]:
                t = t.setdefault(part, {})
            t.setdefault(parts[-1], param_ros[item].value)
        return tree

    def _traj_input_cb(self, msg_arr: CFTrajArray):
        if self.nbr_robots == 0:
            self._ensure_robot_count(len(msg_arr.trajs))

        if self.first_step_time is None:
            self.first_step_time = Time.from_msg(msg_arr.stamp)

        if len(msg_arr.trajs) != self.nbr_robots:
            self.get_logger().warning(
                f"Received {len(msg_arr.trajs)} trajectories for {self.nbr_robots} robots; ignoring message"
            )
            return

        if self.state is None:
            self.get_logger().warning("State not available yet; ignoring first trajectory batch")
            return

        states = state_to_matrix(self.state, self.nbr_robots)
        if self.cf_to_traj is None:
            start_points = [states[i, :2] for i in range(self.nbr_robots)]
            end_points = []
            first_step_time_s = float(self.first_step_time.nanoseconds) * 1e-9
            for msg_traj in msg_arr.trajs:
                traj = message_to_traj(msg_traj, first_step_time_s)
                end_points.append(traj.T_pos[0, :2])
            self.cf_to_traj = solve_assignment(start_points, end_points)
            self.get_logger().info(f"Established trajectory assignment {self.cf_to_traj}")

        now = self.get_clock().now()
        elapsed_s = (now - self.first_step_time).nanoseconds * 1e-9
        first_step_time_s = float(self.first_step_time.nanoseconds) * 1e-9
        reordered_trajs = [msg_arr.trajs[self.cf_to_traj[i]] for i in range(self.nbr_robots)]

        for i, msg_traj in enumerate(reordered_trajs):
            traj = message_to_traj(msg_traj, first_step_time_s)
            self.traj_managers[i].add_trajectory(elapsed_s, traj)

        self._publish_paths(now)

    def _publish_paths(self, now: Time):
        if self.first_step_time is None:
            return

        elapsed_s = (now - self.first_step_time).nanoseconds * 1e-9
        for i, traj_manager in enumerate(self.traj_managers):
            if traj_manager.T_time_prev is None:
                continue

            end_time = min(elapsed_s + self.viz_duration, traj_manager.T_time_prev[-1])
            if end_time <= elapsed_s:
                continue

            times = np.linspace(elapsed_s, end_time, num=32)
            path = Path()
            path.header = Header(stamp=now.to_msg(), frame_id=self.frame_id)
            for t in times:
                kin = traj_manager.query(t, clip=True)
                dt = max(0.0, t - elapsed_s)
                stamp = now + Duration(seconds=dt)
                path.poses.append(pose_stamped_from_position_yaw(stamp, kin.pos, kin.yaw, self.frame_id))
            self.path_pubs[i].publish(path)

    def __call__(self, state):
        if self.nbr_robots == 0:
            state_size = np.asarray(state).size
            if state_size % 13 != 0:
                raise ValueError(f"Cannot infer robot count from state length {state_size}")
            self._ensure_robot_count(state_size // 13)

        states = state_to_matrix(state, self.nbr_robots)

        if self.initial_hold_state is None:
            self.initial_hold_state = states.copy()

        if self.first_step_time is None or any(traj_manager.T_time_prev is None for traj_manager in self.traj_managers):
            return hold_current_positions(self.initial_hold_state, self.height if self.override_height else None)

        now = self.get_clock().now()
        elapsed_s = max(0.0, (now - self.first_step_time).nanoseconds * 1e-9)
        queried_pts = [traj_manager.query(elapsed_s, clip=True) for traj_manager in self.traj_managers]
        return build_full_state_command(states, queried_pts, self.height if self.override_height else None)


def main(args=None):
    rclpy.init(args=args)
    controller = RealmCFTrackerController()
    rclpy.spin(controller)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
