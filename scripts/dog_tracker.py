#!/usr/bin/env python3
import math
import time

import numpy as np
import rclpy
from example_interfaces.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from nav_msgs.msg import Odometry, Path
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import Header
from tf_transformations import euler_from_quaternion


def quat_msg_to_yaw(q) -> float:
    _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
    return yaw


def angular_distance(a: float, b: float) -> float:
    return (a - b + math.pi) % (2 * math.pi) - math.pi


class DogTracker(Node):
    plan_topic = "dog_plan_00"
    odom_topic = "/dog/odom"
    cmd_vel_topic = "/cmd_vel"
    control_hz = 20.0
    pose_timeout_s = 0.2
    viz_duration = 1.0
    pose_shift = np.array([0.20, 0.0])

    kp_vel_posx = 1.0
    kp_vel_posy = 1.0
    kp_w_theta = 1.0
    kp_virtual_ey = 0.5
    omega_max = 1.0

    def __init__(self):
        super().__init__("dog_tracker")
        self.create_subscription(Float32MultiArray, self.plan_topic, self.plan_cb, 1)
        self.create_subscription(Odometry, self.odom_topic, self.odom_cb, 2)
        self.cmd_pub = self.create_publisher(TwistStamped, self.cmd_vel_topic, 2)
        self.path_pub = self.create_publisher(Path, "dog_tracker_path", 1)
        self.timer = self.create_timer(1.0 / self.control_hz, self.control_cb)

        self.pose = None
        self.pose_stamp = None
        self.plan_received_time = None
        self.T_time = None
        self.T_pos = None
        self.T_yaw = None
        self.pos_spl = None
        self.yaw_spl = None
        self._last = {}

    def warn_throttle(self, period_s: float, key: str, msg: str):
        now = time.monotonic()
        last = self._last.get(key)
        if last is None or (now - last) >= period_s:
            self._last[key] = now
            self.get_logger().warning(msg)

    def odom_cb(self, msg: Odometry):
        self.pose_stamp = self.get_clock().now()
        pos = msg.pose.pose.position
        yaw = quat_msg_to_yaw(msg.pose.pose.orientation)
        self.pose = np.array([pos.x, pos.y, yaw], dtype=float)

    def plan_cb(self, msg: Float32MultiArray):
        arr = np.asarray(msg.data, dtype=float)
        if len(arr) < 2:
            self.get_logger().error("Received malformed dog plan.")
            return

        delta_t = float(arr[0])
        n_steps = int(round(arr[1]))
        expected = 2 + 3 * n_steps
        if len(arr) != expected:
            self.get_logger().error(f"Dog plan has wrong length {len(arr)} != {expected}.")
            return

        data = arr[2:].reshape(n_steps, 3)
        self.T_time = np.arange(n_steps, dtype=float) * delta_t
        self.T_pos = data[:, 0:2]
        self.T_yaw = np.unwrap(data[:, 2])
        self.pos_spl = None
        self.yaw_spl = None
        if n_steps >= 2:
            from scipy.interpolate import CubicSpline

            self.pos_spl = CubicSpline(self.T_time, self.T_pos, axis=0, extrapolate=False)
            self.yaw_spl = CubicSpline(self.T_time, self.T_yaw, axis=0, extrapolate=False)
        self.plan_received_time = self.get_clock().now()
        self.publish_path()

    def query_plan(self, elapsed_s: float):
        if self.T_time is None or self.pos_spl is None:
            return None
        t = np.clip(elapsed_s, self.T_time[0], self.T_time[-1])
        pos = np.asarray(self.pos_spl(t))
        vel = np.asarray(self.pos_spl.derivative(1)(t))
        yaw = float(self.yaw_spl(t))
        yawdot = float(self.yaw_spl.derivative(1)(t))
        return pos, vel, yaw, yawdot

    def control_cb(self):
        if self.pose is None or self.pose_stamp is None:
            self.warn_throttle(1.0, "no_pose", "Dog tracker has no pose yet.")
            return
        if self.plan_received_time is None:
            self.warn_throttle(1.0, "no_plan", "Dog tracker has no plan yet.")
            return

        now = self.get_clock().now()
        pose_age = (now - self.pose_stamp).nanoseconds * 1e-9
        if pose_age > self.pose_timeout_s:
            self.warn_throttle(1.0, "pose_timeout", f"Dog pose too old ({pose_age:.2f}s); stopping.")
            self.pub_twist(np.zeros(2), 0.0)
            return

        elapsed_s = (now - self.plan_received_time).nanoseconds * 1e-9
        queried = self.query_plan(elapsed_s)
        if queried is None:
            return
        pW_tgt, vW_tgt, yaw_tgt, yawdot_B = queried

        pW = self.pose[:2]
        yaw = self.pose[2]
        if self.pose_shift is not None:
            mat = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            pW = pW + mat @ self.pose_shift

        errposW = pW_tgt - pW
        R_WB = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
        errposB = R_WB @ errposW
        yaw_tgt_virtual = yaw_tgt + np.arctan(self.kp_virtual_ey * errposB[1])
        err_yaw_virtual = angular_distance(yaw_tgt_virtual, yaw)

        vB_tgt = R_WB @ vW_tgt
        vel_x_cmd = vB_tgt[0] + self.kp_vel_posx * errposB[0]
        vel_y_cmd = vB_tgt[1] + self.kp_vel_posy * errposB[1]
        omega_cmd = yawdot_B + self.kp_w_theta * np.sin(err_yaw_virtual)
        omega_cmd = float(np.clip(omega_cmd, -self.omega_max, self.omega_max))
        self.pub_twist(np.array([vel_x_cmd, vel_y_cmd]), omega_cmd)
        self.publish_path()

    def pub_twist(self, v_cmd: np.ndarray, yaw_cmd: float):
        twist_msg = Twist()
        twist_msg.linear.x = float(v_cmd[0])
        twist_msg.linear.y = float(v_cmd[1])
        twist_msg.angular.z = float(yaw_cmd)
        header = Header(stamp=self.get_clock().now().to_msg())
        self.cmd_pub.publish(TwistStamped(twist=twist_msg, header=header))

    def publish_path(self):
        if self.T_time is None or self.T_pos is None or self.plan_received_time is None:
            return
        now = self.get_clock().now()
        elapsed_s = (now - self.plan_received_time).nanoseconds * 1e-9
        start = np.clip(elapsed_s, self.T_time[0], self.T_time[-1])
        end = min(start + self.viz_duration, self.T_time[-1])
        T_query = np.linspace(start, end, num=32)
        T_pos = np.asarray(self.pos_spl(T_query))
        T_yaw = np.asarray(self.yaw_spl(T_query))
        header = Header(stamp=now.to_msg(), frame_id="world")
        poses = []
        for pos, yaw in zip(T_pos, T_yaw):
            pose = PoseStamped()
            pose.header = header
            pose.pose.position.x = float(pos[0])
            pose.pose.position.y = float(pos[1])
            pose.pose.orientation.z = float(np.sin(0.5 * yaw))
            pose.pose.orientation.w = float(np.cos(0.5 * yaw))
            poses.append(pose)
        self.path_pub.publish(Path(header=header, poses=poses))


def main(args=None):
    rclpy.init(args=args)
    node = DogTracker()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
