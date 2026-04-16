#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node

from crazyflie_interface.msg import CFTraj, CFTrajArray


class InputLemniscate(Node):
    def __init__(self):
        super().__init__("input_lemniscate")
        self.publisher_ = self.create_publisher(CFTrajArray, "traj_input", 10)
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.declare_parameter("cf_ids", [1])
        self.declare_parameter("delta_t", 0.1)
        self.declare_parameter("n_steps", 20)
        self.declare_parameter("speed", 0.8)
        self.declare_parameter("size", 1.0)
        self.declare_parameter("height", 0.0)

        self.cf_ids = [int(v) for v in self.get_parameter("cf_ids").value]
        self.delta_t = float(self.get_parameter("delta_t").value)
        self.n_steps = int(self.get_parameter("n_steps").value)
        self.speed = float(self.get_parameter("speed").value)
        self.size = float(self.get_parameter("size").value)
        self.height = float(self.get_parameter("height").value)

        self.start_time = None
        self.publish_count = 0

    def get_pos_traj(self, time_since_start: float) -> np.ndarray:
        n_dense = 16384
        theta = np.linspace(0.0, 2.0 * np.pi, n_dense) + np.pi / 2.0
        denom = 1.0 + np.sin(theta) ** 2
        T_x = -self.size * (np.cos(theta) / denom)
        T_y = -self.size * (np.sin(theta) * np.cos(theta) / denom)

        dx = np.diff(T_x)
        dy = np.diff(T_y)
        ds = np.sqrt(dx**2 + dy**2)
        s_dense = np.concatenate(([0.0], np.cumsum(ds)))
        total_length = s_dense[-1]

        T_times = np.arange(self.n_steps, dtype=float) * self.delta_t + time_since_start
        s_query = np.mod(self.speed * T_times, total_length)
        xq = np.interp(s_query, s_dense, T_x)
        yq = np.interp(s_query, s_dense, T_y)
        zq = np.full(self.n_steps, self.height, dtype=float)
        return np.stack([xq, yq, zq], axis=1)

    def timer_callback(self):
        now = self.get_clock().now()
        if self.start_time is None:
            self.start_time = now

        time_since_start = (now - self.start_time).nanoseconds * 1e-9
        pos_traj = self.get_pos_traj(time_since_start)
        yaw_traj = np.zeros(self.n_steps, dtype=float)

        trajs = []
        stamp_msg = now.to_msg()
        for _cf_id in self.cf_ids:
            trajs.append(
                CFTraj(
                    stamp=stamp_msg,
                    pos_traj=pos_traj.flatten().tolist(),
                    yaw_traj=yaw_traj.tolist(),
                    n_steps=self.n_steps,
                    delta_t=self.delta_t,
                )
            )

        msg = CFTrajArray(stamp=stamp_msg, cf_ids=self.cf_ids, trajs=trajs)
        self.publisher_.publish(msg)
        self.publish_count += 1
        self.get_logger().info(f"Published lemniscate batch {self.publish_count}", throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = InputLemniscate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
