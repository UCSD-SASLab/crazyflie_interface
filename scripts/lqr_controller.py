#!/usr/bin/env python3
import rclpy
import numpy as np
import rowan
from crazyflie_interface_py.template_controller import TemplateController


class LQRController(TemplateController):
    def __init__(self, node_name='lqr_controller'):
        super().__init__(node_name)
        gain_matrix = np.zeros((4, 7))
        gain_matrix[0, 1] = 0.2  # y -> roll
        gain_matrix[0, 4] = 0.2  # v_y -> roll
        gain_matrix[1, 0] = 0.2  # x -> pitch
        gain_matrix[1, 3] = 0.2  # v_x -> pitch
        gain_matrix[2, 6] = 2.0  # yaw -> yaw_dot
        gain_matrix[3, 2] = -10.0  # z -> thrust
        gain_matrix[3, 5] = -10.0  # v_z -> thrust
        self.gain_matrix = gain_matrix

        self.u_hover = np.array([0.0, 0.0, 0.0, 12.0])

        new_goal_frequency = 0.1
        # Timer for generating new goal
        self.goal_timer = self.create_timer(1.0 / new_goal_frequency, self.generate_random_goal)
        self.goal_position = np.array([0.0, 0.0, 1.0])  # Initial goal
        self.start_controller()

    def generate_random_goal(self):
        p_x = np.random.uniform(-5.0, 5.0)
        p_y = np.random.uniform(-2.5, 2.5)
        p_z = np.random.uniform(0.5, 2.5)
        self.get_logger().info("New goal: {:.1f}, {:.1f}, {:.1f}".format(p_x, p_y, p_z))
        self.goal_position = np.array([p_x, p_y, p_z])
    
    def __call__(self, state):            
        euler_angles = rowan.to_euler(([state[9], state[6], state[7], state[8]]), "xyz")
        yaw = euler_angles[2]
        near_hover_state = np.concatenate([state[0:6], np.array([yaw])])
        u = self.u_hover + self.gain_matrix @ (near_hover_state - np.concatenate((self.goal_position, np.zeros(4))))
        u[:2] = np.clip(u[:2], -0.2, 0.2)
        u[3] = np.clip(u[3], 4.0, 16.0)
        return u


def main(args=None):
    rclpy.init(args=args)
    controller = LQRController()
    rclpy.spin(controller)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
