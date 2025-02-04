#!/usr/bin/env python3
import rclpy
import numpy as np
import rowan
from crazyflie_interface_py.template_controller import TemplateController


class LQRController(TemplateController):
    def __init__(self, node_name='lqr_controller'):
        super().__init__(node_name)
        gain_matrix = np.zeros((4, 7))
        gain_matrix[0, 1] = 0.2  # y -> roll 0.2 orig
        gain_matrix[0, 4] = 0.2  # v_y -> roll 0.2 orig
        gain_matrix[1, 0] = 0.2  # x -> pitch 0.2 
        gain_matrix[1, 3] = 0.2  # v_x -> pitch 0.2
        gain_matrix[2, 6] = 2.0  # yaw -> yaw_dot 2.0
        gain_matrix[3, 2] = -10.0 # z -> thrust -10 orig
        gain_matrix[3, 5] = -10.0  # v_z -> thrust -10 orig
        self.gain_matrix = gain_matrix

        self.u_hover = np.array([0.0, 0.0, 0.0, 12.0])

        new_goal_frequency = 0.1
        # Timer for generating new goal
        #self.goal_timer = self.create_timer(1.0 / new_goal_frequency, self.generate_random_goal)
        self.goal_position = np.array([0.0, 2.0, 2.0])  # Initial goal
        self.start_controller()

    def generate_random_goal(self):
        p_x = np.random.uniform(-3.5, 3.5)
        p_y = np.random.uniform(-1.5, 1.5)
        p_z = np.random.uniform(0.5, 2.2)
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

'''
class LQRController(TemplateController):
    def __init__(self, node_name='lqr_controller'):
        super().__init__(node_name)

        #system dynamics 
        self.dynamics = self.DoubleInt 
        self.target_y = None
        self.target_z = None 

        self.nominal_control = lambda state: np.zeros(2) 
          


class DoubleInt: # this is where the euler approximation of the state is computed. 
            ndim = 4 # (VARIABLE)
            control_dim = 2     # (VARIABLE)
            def __init__(self):
                self.umin = np.array([-1.0, 5.0])  # (VARIABLE)
                self.umax = np.array([1.0, 14.0])    # (VARIABLE)
                self.dt = 0.01  # Time step    
            
            def step(self, state, control):
                print("Control dimensions inside step:", control.shape)
                return state + self.dt * self(state, control)
                print("State dimensions after dynamics computation (step):", result.shape)
            
            def __call__(self, state, control):
                print("State dimensions in __call__:", state.shape)
                print("Control dimensions in __call__:", control.shape)
                result = self.open_loop_dynamics(state) + self.control_jacobian(state) @ control  # (VARIABLE)
                print("Resulting dimensions in __call__:", result.shape)
                return result

            def open_loop_dynamics(self, state):
                out = np.zeros_like(state)
                out[..., 0] = state[..., 2]  # (VARIABLE)
                out[..., 1] = state[..., 3]  # (VARIABLE)
                out[..., 3] = -9.81
                print("Open-loop dynamics output dimensions:", out.shape)
                return out

            def control_jacobian(self, state):
                out = np.zeros((*state.shape[:-1], self.ndim, self.control_dim))
                out[..., 2, 0] = 9.81  # (VARIABLE) Roll affects y_dot 
                out[..., 3, 1] = 1.0  # (VARIABLE) Thrust affects z_dot
                print("Control Jacobian shape:", out.shape)
                return out

            self.target_y = float(input("enter y target"))
            self.target_z = float(input("enter z target"))
            nominal_control = lambda state: np.clip(
                np.array([
                    - 1.0 * (state[0] - self.target_y) - 2.0 * state[2],  # Kept original y-target (6.0), adjusted gains
                    9.81 - 0.75 * (state[1] - self.target_z) - 1.5 * state[3]  # Kept original z-target (6.0), increased damping
                ]),
                dynamics.umin,
                dynamics.umax   
                )

'''
