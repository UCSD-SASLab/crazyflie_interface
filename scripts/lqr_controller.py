#!/usr/bin/env python3
import rclpy
import numpy as np
import rowan
from crazyflie_interface_py.template_controller import TemplateController
# numpy logging only 2 digits
np.set_printoptions(precision=2, suppress=True, floatmode='fixed')

MODE = "lqr" #"2repeats1", # "hover"  # "1repeats2"  # "2repeats1"  # "hover"

class LQRController(TemplateController):
    def __init__(self, node_name='lqr_controller'):
        super().__init__(node_name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self._ros_parameters = self._param_to_dict(self._parameters)
        robots = self._ros_parameters.get('robots', {})
        self.get_logger().info(f"Robots: {robots}")
        self.nbr_robots = len(robots)
        self.get_logger().info(f"Number of robots: {self.nbr_robots}")
        gain_matrix = np.zeros((4, 7))
        gain_matrix[0, 1] = -0.2  # y -> roll
        gain_matrix[0, 4] = -0.2  # v_y -> roll
        gain_matrix[1, 0] = 0.2  # x -> pitch
        gain_matrix[1, 3] = 0.2  # v_x -> pitch
        gain_matrix[2, 6] = 2.0  # yaw -> yaw_dot
        gain_matrix[3, 2] = -10.0  # z -> thrust
        gain_matrix[3, 5] = -10.0  # v_z -> thrust
        self.gain_matrix = gain_matrix

        self.u_hover = np.array([0.0, 0.0, 0.0, 11.95])

        new_goal_frequency = 0.1
        # Timer for generating new goal
        self.goal_timer = self.create_timer(1.0 / new_goal_frequency, self.generate_random_goal)
        #edit? one drone 
        # self.goal_position = np.array([[1.5, 1.5, 1.0]])  # Initial goal 
        # two drones below
        self.goal_position = np.array([[1.5, 1.5, 1.0], [4.5, 1.5, 1.0]])  # Initial goal  # FIXME: forces fixed number of robots robots -> become more generic
        self.start_controller()

    def generate_random_goal(self):
        p_x = np.random.uniform(-2.0, 2.0, self.nbr_robots)
        p_y = np.random.uniform(-2.0, 2.0, self.nbr_robots)
        p_z = np.random.uniform(0.5, 1.5, self.nbr_robots)

        self.goal_position = np.column_stack([p_x, p_y, p_z])
        self.get_logger().info(f"New goals: {self.goal_position}")

    def _param_to_dict(self, param_ros):
        """
        Turn ROS 2 parameters from the node into a dict
        """
        tree = {}
        for item in param_ros:
            t = tree
            for part in item.split('.'):
                if part == item.split('.')[-1]:
                    t = t.setdefault(part, param_ros[item].value)
                else:
                    t = t.setdefault(part, {})
        return tree

    # debug hover - temporary
    def __call__(self, state):
        states = np.array(state).reshape(self.nbr_robots, -1)
        u = np.zeros((self.nbr_robots, 4))
        if MODE == "hover":
            u[:, 3] = self.u_hover[3]  # thrust
            # self.get_logger().info(f"Sending hover command: {hover_cmd}", throttle_duration_sec=0.5)
            
        
        elif MODE == "2repeats1":
            state = states[0]  # robot0(1)
            euler_angles = rowan.to_euler(([state[9], state[6], state[7], state[8]]), "xyz")
            yaw = euler_angles[2]
            near_hover_state = np.concatenate([state[0:6], np.array([yaw])])
            u0 = self.u_hover + self.gain_matrix @ (near_hover_state - np.concatenate((self.goal_position[0], np.zeros(4))))
            u0[:2] = np.clip(u0[:2], -0.2, 0.2)
            u0[3] = np.clip(u0[3], 4.0, 16.0)
            u[0] = u0
            u[1] = u0

        elif MODE == "1repeats2":
            state = states[1]
            euler_angles = rowan.to_euler(([state[9], state[6], state[7], state[8]]), "xyz")
            yaw = euler_angles[2]
            near_hover_state = np.concatenate([state[0:6], np.array([yaw])])
            u1 = self.u_hover + self.gain_matrix @ (near_hover_state - np.concatenate((self.goal_position[1], np.zeros(4))))
            u1[:2] = np.clip(u1[:2], -0.2, 0.2)
            u1[3] = np.clip(u1[3], 4.0, 16.0)
            u[0] = u1
            u[1] = u1

        elif MODE == "lqr":
            for i, state in enumerate(states):
                euler_angles = rowan.to_euler(([state[9], state[6], state[7], state[8]]), "xyz")
                yaw = euler_angles[2]
                near_hover_state = np.concatenate([state[0:6], np.array([yaw])])
                u[i] = self.u_hover + self.gain_matrix @ (near_hover_state - np.concatenate((self.goal_position[i], np.zeros(4))))
                u[i, :2] = np.clip(u[i, :2], -0.2, 0.2)
                # u[i, 2] = np.clip(u[i, 2], -1.0, 1.0)  # USE WHEN URI=[6,7,9]  # TODO: ST
                u[i, 3] = np.clip(u[i, 3], 4.0, 16.0) 
        elif MODE == "2off":
            state = states[0]  # robot0(1)
            euler_angles = rowan.to_euler(([state[9], state[6], state[7], state[8]]), "xyz")
            yaw = euler_angles[2]
            near_hover_state = np.concatenate([state[0:6], np.array([yaw])])
            u0 = self.u_hover + self.gain_matrix @ (near_hover_state - np.concatenate((self.goal_position[0], np.zeros(4))))
            u0[:2] = np.clip(u0[:2], -0.2, 0.2)
            u0[3] = np.clip(u0[3], 4.0, 16.0)
            u[0] = u0           
                                               
        return u.flatten()
                                               
            

    '''
    # if use the call below, both drones will fly randomly (but not what we wanted) and later both will fly out of bounds
    def __call__(self, state):
        # For 1 robot            
        # euler_angles = rowan.to_euler(([state[9], state[6], state[7], state[8]]), "xyz")
        # yaw = euler_angles[2]
        # near_hover_state = np.concatenate([state[0:6], np.array([yaw])])
        # u = self.u_hover + self.gain_matrix @ (near_hover_state - np.concatenate((self.goal_position, np.zeros(4))))
        # u[:2] = np.clip(u[:2], -0.2, 0.2)
        # u[3] = np.clip(u[3], 4.0, 16.0)
        # return u
        # For 2 robots
        nbr_robots = 2


        states = np.array(state).reshape(nbr_robots, -1)  # FIXME: forces fixed number of robots robots -> become more generic
        u = np.zeros((states.shape[0], 4))
        for i, state in enumerate(states):
            self.get_logger().info(f"State for robot {i}: {state}", throttle_duration_sec=0.1)  # Log every second
            # if i == 1:
            #     break  # just have 0 control for the second robot (temporary)
            euler_angles = rowan.to_euler(([state[9], state[6], state[7], state[8]]), "xyz")
            yaw = euler_angles[2]
            near_hover_state = np.concatenate([state[0:6], np.array([yaw])])
            u[i] = self.u_hover + self.gain_matrix @ (near_hover_state - np.concatenate((self.goal_position[i], np.zeros(4))))
            u[i, :2] = np.clip(u[i, :2], -0.2, 0.2)
            u[i, 3] = np.clip(u[i, 3], 4.0, 16.0)
        flattened_u = u.flatten()
        self.get_logger().info(f"Flattened control input: {flattened_u}", throttle_duration_sec=0.1)
        return u.flatten()
    '''


def main(args=None):
    rclpy.init(args=args)
    controller = LQRController()
    rclpy.spin(controller)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
