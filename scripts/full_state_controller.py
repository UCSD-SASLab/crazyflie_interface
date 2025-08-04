#!/usr/bin/env python3
import rclpy
import numpy as np
import rowan
from crazyflie_interface_py.template_controller import TemplateController
# numpy logging only 2 digits
np.set_printoptions(precision=2, suppress=True, floatmode='fixed')

MODE = ["hover", "figure8", "circle"][2]


# u: u[0] = p_x, u[1] = p_y, u[2] = p_z, u[3] = v_x, u[4] = v_y, u[5] = v_z, u[6:10] = quaternion, u[10] = omega_x, u[11] = omega_y, u[12] = omega_z, 
#    u[13] = acc_x, u[14] = acc_y, u[15] = acc_z
class FullStateController(TemplateController):
    def __init__(self, node_name='lqr_controller'):
        self.control_publisher_topic = 'cf_interface/control_full_state'
        super().__init__(node_name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self._ros_parameters = self._param_to_dict(self._parameters)
        robots = self._ros_parameters.get('robots', {})
        self.get_logger().info(f"Robots: {robots}")
        self.nbr_robots = len(robots)
        self.get_logger().info(f"Number of robots: {self.nbr_robots}")
        self.start_controller()
        self.iteration = 0

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
        u = np.zeros((states.shape[0], 16))
        u[:, 9] = 1.0  # w -> zero rotation by default (unit quaternion)
        if MODE == "hover":
            for i, state in enumerate(states):
                u[i, 0:3] = np.array([1.5 * i, 1.5 * i, 1.0])  # goal position for each robot
        elif MODE == "circle":
            for i, state in enumerate(states):
                angle = 2 * np.pi * self.iteration / 1000
                radius = 1.5
                u[i, 0:3] = np.array([radius * np.cos(angle), radius * np.sin(angle), 1.0 + 1.0 * i])
        else:
            raise NotImplementedError(f"Mode {MODE} not implemented")
        self.iteration += 1
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
    controller = FullStateController()
    rclpy.spin(controller)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
