#!/usr/bin/env python3
import rclpy
import numpy as np
import rowan
from crazyflie_interface_py.template_controller import TemplateController
from std_msgs.msg import Bool
# numpy logging only 2 digits
np.set_printoptions(precision=2, suppress=True, floatmode='fixed')

MODE = ["hover", "figure8", "circle", "6d"][3] 


# u: u[0] = p_x, u[1] = p_y, u[2] = p_z, u[3] = v_x, u[4] = v_y, u[5] = v_z, u[6:10] = quaternion, u[10] = omega_x, u[11] = omega_y, u[12] = omega_z, 
#    u[13] = acc_x, u[14] = acc_y, u[15] = acc_z
class FullStateController(TemplateController):
    def __init__(self, node_name='full_state_controller'):
        self.control_publisher_topic = 'cf_interface/control_full_state'
        super().__init__(node_name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self._ros_parameters = self._param_to_dict(self._parameters)
        robots = self._ros_parameters.get('robots', {})
        self.get_logger().info(f"Robots: {robots}")
        self.nbr_robots = len(robots)
        self.get_logger().info(f"Number of robots: {self.nbr_robots}")
        self.create_subscription(Bool, 'cf_interface/flight_status', self.flight_status_callback, 1)
        self.in_flight = False

        #add following
        self.initial_positions = []
        for cf_name in robots:
            pos = robots[cf_name].get('initial_position', [0.0, 0.0, 0.0])
            self.initial_positions.append(np.array(pos))
        
        # adding the following for 6d
        self.dt = 0.01 # 100 Hz
        self.positions = np.array(self.initial_positions)
        self.velocities = np.zeros((self.nbr_robots, 3))

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
    
    def flight_status_callback(self, msg):
        if msg.data:
            self.in_flight = True
            

    def __call__(self, state):
        states = np.array(state).reshape(self.nbr_robots, -1)
        u = np.zeros((states.shape[0], 16))
        u[:, 9] = 1.0  # w -> zero rotation by default (unit quaternion)

        #trying hover, the following is working
        if MODE == "hover":
            for i in range(self.nbr_robots):
                u[i, 0:3] = self.initial_positions[i] + np.array([0.0, 0.0, 1.0])
                u[i, 3:6] = [0.0, 0.0, 0.0]     # Velocity = zero
                u[i, 6:10] = [0.0, 0.0, 0.0, 1.0]   #Orientation = default
                u[i, 10:16] = [0.0] * 6   # Angular velocity & acceleration = zero
        # original hover - not working properly
        # if MODE == "hover":
        #     for i, state in enumerate(states):
        #         u[i, 0:3] = np.array([1.5 * i, 1.5 * i, 1.0])  # goal position for each robot


        elif MODE == "6d": # working
            dt = self.dt
            def accel_fn(pos, vel, i):
                # Replace with actual logic later; simple test for now
                if self.iteration < 50:
                    return np.array([0.0, 0.0, 0.0])
                else:
                    return np.array([0.2, 0.2, 0.2]) if i == 0 else np.array([0.0, 0.2, 0.0])

            for i in range(self.nbr_robots):
                pos = self.positions[i]
                vel = self.velocities[i]
                acc = accel_fn(pos, vel, i)

                # Runge-Kutta 4th order integration
                k1p, k1v = vel,                    acc
                k2p, k2v = vel + 0.5*k1v*dt,       acc
                k3p, k3v = vel + 0.5*k2v*dt,       acc
                k4p, k4v = vel + k3v*dt,           acc

                new_pos = pos + (dt/6.0)*(k1p + 2*k2p + 2*k3p + k4p)
                new_vel = vel + (dt/6.0)*(k1v + 2*k2v + 2*k3v + k4v)

                self.positions[i] = new_pos
                self.velocities[i] = new_vel

                u[i, 0:3] = new_pos
                u[i, 3:6] = new_vel
                u[i, 6:10] = [0.0, 0.0, 0.0, 1.0]   # Identity quaternion
                u[i, 10:13] = [0.0] * 3             # Angular velocity
                u[i, 13:16] = acc                   # Linear acceleration

                self.get_logger().info(
                    f"[Robot {i}] Pos: {new_pos}, Vel: {new_vel}, Acc: {acc}",
                    throttle_duration_sec=0.5
                )

            if self.in_flight:
                self.iteration += 1


        # elif MODE == "6d":
        #     # Runge Kutta 4th order integration for 6D control
        #     def accel_fn(pos, vel, i):
        #         # Example acceleration function, will be replaced by actual dynamics
        #         if self.iteration % 100 < 50:
        #             return np.array([0.1, 0.1, 0.1])
        #         else:
        #             return np.array([-0.1, -0.1, -0.1])

        #     for i, state in enumerate(states):
        #         pos = state[0:3]
        #         vel = state[3:6]
        #         acc = accel_fn(pos, vel, i)
        #         # self.get_logger().info(f"Robot {i} pos: {pos}, vel: {vel}", throttle_duration_sec=0.5)
        #         dt = self.dt
        #         # Runge-Kutta 4th order integration for velocity and position
        #         k1v = accel_fn(pos, vel, i)
        #         k2v = accel_fn(pos + 0.5 * vel * dt, vel + 0.5 * k1v * dt, i)
        #         k3v = accel_fn(pos + 0.5 * vel * dt, vel + 0.5 * k2v * dt, i)
        #         k4v = accel_fn(pos + vel * dt, vel + k3v * dt, i)
        #         new_vel = vel + (dt / 6.0) * (k1v + 2*k2v + 2*k3v + k4v)
        #         k1p = vel
        #         k2p = vel + 0.5 * k1p * dt
        #         k3p = vel + 0.5 * k2p * dt
        #         k4p = vel + k3p * dt
        #         new_pos = pos + (dt / 6.0) * (k1p + 2*k2p + 2*k3p + k4p)
        #         self.velocities[i] = new_vel
        #         self.positions[i] = new_pos
        #         u[i, 0:3] = new_pos
        #         u[i, 3:6] = new_vel
        #         u[i, 6:10] = [0.0, 0.0, 0.0, 1.0]
        #         u[i, 10:13] = [0.0] * 3
        #         u[i, 13:16] = acc
        #         self.get_logger().info(f"Drone {i} acc: {acc}, vel: {vel}, new_vel: {new_vel}")


            # Euler's integration for 6D control - not fully working yet
            # if i in range(self.nbr_robots):
            #     # if iteration \in [0, 50], [100, 150], [200, 250], ... then hover
            #     if self.iteration % 100 < 50:
            #         acc = np.array([0.1, 0.1, 0.1])
            #     else:
            #         acc = np.array([-0.1, -0.1, -0.1])
            #     self.velocities[i] = states[i, 3:6] + acc * self.dt
            #     self.positions[i] = states[i, 0:3] + states[i, 3:6] * self.dt
            #     u[i, 0:3] = self.positions[i]
            #     u[i, 3:6] = self.velocities[i]
            #     u[i, 6:10] = [0.0, 0.0, 0.0, 1.0] 
            #     u[i, 10:16] = [0.0] * 6

        else:
            raise NotImplementedError(f"Mode {MODE} not implemented")
        if self.in_flight:
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
