#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
from crazyflie_interface.srv import Command
from crazyflie_interfaces.srv import Takeoff, Land, NotifySetpointsStop
from crazyflie_interfaces.srv import Arm
from crazyflie_interfaces.msg import FullState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from example_interfaces.msg import Float32MultiArray
from std_msgs.msg import Bool
from crazyflie_interface.msg import StateStamped
from functools import partial
import rowan
from collections import deque

MODE = "both"
CONTROL_MODE = "control"   # "full_state" for 12d or "control" for 20d only
ANGULAR_VEL_CALC_METHOD = ["direct", "direct_averaged", "finite_difference"][1]  # How to get angular velocity from orientation
AVERAGE_WINDOW_SIZE = 30  # Only used if ANGULAR_VEL_CALC_METHOD is "direct_averaged"
CLIP_THETA_OMEGA = True # clips thetas to
ANGLE_MAX, ANGLE_VEL_MAX = 0.35, 2.0

class CfInterface(Node):
    def __init__(self, node_name='cf_interface'):
        super().__init__(node_name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        # Add parameter for YAML config path
        self._ros_parameters = self._param_to_dict(self._parameters)

        robots = self._ros_parameters.get('robots', {})    
        self.crazyflie_names = [name for name, data in robots.items() if data.get('enabled', False)]
        self.uris = [data.get('uri') for name, data in robots.items() if data.get('enabled', False)]
        self.get_logger().info(f"Loaded robots: {self.crazyflie_names}")
        # Create high level command interface for taking off, landing and calibrating
        self.create_service(Command, 'cf_interface/command', self.handle_command)
        self.in_flight = False
        self.takeoff_service = self.create_client(Takeoff, 'all/takeoff')
        # self.takeoff_service.wait_for_service()
        self.land_service = self.create_client(Land, 'all/land')
        # self.land_service.wait_for_service()
        self.get_logger().info(f"Created takeoff and land services for {self.crazyflie_names}")
        self.state = [None for _ in self.crazyflie_names[:2]]

        self.time_init = None
        self.get_logger().info(f"URIs: {self.uris}")
        # self.uris = [4]
        # Create separate service for each robot
        self.notify_setpointstop_services = {}
        for name in self.crazyflie_names:
            if name in ["cf233", "cf234"]:
                continue
            self.notify_setpointstop_services[name] = self.create_client(NotifySetpointsStop, f"{name}/notify_setpoints_stop")
            # self.notify_setpointstop_services[name].wait_for_service()
            self.get_logger().info(f"Created notify_setpoints_stop service for {name}")

        self.zero_control_out_msg = Twist()
        self.zero_control_out_msg.linear.x = 0.0
        self.zero_control_out_msg.linear.y = 0.0
        self.zero_control_out_msg.linear.z = 0.0
        self.zero_control_out_msg.angular.z = 0.0

        # Setup queue if using direct_averaged method
        if ANGULAR_VEL_CALC_METHOD == "direct_averaged":
            self.omega_queues = {uri: deque(maxlen=AVERAGE_WINDOW_SIZE) for uri in self.uris}

        # State sub/pub
        self.get_logger().info(f"Setting up state publisher for {self.crazyflie_names}")
        # get backend from parameter server
        self.backend = self._ros_parameters['backend']
        self.get_logger().info("Backend: {}".format(self.backend))
        self.get_logger().info(f"crazyflie name: {self.crazyflie_names}")
        if self.backend in ["cflib", "sim"]:
            for i, cf_name in enumerate(self.crazyflie_names):
                if cf_name not in ["cf233", "cf234"]:
                    uri = self.uris[i]
                    self.get_logger().info(f"Creating subscription for {cf_name} with uri {uri}")
                    self.create_subscription(Odometry, f"{cf_name}/odom", partial(self.callback_state, uri=uri), 1)
        else: 
            raise NotImplementedError("Backend not yet supported")
        self.state_publisher = self.create_publisher(StateStamped, 'cf_interface/state', 1)
        self.flight_status_publisher = self.create_publisher(Bool, 'cf_interface/flight_status', 1)
        self.flight_status_callback = self.create_timer(1.0, self.callback_flight_status)

        # Control sub/pub
        if self.backend in ["cflib", "sim"]:
            if CONTROL_MODE == "control":
                self.cmd_vel_publishers = {}
                self.get_logger().info(f"Setting up cmd_vel publishers for: {self.crazyflie_names}")
                for name in self.crazyflie_names:
                    if name not in ["cf233", "cf234"]:
                        self.cmd_vel_publishers[name] = self.create_publisher(
                            Twist, f"{name}/cmd_vel_legacy", 1
                        )
                    self.get_logger().info(f"Created cmd_vel publisher for {name}")
                    # TODO: Add an else option for cmd_full_state and create its associated publisher
            elif CONTROL_MODE == "full_state":
                self.cmd_full_state_publishers = {}
                self.get_logger().info(f"Setting up cmd_full_state publishers for: {self.crazyflie_names}")
                for name in self.crazyflie_names:
                    if name not in ["cf233", "cf234"]:
                        self.cmd_full_state_publishers[name] = self.create_publisher(
                            FullState, f"{name}/cmd_full_state", 1
                        )
                self.FullStateMsg = FullState()
                self.FullStateMsg.header.frame_id = '/world'
        else:
            raise NotImplementedError("Backend not yet supported")
        self.arm_service = self.create_client(Arm, 'all/arm')
        req = Arm.Request()
        req.arm = True
        self.arm_service.wait_for_service()
        self.arm_service.call_async(req)
        self.get_logger().info("Arming Crazyflie")        
        if CONTROL_MODE == "control":
            self.create_subscription(Float32MultiArray, 'cf_interface/control', self.callback_control, 1)
        elif CONTROL_MODE == "full_state":
            self.create_subscription(Float32MultiArray, 'cf_interface/control_full_state', self.callback_control_full_state, 1)
        else:
            raise NotImplementedError("Control mode not yet supported")
        self.get_logger().info(f"Control mode: {CONTROL_MODE}")
        self.state_is_publishing = False
        self.state_publisher_timer = self.create_timer(0.01, self.state_publisher_callback)

        # self.last_state is dict of nones for uri keys
        self.last_euler_state = {uri: None for uri in self.uris}
        self.last_time = {uri: None for uri in self.uris}

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

    def callback_flight_status(self):
        flight_status_msg = Bool()
        flight_status_msg.data = self.in_flight
        self.flight_status_publisher.publish(flight_status_msg)

    def handle_command(self, request, response):
        """
        Processes service that handles high level requests to the drone
        """
        if request.command == "takeoff":
            if self.in_flight:
                response.success = False
                response.message = "Already in flight"
            else:
                req = Takeoff.Request()
                req.group_mask = 0  # all crazyflies
                req.height = 1.0
                req.duration = rclpy.duration.Duration(seconds=2.0).to_msg()
                self.takeoff_service.call_async(req)
                self.takeoff_timer = self.create_timer(5.0, self.toggle_post_takeoff)
                response.success = True
        elif request.command == "land":
            if not self.in_flight:
                response.success = False
                response.message = "Not in flight"
            else:
                # 1. Stop sending low level control commands
                self.toggle_to_land()
                # 2. Inform drone of no more low level commands
                req = NotifySetpointsStop.Request()
                req.group_mask = 0 
                req.remain_valid_millisecs = 10
                for name in self.crazyflie_names:
                    if name in ["cf233", "cf234"]:
                        continue
                    self.notify_setpointstop_services[name].call_async(req)
                # 3. Send land command (twice to ensure it is not missed)
                req = Land.Request()
                req.group_mask = 0
                req.height = 0.05
                req.duration = rclpy.duration.Duration(seconds=3.0).to_msg()
                for _ in range(2):
                    self.land_service.call_async(req)
                    rclpy.spin_once(self, timeout_sec=0.1)
                response.success = True
        elif request.command == "calibrate":
            response.success = False
            response.message = "Calibration not supported yet"   
        else:
            response.success = False
            response.message = "Unknown command, {}".format(request.command)
        return response
    
    def toggle_to_land(self):
        self.in_flight = False
        self.callback_flight_status()
        self.get_logger().info(f"In flight: {self.in_flight}")

    def toggle_post_takeoff(self):
        if not self.in_flight:
            # Initialize low level controller
            if CONTROL_MODE == "control":
                for _ in range(2):
                    for i, name in enumerate(self.crazyflie_names):
                        if name in ["cf233", "cf234"]:
                            continue
                        self.cmd_vel_publishers[name].publish(self.zero_control_out_msg)
            self.destroy_timer(self.takeoff_timer)
        self.in_flight = True
        self.callback_flight_status()

    def callback_state(self, msg, uri):
        # Depends on the type of message received 
        # TODO: Check how it works to interface with pybullet_drones in ros?
        # TODO Annie: for loop over all crazyflies
        if isinstance(msg, Odometry):
            # On first message from any robot, set time_init
            if all(s is None for s in self.state):  # only at first time point
                self.time_init = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
            vel = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
            quat = np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, 
                             msg.pose.pose.orientation.w])
            quat_mod = np.array([quat[3], quat[0], quat[1], quat[2]])  # [qw, qx, qy, qz]e
            euler_angles = rowan.to_euler(quat_mod, "xyz")
            # The euler angles here are flipped compared to the drone convention
            roll = -euler_angles[0]   # θ_y  (post sign change: +roll = positive y acceleration)
            pitch = euler_angles[1]  # θ_x  (without sign change: +pitch = positive x acceleration)
            yaw = euler_angles[2]
            euler_xyz = np.array([pitch, roll, yaw])  # [θ_x, θ_y, θ_z] in radians

            if ANGULAR_VEL_CALC_METHOD == "direct":
                omega = np.array([msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z])
                if self.backend == "sim":
                    omega = omega
                else:
                    omega = omega * np.pi / 180.0  # Convert to rad/s 
            
            elif ANGULAR_VEL_CALC_METHOD == "finite_difference":
                time_now = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                if self.last_euler_state[uri] is None:
                    omega = np.array([0.0, 0.0, 0.0])
                else:
                    # convert quaternion to euler
                    omega = (euler_xyz - self.last_euler_state[uri]) / (time_now - self.last_time[uri])  # Assuming 100 Hz update rate
                
                if CLIP_THETA_OMEGA:
                    euler_xyz[0:2] = np.clip(euler_xyz[0:2], -ANGLE_MAX, ANGLE_MAX)

                self.last_euler_state[uri] = euler_xyz
                self.last_time[uri] = time_now

            elif ANGULAR_VEL_CALC_METHOD == "direct_averaged":
                omega_curr = np.array([msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z])
                if self.backend == "sim":
                    omega_curr = omega_curr
                else:
                    omega_curr = omega_curr * np.pi / 180.0

                if CLIP_THETA_OMEGA:
                    omega_curr = np.clip(omega_curr, -ANGLE_VEL_MAX, ANGLE_VEL_MAX)
                
                self.omega_queues[uri].append(omega_curr)
                omega = np.mean(np.stack(self.omega_queues[uri]), axis=0)
            
            else: 
                raise NotImplementedError("Angular velocity calculation method not yet supported: {}".format(ANGULAR_VEL_CALC_METHOD))
            
            if CLIP_THETA_OMEGA:
                euler_xyz[0:2] = np.clip(euler_xyz[0:2], -ANGLE_MAX, ANGLE_MAX)
                omega = np.clip(omega, -ANGLE_VEL_MAX, ANGLE_VEL_MAX)
                
                # euler_xyz_remod = np.array([-euler_xyz[1], euler_xyz[0], euler_xyz[2]])
                quat_mod_clipped = rowan.from_euler(-euler_xyz[1], euler_xyz[0], euler_xyz[2], "xyz")
                quat = np.array([quat_mod_clipped[1], quat_mod_clipped[2], quat_mod_clipped[3], quat_mod_clipped[0]])
            
            # For the uri need the index of the crazyflie
            if uri not in self.uris:
                self.get_logger().error("URI {} not found in uris list".format(uri))
                return
            # Find index of uri in self.uris
            index = self.uris.index(uri)
            # self.get_logger().info(f"{omega}", throttle_duration_sec=0.2)
            self.state[index] = np.concatenate((pos, vel, quat, omega))
            self.timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            
        else:
            raise NotImplementedError("Message type not yet supported")
    
    def state_publisher_callback(self):
        # If any state is None, return
        if any(s is None for s in self.state[:2]):  # only check first two robots
            self.get_logger().info(f"State not yet initialized", throttle_duration_sec=1.0)
            return
        if not self.state_is_publishing:
            self.get_logger().info(f"Started publishing state")
            self.state_is_publishing = True
        
        # self.get_logger().info(f"state:" f"{np.array(self.state).flatten()}", throttle_duration_sec=0.1)
        state_msg = StateStamped()
        # Turn list into flattened array
        state = np.array(self.state).flatten()
        state_msg.data = state.tolist()
        state_msg.time = self.timestamp - self.time_init
        self.state_publisher.publish(state_msg)

    # This function sends an Odom message to the crazyflies with a fixed state (position x y z) for each drone
    def callback_control_full_state(self, msg):
        if not self.in_flight:
            return
        for i, name in enumerate(self.crazyflie_names):
            if name in ["cf233", "cf234"]:
                continue
            control = np.array(msg.data[16*i:16*(i+1)])
            # self.get_logger().info(f"CONTROL IN cf_interface: {control}")
            ctrl_msg = self.FullStateMsg
            ctrl_msg.header.stamp = self.get_clock().now().to_msg()
            ctrl_msg.pose.position.x = float(control[0])
            ctrl_msg.pose.position.y = float(control[1])
            ctrl_msg.pose.position.z = float(min(max(control[2], 0.2), 2.2))  # To be changed if desired
            ctrl_msg.twist.linear.x = float(control[3])
            ctrl_msg.twist.linear.y = float(control[4])
            ctrl_msg.twist.linear.z = float(control[5])
            # ctrl_msg.twist.linear.x = 0.0
            # ctrl_msg.twist.linear.y = 0.0
            # ctrl_msg.twist.linear.z = 0.0
            # ctrl_msg.pose.orientation.w = float(control[6])
            # ctrl_msg.pose.orientation.x = float(control[7])
            # ctrl_msg.pose.orientation.y = float(control[8])
            # ctrl_msg.pose.orientation.z = float(control[9])
            ctrl_msg.pose.orientation.x = 0.
            ctrl_msg.pose.orientation.y = 0.
            ctrl_msg.pose.orientation.z = 0.
            ctrl_msg.pose.orientation.w = 1.
            ctrl_msg.twist.angular.x = float(control[10])
            ctrl_msg.twist.angular.y = float(control[11])
            ctrl_msg.twist.angular.z = float(control[12])
            # ctrl_msg.twist.angular.x = 0.
            # ctrl_msg.twist.angular.y = 0.
            # ctrl_msg.twist.angular.z = 0.
            ctrl_msg.acc.x = float(control[13])
            ctrl_msg.acc.y = float(control[14])
            ctrl_msg.acc.z = float(control[15])

            # self.get_logger().info(f"IN CF_INTERFACE: (x,y,z)={(float(control[0]), float(control[1]), float(min(max(control[2], 0.2), 2.2)))}")

            self.cmd_full_state_publishers[name].publish(ctrl_msg)
            # cmd_full_state: pose (3d position + quaternion orientation), velocity (3d linear + 3d angular), acceleration (3d linear)
            # full control size: 16d (pos, vel, quat, omega, acc)


    def callback_control(self, msg):
        num_robots = min(len(self.crazyflie_names), 2)
        assert len(msg.data) == 4 * num_robots
        if not self.in_flight:
            return
        if MODE == "both":
            for i, name in enumerate(self.crazyflie_names):
                if name in ["cf233", "cf234"]:
                    continue
                control = np.array(msg.data[4*i:4*(i+1)])
                # self.get_logger().info(f"Control for {name}: {control}")
                control = self.convert_and_clip_control(control)
                control_msg = Twist()
                control_msg.linear.y = float(control[0])
                control_msg.linear.x = float(control[1])
                control_msg.angular.z = float(control[2])
                control_msg.linear.z = float(control[3])
                # self.get_logger().info(f"Publishing control for {name}: {control_msg}")
                # self.get_logger().info(f"Publishing control for {name}: {control_msg}", throttle_duration_sec=0.1)
                self.cmd_vel_publishers[name].publish(control_msg)
        elif MODE == "1only":
            control = np.array(msg.data)  # nbr_robots*m 
            control = control[:4]  # Only take first 4 values (only robot 0)
            control = self.convert_and_clip_control(control)
            control_msg = Twist()
            control_msg.linear.y = float(control[0])
            control_msg.linear.x = float(control[1])
            control_msg.angular.z = float(control[2])
            control_msg.linear.z = float(control[3])
            # self.get_logger().info(f"Publishing control for robot 0: {control_msg}", throttle_duration_sec=0.1)
            self.cmd_vel_publishers[self.crazyflie_names[0]].publish(control_msg)
        elif MODE == "2only":
            control = np.array(msg.data)
            control = control[4:]  # Only take last 4 values (only robot 1)
            control = self.convert_and_clip_control(control)
            control_msg = Twist()
            control_msg.linear.y = float(control[0])
            control_msg.linear.x = float(control[1])
            control_msg.angular.z = float(control[2])
            control_msg.linear.z = float(control[3])
            # self.get_logger().info(f"Publishing control for robot 1: {control_msg}", throttle_duration_sec=0.1)
            self.cmd_vel_publishers[self.crazyflie_names[1]].publish(control_msg)
        elif MODE == "2zeros":
            control = np.array(msg.data)
            control = control[:4] # Only take first 4 values (only robot 0)
            control = self.convert_and_clip_control(control)
            control_msg = Twist()
            control_msg.linear.y = float(control[0])
            control_msg.linear.x = float(control[1])
            control_msg.angular.z = float(control[2])
            control_msg.linear.z = float(control[3])
            self.cmd_vel_publishers[self.crazyflie_names[0]].publish(control_msg)
            zero_control_msg = Twist()
            self.cmd_vel_publishers[self.crazyflie_names[1]].publish(zero_control_msg)
        else:
            raise NotImplementedError("Mode not yet supported: {}".format(MODE))

    def convert_and_clip_control(self, control_model):
        # Convert control model to drone control
        control_drone = control_model.copy()
        control_drone[:3] = np.degrees(control_drone[:3])
        control_drone[:2] = np.clip(control_drone[:2], -90, 90)  # No clipping on yaw rate
        control_drone[0] = -control_drone[0]  # Inverting roll for crazyflie
        if self.backend == "sim":
            control_drone[2] = -control_drone[2]  # Inverting yaw rate for simulation
        control_drone[3] = np.clip(control_drone[3] * 4096.0, 10000, 65535)  # Clipping required to function
        # TODO: Clipping if desired (I think this shouldn't be set in the interface, but rather own algorithm)
        return control_drone


def main(args=None):
    rclpy.init(args=args)
    interface_node = CfInterface()
    rclpy.spin(interface_node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
