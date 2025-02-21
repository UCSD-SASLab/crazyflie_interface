#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
from crazyflie_interface.srv import Command
from crazyflie_interfaces.srv import Takeoff, Land, NotifySetpointsStop
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from example_interfaces.msg import Float32MultiArray


class CfInterface(Node):
    def __init__(self, node_name='cf_interface'):
        super().__init__(node_name)
        # Create high level command interface for taking off, landing and calibrating
        self.create_service(Command, 'cf_interface/command', self.handle_command)
        self.in_flight = False
        self.takeoff_service = self.create_client(Takeoff, 'cf231/takeoff')
        self.takeoff_service.wait_for_service()
        self.land_service = self.create_client(Land, 'cf231/land')
        self.land_service.wait_for_service()

        self.notify_setpointstop_service = self.create_client(NotifySetpointsStop, 'cf231/notify_setpoints_stop')
        self.notify_setpointstop_service.wait_for_service()

        self.zero_control_out_msg = Twist()
        self.zero_control_out_msg.linear.x = 0.0
        self.zero_control_out_msg.linear.y = 0.0
        self.zero_control_out_msg.linear.z = 0.0
        self.zero_control_out_msg.angular.z = 0.0

        # State sub/pub
        # get backend from parameter server
        self.declare_parameter("backend", rclpy.Parameter.Type.STRING)
        self.backend = self.get_parameter("backend").value
        self.get_logger().info("Backend: {}".format(self.backend))
        if self.backend in ["cflib", "sim"]:
            self.get_logger().info("Subscribing to cf231/odom")
            self.create_subscription(Odometry, 'cf231/odom', self.callback_state, 10)
        else: 
            raise NotImplementedError("Backend not yet supported")
        self.state_publisher = self.create_publisher(Float32MultiArray, 'cf_interface/state', 10)

        # Control sub/pub
        self.create_subscription(Float32MultiArray, 'cf_interface/control', self.callback_control, 10)
        if self.backend in ["cflib", "sim"]:
            self.low_level_controller_pub = self.create_publisher(Twist, 'cf231/cmd_vel_legacy', 10)
        else:
            raise NotImplementedError("Backend not yet supported")

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
                req.height = 0.5
                req.duration = rclpy.duration.Duration(seconds=2.0).to_msg()
                self.takeoff_service.call_async(req)
                self.takeoff_timer = self.create_timer(5.0, self.toggle_in_flight)
                response.success = True
        elif request.command == "land":
            if not self.in_flight:
                response.success = False
                response.message = "Not in flight"
            else:
                # 1. Stop sending low level control commands
                self.toggle_in_flight()
                # 2. Inform drone of no more low level commands
                req = NotifySetpointsStop.Request()
                req.group_mask = 0 
                req.remain_valid_millisecs = 10
                self.notify_setpointstop_service.call_async(req)
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
    
    def toggle_in_flight(self):
        if not self.in_flight:
            # Initialize low level controller
            for _ in range(5):
                self.low_level_controller_pub.publish(self.zero_control_out_msg)
            self.destroy_timer(self.takeoff_timer)
        self.in_flight = not self.in_flight

    def callback_state(self, msg):
        # Depends on the type of message received 
        # TODO: Check how it works to interface with pybullet_drones in ros?
        if isinstance(msg, Odometry):
            pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
            vel = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
            quat = np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, 
                             msg.pose.pose.orientation.w])
            omega = np.array([msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z])
            self.state = np.concatenate((pos, vel, quat, omega))
            self.state_publisher.publish(Float32MultiArray(data=self.state))
        else:
            raise NotImplementedError("Message type not yet supported")
        
    def callback_control(self, msg):
        assert isinstance(msg, Float32MultiArray) and len(msg.data) == 4
        # Convert to units for the drone
        if not self.in_flight:
            return
        control = self.convert_and_clip_control(np.array(msg.data))
        # Publish control
        control_msg = Twist()
        control_msg.linear.y = float(control[0])
        control_msg.linear.x = float(control[1])
        control_msg.angular.z = float(control[2])
        control_msg.linear.z = float(control[3])
        self.low_level_controller_pub.publish(control_msg)

    def convert_and_clip_control(self, control_model):
        # Convert control model to drone control
        control_drone = control_model.copy()
        control_drone[:3] = np.degrees(control_drone[:3])
        control_drone[:2] = np.clip(control_drone[:2], -90, 90)  # No clipping on yaw rate
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
