#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
from example_interfaces.msg import Float32MultiArray
from crazyflie_interface.msg import StateStamped
# numpy logging only 2 digits
np.set_printoptions(precision=2, suppress=True, floatmode='fixed')

class TemplateController(Node):
    def __init__(self, node_name='template_controller', controller_rate=50.0, allow_undeclared_parameters=False, automatically_declare_parameters_from_overrides=False):
        super().__init__(node_name, allow_undeclared_parameters=allow_undeclared_parameters, automatically_declare_parameters_from_overrides=automatically_declare_parameters_from_overrides)
        self.controller_rate = controller_rate
        # Subclasses should call start_controller() at the end of their __init__ method
        self.state = None

        if not hasattr(self, 'control_publisher_topic'):
            self.control_publisher_topic = 'cf_interface/control'
        if not hasattr(self, 'state_subscriber_topic'):
            self.state_subscriber_topic = 'cf_interface/state'

        self.control_pub = self.create_publisher(Float32MultiArray, self.control_publisher_topic, 1)
        self.create_subscription(StateStamped, self.state_subscriber_topic, self.callback_state, 1)

    def start_controller(self):
        self.create_timer(1.0 / self.controller_rate, self.publish_control)
    
    def callback_state(self, msg):
        # only log every x seconds
        # self.get_logger().info(f"Received state: {np.array(msg.data)}", throttle_duration_sec=0.1)
        self.state = np.array(msg.data)

    def __call__(self, state):
        raise NotImplementedError("Must be subclassed")

    def publish_control(self):
        if self.state is None:
            return
        u = list(self(self.state))
        control_msg = Float32MultiArray()
        control_msg.data = u
        self.control_pub.publish(control_msg) # array of size 4 -> 8
