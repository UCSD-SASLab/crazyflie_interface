#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
from example_interfaces.msg import Float32MultiArray
from crazyflie_interface.msg import StateStamped


class TemplateController(Node):
    def __init__(self, node_name='template_controller', controller_rate=50.0):
        super().__init__(node_name)
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
        self.state = np.array(msg.data)

    def __call__(self, state):
        raise NotImplementedError("Must be subclassed")

    def publish_control(self):
        if self.state is None:
            return
        u = list(self(self.state))
        control_msg = Float32MultiArray()
        control_msg.data = u
        self.control_pub.publish(control_msg)
