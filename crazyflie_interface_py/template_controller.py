#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
from example_interfaces.msg import Float64MultiArray

class TemplateController(Node):
    def __init__(self, node_name='template_controller', controller_rate=50.0):
        super().__init__(node_name)
        self.controller_rate = controller_rate
        # Subclasses should call start_controller() at the end of their __init__ method
        self.state = None
        self.control_pub = self.create_publisher(Float64MultiArray, 'cf_interface/control', 1)
        self.create_subscription(Float64MultiArray, 'cf_interface/state', self.callback_state, 1)

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
        control_msg = Float64MultiArray()
        control_msg.data = u
        self.control_pub.publish(control_msg)
