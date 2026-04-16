#!/usr/bin/env python3
import rclpy

from crazyflie_interface_py.iterative_planning_controller import HerdController


def main(args=None):
    rclpy.init(args=args)
    controller = HerdController()
    rclpy.spin(controller)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
