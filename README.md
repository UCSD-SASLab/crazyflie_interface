# Crazyflie Interface package
This package provides a basic interface to use low-level control (roll, pitch, yaw rate, thrust) to interface with the crazyflie 2.1.
It is meant as the base package upon which more elaborate controllers can be built.

The `template_controller.py` file provides the template to inherit for creating one owns controller. It can be imported as follows: `from crazyflie_interface_py.template_controller import TemplateController`.
- We provide a basic `lqr_controller.py` to showcase how to inherit from the `TemplateController`.
- The `template_controller.py` has the following base functionality (which can be overridden) implemented:
    - Keeps track of `self.state` (subscribed from the `cf_interface/state` topic)
    - Provides a wrapper around `__call__(self, state)`; `publish_control.py` which converts the control commands to ROS2 message and publishes to the `cf_interface/control` topic.
    - Sets the control rate & associated timer.


The base functionality (`ros2 launch crazyflie_interface launch.py`) does the following:
- Command line arguments: backend (currently: `sim`, `cflib`), uri (the robot id, for `cflib` backend)
- Launches the `crazyflie launch.py` launch file with the given backend and uri. This initiates:
    - Motion capture tracking (if `cflib` backend)
    - Connection to the crazyflie (if `cflib` backend)
    - Basic simulation engine (if `sim` backend)
- Launches the `cf_interface.py` node which:
    - Subscribes to the relevant crazyflie state topic, converts to standard form (see below) and publishes it to `cf_interface/state` topic.
    - Subscribes to the `cf_interface/control` topic, converts to crazyflie format and publishes the low-level control commands to the crazyflie
    - Provides the `cf_interface/command` service for high-level interface to crazyflie: `takeoff`, `land`
        - Example call: `ros2 service call /cf_interface/command crazyflie_interface/srv/Command "{command: 'takeoff'}"`
    

## Standard form
State message is as follows: [x, y, z, vx, vy, vz, q_x, q_y, q_z, q_w, wx, wy, wz]
- Linear velocities are in world frame
- Angular velocities are in body frame

Control message is as follows: [roll, pitch, yaw_rate, thrust]
- Roll, pitch are in radians
- Yaw rate is in radians per second
- Thrust is in Newtons per kilogram (same as gravity)

## Getting started

### Dependencies
- [UCSD SASLab's crazyswarm2 package](https://github.com/UCSD-SASLab/crazyswarm2).
- Tested on ROS2 Humble only.

### Installation
- `colcon build --packages-select crazyflie_interface` from `ros2_ws` directory.
- `source install/setup.bash` from `ros2_ws` directory.

### Running base functionality (sim)
1. `ros2 launch crazyflie_interface base_controller_launch.py backend:=sim` for simulation backend.
2. `ros2 service call /cf_interface/command crazyflie_interface/srv/Command "{command: 'takeoff'}"` to takeoff.
3. It will start flying to random position targets.
3. `ros2 service call /cf_interface/command crazyflie_interface/srv/Command "{command: 'land'}"` to land.


### Running base functionality (hardware)
1. Identify drone URI (# under drone)
2. `ros2 launch crazyflie_interface base_controller_launch.py backend:=cflib uri:=uri` for hardware backend.
3. `ros2 service call /cf_interface/command crazyflie_interface/srv/Command "{command: 'takeoff'}"` to takeoff.
4. It will start flying to random targets
5. `ros2 service call /cf_interface/command crazyflie_interface/srv/Command "{command: 'land'}"` to land.

## TODOs
- [ ] Add GUI for command service.
- [ ] Add `gym_pybullet_drones` interface for simulation backend.
- [ ] Provide option for running rviz with `gym_pybullet_drones`.
- [ ] Add time to state message.
- [ ] Add calibration as part of command service in `cf_interface.py`.
