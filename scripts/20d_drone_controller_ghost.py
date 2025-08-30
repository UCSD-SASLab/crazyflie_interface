#!/usr/bin/env python3
import rclpy
import numpy as np
import torch
import rowan
import sys
import os
import time
import json
import pickle
import inspect
from datetime import datetime
from std_msgs.msg import Bool

# Add the scripts directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crazyflie_interface_py.template_controller import TemplateController
from deepreach.utils.modules import SingleBVPNet
from deepreach.dynamics import DronePursuitEvasion20D
from deepreach import dynamics
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

# Set device for PyTorch
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Model path - update this to your actual 20D model path
MODEL_PATH = "/mounted_volume/ros2_ws/src/crazyflie_interface/scripts/dr_models/DronePursuitEvasion20D_MPC_halfellipse"

# numpy logging only 2 digits
np.set_printoptions(precision=2, suppress=True, floatmode='fixed')

MODE = ["hover", "deepreach"][1]  # Default to deepreach mode
GHOST_AGENT = ["pursuer", "evader", "both"][0]
GHOST_CONTROL_MODE = ["hover", "circle", "deepreach"][2]  # How to control the ghost agent (NOTE only when GHOST_AGENT is not "both")
INIT_SETUP = 1

class DeepReach20DControllerGhost(TemplateController):
    def __init__(self, node_name='deepreach_20d_controller_ghost'):
        super().__init__(node_name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.ghost_state_pursuer = np.zeros(10)
        self.ghost_state_evader = np.zeros(10)

        self.in_flight = False
        self.create_subscription(Bool, "cf_interface/flight_status", self.flight_status_callback, 1)

        ## TEST
        # self.ghost_state_evader[[0,4,8]] = np.array([-0.25, -0.25, 1.0])  # Initial EVADER ghost position
        # self.ghost_state_pursuer[[0,4,8]] = np.array([0.0, 0.0, 1.0])  # Initial PURSUER ghost position

        ## 1 - FACE TO FACE ##
        if INIT_SETUP == 1:
            self.ghost_state_evader = np.array([-0.25, 0., 0., 0., 0., 0., 0., 0., 0.5, 0.])  # Initial EVADER ghost position 
            self.ghost_state_pursuer = np.array([0.25, 0., 0., 0., 0., 0., 0., 0., 0.5, 0.])  # Initial PURSUER ghost position

        ## 2 - OFFSET ##
        elif INIT_SETUP == 2:
            self.ghost_state_evader = np.array([-0.3, 0., 0., 0., 0.2, 0., 0., 0., 0.3, 0.])  # Initial EVADER ghost position 
            self.ghost_state_pursuer = np.array([0.3, 0., 0., 0.,-0.2, 0., 0., 0., 0.7, 0.])  # Initial PURSUER ghost position

        ## 3 - DIFF HEIGHTS ##
        elif INIT_SETUP == 3:
            self.ghost_state_evader = np.array([-0.1, 0., 0., 0., 0.1, 0., 0., 0., 0.2, 0.])  # Initial EVADER ghost position 
            self.ghost_state_pursuer = np.array([0.3, 0., 0., 0., 0.1, 0., 0., 0., 0.8, 0.])  # Initial PURSUER ghost position

        ## 4 - EVADER ABOVE ##
        elif INIT_SETUP == 4:
            self.ghost_state_evader = np.array([0.25, 0., 0., 0., 0., 0., 0., 0., 0.7, 0.])  # Initial EVADER ghost position 
            self.ghost_state_pursuer = np.array([-0.25, 0., 0., 0., 0., 0., 0., 0., 0.3, 0.])  # Initial PURSUER ghost position
        
        ## 5 - EVADER ABOVE NEAR STATE SPACE ##
        elif INIT_SETUP == 5:
            self.ghost_state_evader = np.array([0., 0., 0., 0., 1.8, 0., 0., 0., 0.7, 0.])  # Initial EVADER ghost position 
            self.ghost_state_pursuer = np.array([0., 0., 0., 0., 1.4, 0., 0., 0., 0.3, 0.])  # Initial PURSUER ghost position

        ## 6 - EVADER IN CORNER ##
        elif INIT_SETUP == 6:
            self.ghost_state_evader = np.array([-3.8, 0., 0., 0., 1.8, 0., 0., 0., 0.7, 0.])  # Initial EVADER ghost position 
            self.ghost_state_pursuer = np.array([-3.6, 0., 0., 0., 1.6, 0., 0., 0., 0.3, 0.])  # Initial PURSUER ghost position

        else:
            raise ValueError("INIT_SETUP must be an integer between 1 and 6")

        # Get robot parameters
        self._ros_parameters = self._param_to_dict(self._parameters)
        robots = self._ros_parameters.get('robots', {})
        self.get_logger().info(f"Robots: {robots}")
        self.nbr_flying_robots = len(robots)
        self.nbr_robots = self.nbr_flying_robots + 1
        self.get_logger().info(f"Number of robots (including ghost): {self.nbr_robots}")

        # Create marker publisher for ghost visualization
        self.marker_pub = self.create_publisher(Marker, 'ghost_evader_marker', 1)
        
        if GHOST_AGENT in ["evader", "both"]:
            self.evader_marker = Marker()
            self.evader_marker.ns = "ghost_evader"
            self.evader_marker.header.frame_id = "world"
            self.evader_marker.type = Marker.SPHERE
            self.evader_marker.action = Marker.ADD
            self.evader_marker.scale.x = 0.1
            self.evader_marker.scale.y = 0.1
            self.evader_marker.scale.z = 0.1
            self.evader_marker.color.a = 1.0
            self.evader_marker.color.r = 0.0
            self.evader_marker.color.g = 1.0
            self.evader_marker.color.b = 0.0
            self.evader_marker.pose.position.x = float(self.ghost_state_evader[0].item())
            self.evader_marker.pose.position.y = float(self.ghost_state_evader[4].item())
            self.evader_marker.pose.position.z = float(self.ghost_state_evader[8].item())
            self.evader_marker.id = 0
            self.marker_pub.publish(self.evader_marker)

        if GHOST_AGENT in ["pursuer", "both"]:
            self.pursuer_marker = Marker()
            self.pursuer_marker.ns = "ghost_pursuer"
            self.pursuer_marker.header.frame_id = "world"
            self.pursuer_marker.type = Marker.SPHERE
            self.pursuer_marker.action = Marker.ADD
            self.pursuer_marker.scale.x = 0.1
            self.pursuer_marker.scale.y = 0.1
            self.pursuer_marker.scale.z = 0.1
            self.pursuer_marker.color.a = 1.0
            self.pursuer_marker.color.r = 1.0
            self.pursuer_marker.color.g = 0.0
            self.pursuer_marker.color.b = 0.0
            self.pursuer_marker.pose.position.x = float(self.ghost_state_pursuer[0].item())
            self.pursuer_marker.pose.position.y = float(self.ghost_state_pursuer[4].item())
            self.pursuer_marker.pose.position.z = float(self.ghost_state_pursuer[8].item())
            self.pursuer_marker.id = 1
            self.marker_pub.publish(self.pursuer_marker)

        with open(os.path.join(MODEL_PATH, "orig_opt.pickle"), 'rb') as f:
            self.orig_opt = pickle.load(f)
        
        # Initialize DeepReach components if using deepreach mode
        if MODE == "deepreach":
            # Initialize 20D dynamics
            # TODO: Make sure all parameters are correct
            # self.dynamics = DronePursuitEvasion20D(
            #     thrust_max=16.0,
            #     max_angle=0.3,  # radians
            #     max_torque=0.3,
            #     capture_radius=0.25,
            #     set_mode='avoid'
            # )

            # self.model = SingleBVPNet(
            #     in_features=25,  # 20 state + 1 time + 4 periodic transforms
            #     hidden_features=512,
            #     num_hidden_layers=3,
            #     out_features=1,
            #     type='sine',
            #     periodic_transform_fn=self.dynamics.periodic_transform_fn 
            # )

            dynamics_class = getattr(dynamics, self.orig_opt.dynamics_class)
            self.dynamics = dynamics_class(**{argname: getattr(self.orig_opt, argname)
                          for argname in inspect.signature(dynamics_class).parameters.keys() if argname != 'self'})
            
            self.model = SingleBVPNet(in_features=self.dynamics.input_dim, out_features=1, type=self.orig_opt.model, mode=self.orig_opt.model_mode,
                             final_layer_factor=1., hidden_features=self.orig_opt.num_nl, num_hidden_layers=self.orig_opt.num_hl,
                             periodic_transform_fn=self.dynamics.periodic_transform_fn)

            checkpoint = torch.load(os.path.join(MODEL_PATH, "model_final.pth"), map_location=device, weights_only=True)
            self.model.load_state_dict(checkpoint["model"])
            self.model.to(device)
            self.model.eval()
            self.get_logger().info("DeepReach 20D model loaded successfully, modelpath = " + MODEL_PATH)
        
        self.start_controller()
        self.iteration = 0
        
        # Initialize state history for RK4 integration
        self.dt = 1.0/self.controller_rate # Control period (50Hz)

        self.get_logger().info(f"Control period: {self.dt}")

        # Initialize JSON logging
        self.log_data = []
        self.start_time = time.time()  # Track start time for relative timestamps
        self.log_filename = f"/mounted_volume/drone_experiment_data/20drones_{GHOST_AGENT}ghost_ic{INIT_SETUP}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.get_logger().info(f"JSON logging enabled. Log file: {self.log_filename}")
        
        # Track first occurrence of events
        self.first_collision_warning_time = None
        self.first_out_of_bounds_time = None
    
    def flight_status_callback(self, msg):
        if msg.data:
            self.in_flight = True
        
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

    def __call__(self, state):
        """
        Main control function for torque/thrust control
        
        Args:
            state: Array containing state data for all robots
                   Format: [x, y, z, vx, vy, vz, qx, qy, qz, qw, ...] for each robot
        
        Returns:
            control: Flattened array of control inputs for all robots
                     Format: 4 elements per robot [roll, pitch, yaw_rate, thrust]
        """
        states = np.array(state).reshape(self.nbr_flying_robots, -1)
        u = np.zeros((self.nbr_flying_robots, 4))
        
        if MODE == "hover":
            # Hover mode - set thrust to hover value
            self.u_hover = np.array([0.0, 0.0, 0.0, 11.95])
            u[:] = self.u_hover  # thrust

            pos = states[0, 0:3]    # x, y, z
            vel = states[0, 3:6]    # vx, vy, vz
            quat_raw = states[0, 6:10]  # qx, qy, qz, qw
            omega = states[0, 10:13] # omega_x, omega_y, omega_z
            
            # Convert quaternion from [qx, qy, qz, qw] to [qw, qx, qy, qz] format for rowan
            quat = np.array([quat_raw[3], quat_raw[0], quat_raw[1], quat_raw[2]])  # [qw, qx, qy, qz]
            
            # Convert quaternion to Euler angles to get roll and pitch
            euler_angles = rowan.to_euler(quat, "xyz")
            # The euler angles here are flipped compared to the drone convention
            roll = -euler_angles[0]   # θ_y  (post sign change: +roll = positive y acceleration)
            pitch = euler_angles[1]  # θ_x  (without sign change: +pitch = positive x acceleration)
                
        elif MODE == "deepreach":
            # Handle takeoff and hover phase
            if(self.nbr_robots != 2):
                raise ValueError("Pursuit evasion mode only supports 2 drones")

            # Extract full state information from robot states
            # Initialize 20D state with zeros for angles and angular velocities
            drone_20d_state = np.zeros(20)
            
            for i, robot_state in enumerate(states):

                if GHOST_AGENT != "both":
                    # Full state format: [x, y, z, vx, vy, vz, qx, qy, qz, qw, omega_x, omega_y, omega_z, ...]
                    pos = robot_state[0:3]    # x, y, z
                    vel = robot_state[3:6]    # vx, vy, vz
                    quat_raw = robot_state[6:10]  # qx, qy, qz, qw
                    omega = robot_state[10:13] # omega_x, omega_y, omega_z
                    
                    # Convert quaternion from [qx, qy, qz, qw] to [qw, qx, qy, qz] format for rowan
                    quat = np.array([quat_raw[3], quat_raw[0], quat_raw[1], quat_raw[2]])  # [qw, qx, qy, qz]
                    
                    # Convert quaternion to Euler angles to get roll and pitch
                    euler_angles = rowan.to_euler(quat, "xyz")
                    # The euler angles here are flipped compared to the drone convention
                    roll = -euler_angles[0]   # θ_y  (post sign change: +roll = positive y acceleration)
                    pitch = euler_angles[1]  # θ_x  (without sign change: +pitch = positive x acceleration)
                
                if GHOST_AGENT == "pursuer":  # Live Drone 1 (evader)
                    # [x1, v1_x, θ1_x, ω1_x, y1, v1_y, θ1_y, ω1_y, z1, v1_z]
                    drone_20d_state[0] = pos[0]   # x1
                    drone_20d_state[1] = vel[0]   # v1_x
                    drone_20d_state[2] = pitch    # θ1_x (pitch angle)
                    drone_20d_state[3] = omega[0] # ω1_x (angular velocity around x)
                    drone_20d_state[4] = pos[1]   # y1
                    drone_20d_state[5] = vel[1]   # v1_y
                    drone_20d_state[6] = roll     # θ1_y (roll angle)
                    drone_20d_state[7] = omega[1] # ω1_y (angular velocity around y)
                    drone_20d_state[8] = pos[2]   # z1
                    drone_20d_state[9] = vel[2]   # v1_z

                    # Create a simple ghost state for the pursuer
                    drone_20d_state[10:20] = self.ghost_state_pursuer

                elif GHOST_AGENT == "evader":  # Live Drone 2 (pursuer)
                    # [x2, v2_x, θ2_x, ω2_x, y2, v2_y, θ2_y, ω2_y, z2, v2_z]
                    drone_20d_state[10] = pos[0]  # x2
                    drone_20d_state[11] = vel[0]  # v2_x
                    drone_20d_state[12] = pitch   # θ2_x (pitch angle)
                    drone_20d_state[13] = omega[0] # ω2_x (angular velocity around x)
                    drone_20d_state[14] = pos[1]  # y2
                    drone_20d_state[15] = vel[1]  # v2_y
                    drone_20d_state[16] = roll    # θ2_y (roll angle)
                    drone_20d_state[17] = omega[1] # ω2_y (angular velocity around y)
                    drone_20d_state[18] = pos[2]  # z2
                    drone_20d_state[19] = vel[2]  # v2_z

                    # Create a simple ghost state for the evader
                    drone_20d_state[0:10] = self.ghost_state_evader

                elif GHOST_AGENT == "both":
                    drone_20d_state[0:10] = self.ghost_state_evader
                    drone_20d_state[10:20] = self.ghost_state_pursuer

                else:
                    raise ValueError(f"Unknown GHOST_AGENT: {GHOST_AGENT}")
            
            # Convert to tensor for DeepReach
            drone_20d_state_tensor = torch.tensor(drone_20d_state, dtype=torch.float32, device=device)

            self.get_logger().info(f"Evader state: {drone_20d_state[0:10]}")
            self.get_logger().info(f"Pursuer state: {drone_20d_state[10:20]}")
            
            # Add time dimension for DeepReach input: [time, state]
            time_tensor = torch.tensor([1.0], dtype=torch.float32, device=device)
            deepreach_input = torch.cat([time_tensor, drone_20d_state_tensor]).unsqueeze(0)  # [1, 21]
            
            traj_policy_results = self.model(
                {"coords": self.dynamics.coord_to_input(deepreach_input)}
            )
            
            model_out = traj_policy_results["model_out"]
            model_in = traj_policy_results["model_in"]
            
            # Ensure proper shape for dynamics calculations
            if model_out.dim() == 1:
                model_out = model_out.unsqueeze(0)
            
            dv = self.dynamics.io_to_dv(
                model_in,
                model_out.squeeze(dim=-1),
            ).detach()

            value = self.dynamics.io_to_value(
                model_in,
                model_out.squeeze(dim=-1),
            ).detach()

            ellx = self.dynamics.boundary_fn(drone_20d_state_tensor).detach()
            
            # Use gradient to compute optimal control and disturbance
            optimal_u = self.dynamics.optimal_control(drone_20d_state_tensor, dv[..., 1:])
            optimal_d = self.dynamics.optimal_disturbance(drone_20d_state_tensor, dv[..., 1:])
            
            # Debug gradients for thrust control
            dVdv1_z = dv[..., 1:][0, 9].item()  # gradient w.r.t. v1_z
            dVdv2_z = dv[..., 1:][0, 19].item()  # gradient w.r.t. v2_z
            #self.get_logger().info(f"Gradients - dVdv1_z: {dVdv1_z:.4f}, dVdv2_z: {dVdv2_z:.4f}")
            
            # Extract control inputs from DeepReach
            # Control: [S1_x, S1_y, T1_z] (evader)
            # Disturbance: [S2_x, S2_y, T2_z] (pursuer)
            evader_control = np.array([
                self.dynamics.max_torque * optimal_u[0, 0].item(),  # S1_x
                self.dynamics.max_torque * optimal_u[0, 1].item(),  # S1_y
                # self.dynamics.thrust_max * self.dynamics.k_T * optimal_u[0, 2].item()   # T1_z  # FIXME: Check whether this is correct
                self.dynamics.thrust_max * optimal_u[0, 2].item()   # T1_z
            ])
            pursuer_control = np.array([
                self.dynamics.max_torque * optimal_d[0, 0].item(),  # S2_x
                self.dynamics.max_torque * optimal_d[0, 1].item(),  # S2_y
                # self.dynamics.thrust_max * self.dynamics.k_T * optimal_d[0, 2].item()   # T2_z
                self.dynamics.thrust_max * optimal_d[0, 2].item()   # T2_z
            ])

            self.get_logger().info(f"Evader control: {evader_control}")
            self.get_logger().info(f"Pursuer control: {pursuer_control}")
            self.get_logger().info(f"value: {value.item():.4f}")
            self.get_logger().info(f"ellx: {ellx.item():.4f}")
            # Apply control limits for safety     # T2_z
            # FIXME: Add in that we want to control yaw again
            # Convert DeepReach controls to Crazyflie format: [roll, pitch, yaw_rate, thrust]
            if GHOST_AGENT == "pursuer":
                # Evader (drone 0) : DeepReach control
                u[0, 0] = evader_control[1]  # roll  # drone convention (+roll = positive y acceleration)
                u[0, 1] = -evader_control[0]  # pitch # SIGN CHANGE for drone convention (+pitch = negative x acceleration)
                u[0, 2] = 0.0  # yaw_rate
                u[0, 3] = evader_control[2]  # thrust

                # We want to calculate (if necessary the updated ghost state)
                if GHOST_CONTROL_MODE == "hover":
                    self.ghost_state_pursuer = self.ghost_state_pursuer  # No change, maintain hover
                elif GHOST_CONTROL_MODE == "circle":
                    # self.ghost_state_pursuer[0] = initial_ghost_state[0] + 1.0 * np.cos(0.2 * self.iteration * self.dt)  # x
                    raise NotImplementedError("Circle mode not implemented yet")
                elif GHOST_CONTROL_MODE == "deepreach":
                    # Imagine what the next 20d state would be by integrating deepreach forward
                    f = self.dynamics.dsdt(drone_20d_state_tensor, optimal_u, optimal_d)
                    if self.in_flight:
                        next_state = drone_20d_state_tensor + self.dt * f.squeeze(0)
                        self.ghost_state_pursuer = next_state[10:20].cpu().numpy()  # Update ghost state to next pursuer state
                else:
                    raise ValueError(f"Unknown GHOST_CONTROL_MODE: {GHOST_CONTROL_MODE}")
                if self.iteration % 5 == 0:
                    self.pursuer_marker.pose.position.x = float(self.ghost_state_pursuer[0].item())
                    self.pursuer_marker.pose.position.y = float(self.ghost_state_pursuer[4].item())
                    self.pursuer_marker.pose.position.z = float(self.ghost_state_pursuer[8].item())
                    self.marker_pub.publish(self.pursuer_marker)

            elif GHOST_AGENT == "evader":
                # Pursuer (drone 1) : fixed hover
                u[0, 0] = pursuer_control[1]  # roll  # drone convention (+roll = positive y acceleration)
                u[0, 1] = -pursuer_control[0]  # pitch # SIGN CHANGE for drone convention (+pitch = negative x acceleration)
                u[0, 2] = 0.0
                u[0, 3] = pursuer_control[2]  # thrust

                if GHOST_CONTROL_MODE == "hover":
                    self.ghost_state_evader = self.ghost_state_evader  # No change, maintain hover
                elif GHOST_CONTROL_MODE == "circle":
                    # self.ghost_state_evader[0] = initial_ghost_state[0] + 1.0 * np.cos(0.2 * self.iteration * self.dt)  # x
                    raise NotImplementedError("Circle mode not implemented yet")
                elif GHOST_CONTROL_MODE == "deepreach":
                    # Imagine what the next 20d state would be by integrating deepreach forward
                    f = self.dynamics.dsdt(drone_20d_state_tensor, optimal_u, optimal_d)
                    if self.in_flight:
                        next_state = drone_20d_state_tensor + self.dt * f.squeeze(0)
                        self.ghost_state_evader = next_state[0:10].cpu().numpy()  # Compute next evader state
                else:
                    raise ValueError(f"Unknown GHOST_CONTROL_MODE: {GHOST_CONTROL_MODE}")
                if self.iteration % 5 == 0:
                    self.evader_marker.pose.position.x = float(self.ghost_state_evader[0].item())
                    self.evader_marker.pose.position.y = float(self.ghost_state_evader[4].item())
                    self.evader_marker.pose.position.z = float(self.ghost_state_evader[8].item())
                    self.marker_pub.publish(self.evader_marker)
            
            elif GHOST_AGENT == "both":
                f = self.dynamics.dsdt(drone_20d_state_tensor, optimal_u, optimal_d)
                if self.in_flight:
                    next_state = drone_20d_state_tensor + self.dt * f.squeeze(0)
                    self.ghost_state_evader = next_state[0:10].cpu().numpy()  # Compute next evader state
                    self.ghost_state_pursuer = next_state[10:20].cpu().numpy()  # Compute next pursuer state
                if self.iteration % 5 == 0:
                    self.evader_marker.pose.position.x = float(self.ghost_state_evader[0].item())
                    self.evader_marker.pose.position.y = float(self.ghost_state_evader[4].item())
                    self.evader_marker.pose.position.z = float(self.ghost_state_evader[8].item())
                    self.marker_pub.publish(self.evader_marker)
                    self.pursuer_marker.pose.position.x = float(self.ghost_state_pursuer[0].item())
                    self.pursuer_marker.pose.position.y = float(self.ghost_state_pursuer[4].item())
                    self.pursuer_marker.pose.position.z = float(self.ghost_state_pursuer[8].item())
                    self.marker_pub.publish(self.pursuer_marker)
            
            
            # TODO AY: Add in a marker for the ghost drone in RViz to visualize the ghost position
                # We want to calculate (if necessary the updated ghost state)
            else:
                raise ValueError(f"Unknown GHOST_AGENT: {GHOST_AGENT}")
            
            # Apply final safety limits
            u[:, :2] = np.clip(u[:, :2], -0.3, 0.3)  # roll, pitch limits
            u[:, 3] = np.clip(u[:, 3], 4.0, 16.0)    # thrust limits
                
        else:
            raise NotImplementedError(f"Mode {MODE} not implemented")
            
        self.iteration += 1
        
        # Log the actual positions of the robots
        # Extract actual positions and velocities from [x, y, z, vx, vy, vz] format
        if MODE == "deepreach":
            actual_pos1 = drone_20d_state[[0, 4, 8]]  # Current actual position [x, y, z]
            actual_vel1 = drone_20d_state[[1, 5, 9]]  # Current actual velocity [vx, vy, vz]
            actual_pos2 = drone_20d_state[[10, 14, 18]]  # Current actual position [x, y, z]
            actual_vel2 = drone_20d_state[[11, 15, 19]]  # Current actual velocity [vx, vy, vz]

            xydist = np.linalg.norm(actual_pos1[0:2] - actual_pos2[0:2])
            z_dist = np.abs(actual_pos1[2] - actual_pos2[2])

            # Box bounds check (Evader only)
            box_min = np.array([-4.5, -2.5, 0.0])  # min x, y, z
            box_max = np.array([ 4.5,  2.5, 2.5])  # max x, y, z

            if(xydist < 0.25 and z_dist < 0.75):
                # Track first collision warning time
                if self.first_collision_warning_time is None:
                    self.first_collision_warning_time = time.time() - self.start_time
                self.get_logger().info(f"Collision warning! Distance: {xydist:.2f} m in xy and {z_dist:.2f} m in z direction")

            if np.any(actual_pos1 < box_min) or np.any(actual_pos1 > box_max):
                # Track first out of bounds time
                if self.first_out_of_bounds_time is None:
                    self.first_out_of_bounds_time = time.time() - self.start_time
                self.get_logger().warn(
                    f"Evader OUT OF BOUNDS: position {actual_pos1}"
                )

            if self.iteration % 5 == 0:  # Log every 50 iterations (about once per second)
                self.get_logger().info(f"DeepReach 20D Mode - Iteration {self.iteration}: Current positions for {self.nbr_robots} robots")

                self.get_logger().info(f"Evader actual position: [{actual_pos1[0]:.2f}, {actual_pos1[1]:.2f}, {actual_pos1[2]:.2f}]")
                self.get_logger().info(f"Evader actual velocity: [{actual_vel1[0]:.2f}, {actual_vel1[1]:.2f}, {actual_vel1[2]:.2f}]")
                self.get_logger().info(f"Pursuer actual position: [{actual_pos2[0]:.2f}, {actual_pos2[1]:.2f}, {actual_pos2[2]:.2f}]")
                self.get_logger().info(f"Pursuer actual velocity: [{actual_vel2[0]:.2f}, {actual_vel2[1]:.2f}, {actual_vel2[2]:.2f}]")

                if self.first_collision_warning_time is not None or self.first_out_of_bounds_time is not None:
                    self.get_logger().warn("Safety event detected - ending control loop")

            log_entry = {
                "timestamp": time.time() - self.start_time,  # Relative time in seconds
                "evader": {
                    "actual_position": actual_pos1.tolist(),
                    "actual_velocity": actual_vel1.tolist(),
                    "control": evader_control.tolist() if MODE == "deepreach" else None,
                    "full_state": drone_20d_state[0:10].tolist()
                },
                "pursuer": {
                    "actual_position": actual_pos2.tolist(),
                    "actual_velocity": actual_vel2.tolist(),
                    "control": pursuer_control.tolist() if MODE == "deepreach" else None,
                    "full_state": drone_20d_state[10:20].tolist()
                },
                "distances": {
                    "xy_distance": float(xydist),
                    "z_distance": float(z_dist)
                },
                "events": {
                    "first_collision_warning_time": self.first_collision_warning_time,
                    "first_out_of_bounds_time": self.first_out_of_bounds_time
                }
            }
            self.log_data.append(log_entry)

        self.get_logger().info(f"Control: {u}")
        self.get_logger().info(f"Position: {state[0:3]}")
        self.get_logger().info(f"Roll for DR: {roll:.2f}, Pitch for DR: {pitch:.2f}")
        return u.flatten()

    def save_log_file(self):
        """Save the logged data to a JSON file."""
        if self.log_data:
            try:
                with open(self.log_filename, 'w') as f:
                    json.dump({
                        "metadata": {
                            "timestamp": datetime.now().isoformat(),
                            "total_iterations": len(self.log_data),
                            "mode": MODE,
                            "model_path": MODEL_PATH
                        },
                        "data": self.log_data
                    }, f, indent=2)
                self.get_logger().info(f"Log data saved to {self.log_filename}")
            except Exception as e:
                self.get_logger().error(f"Failed to save log file: {e}")


def main(args=None):
    rclpy.init(args=args)
    controller = DeepReach20DControllerGhost()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        print("Experiment ended by keyboard interrupt")
    finally:
        # Save log file before shutting down
        controller.save_log_file()
        rclpy.shutdown()
        
        # Check if safety event occurred and log it
        if controller.first_collision_warning_time is not None or controller.first_out_of_bounds_time is not None:
            print(f"Experiment ended due to safety event:")
            if controller.first_collision_warning_time is not None:
                print(f"  - First collision warning at {controller.first_collision_warning_time:.2f} seconds")
            if controller.first_out_of_bounds_time is not None:
                print(f"  - First out-of-bounds event at {controller.first_out_of_bounds_time:.2f} seconds")


if __name__ == "__main__":
    main()