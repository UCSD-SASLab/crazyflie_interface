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
from deepreach.dynamics.dynamics import DronePursuitEvasion20D
from deepreach.dynamics import dynamics
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from collections import deque

# Set device for PyTorch
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Model path - update this to your actual 20D model path

# numpy logging only 2 digits
np.set_printoptions(precision=2, suppress=True, floatmode='fixed')

MODE = ["hover", "deepreach"][1]  # Default to deepreach mode
GHOST_CONTROL_MODE = ["hover", "circle", "deepreach"][2]  # How to control the ghost agent (NOTE only when GHOST_AGENT is not "both")
INIT_SETUP = 2
LOOKBACK_TIME = 1. # deepreach
CONTROLLER_RATE = 50.   # NOTE: WILL TRIED 50, 30, 10 --> 30 maybe best?
CALIBRATE_FIRST = True

GHOST_AGENT = ["pursuer", "evader", "both", "none"][3]
CLAMP_RPYT_CONTROLS = False
REAL_TORQUE_MAG = 0.1
# REAL_THRUST_MAX = 10.
REAL_THRUST_MAX = 10.5
REAL_THRUST_MIN = 8.5

GHOST_PURSUER_SLOW_FACTOR = 1. # 0.5 # takes factor * step_size in integration 
GHOST_EVADER_SLOW_FACTOR = 1. # 0.5 # takes factor * step_size in integration
GHOST_PURSUER_SIMPLE = False
GHOST_EVADER_SIMPLE = False
SIMPLE_RADIUS = 1.5
SIMPLE_FREQ = 0.05
SIMPLE_HEIGHT = 1.

TWOPLAYER_MODEL_NAME = "halfellipse_PEonly" # "halfellipse_PEonly"

TWOPLAYER_MODEL_FOLLOW_NAME = "halfellipse_PEonly"
USE_FOLLOW_FILTER = False # Whether to apply the follow strategy for the pursuer
FOLLOW_VALUE_THRESHOLD = 0.1 # If value fn > threshold, switch to follow strategy

USE_PURSUER_ARENA_FILTER = True # Use add'l value fn to contain agents (MODE = "deepreach" only)
USE_EVADER_ARENA_FILTER = True
SINGLEAGENT_MODEL_NAME = "Drone10D_posvel" # "Drone10D_posvel" (BEST) "Drone10D_posvel" # "lowerbounds_lowthrust", "Drone10D_MPC_box", "Drone10D_omega2_box"
EVADER_ARENA_VALUE_THRESHOLD = 0.1
PURSUER_ARENA_VALUE_THRESHOLD = 0.0 # If arena val fn < threshold, switch to stay-in-box strategy

USE_SMOOTH_ARENA_FILTER = True
BETA_SMOOTHING = 1.

LOAD_PRESOLVED_EVADER_TRAJ = False  # Whether to load a presolved trajectory for the evader agent
PRESOLVED_EVADER_FILE = "EVADER_STATES_20drones_pursuerghost_ic2_20250903_205521.npz"  # File containing presolved evader trajectory

WAYPOINT_CONTROL = False # send iterative waypoints to follow (rather than RPYT control)
WP_INTEGRATION_HZN = 0.15 # how far to integrate trajectory for next waypoint
# WP_RATE_PER_CTRL = 5 # waypoint publications per control actions, hence true freq = CONTROLLER_RATE / WP_RATE_PER_CTRL  # Not used
INTEG_STRETCH_FACTOR = 0.75 # stretch the time_step s.t. xi <- xi + stretch * dt * f(xi, ui, di)

USE_EMERGENCY_ARENA_OVERRIDE = True # Whether to override controls to keep agents in arena
ARENA_X_LIMIT = 3.8
ARENA_Y_LIMIT = 1.7
ARENA_Z_MIN = 0.4
ARENA_Z_MAX = 2.0
LQR_OVERRIDE_EXIT_THRESH = 0.5 # When to exit LQR override (exiting agent(s) within this distance of last in bounds pos)
RESET_PROJ_FACTOR = 0.9 # When resetting to last safe pos, scale reset towards center of arena by this factor (to avoid deadlock)

class DeepReach20DControllerGhost(TemplateController):
    def __init__(self, node_name='deepreach_20d_controller_ghost'):
        self.circle_iteration = 0  # TEMP ST
        if WAYPOINT_CONTROL:
            self.control_publisher_topic = 'cf_interface/control_full_state'

        super().__init__(node_name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True,
                         controller_rate=CONTROLLER_RATE)
        self.ghost_state_pursuer = np.zeros(10)
        self.ghost_state_evader = np.zeros(10)

        self.in_flight = False
        self.in_bounds_evader = True
        self.in_bounds_pursuer = True
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
            self.ghost_state_evader = np.array([0.1, 0., 0., 0., 0.2, 0., 0., 0., 1.0, 0.])  # Initial EVADER ghost position 
            self.ghost_state_pursuer = np.array([-2.0, 0., 0., 0., -0.2, 0., 0., 0., 1.0, 0.])  # Initial PURSUER ghost position
            # self.ghost_state_evader = np.array([0.3, 0., 0., 0., -0.2, 0., 0., 0., 1.0, 0.])  # Initial EVADER ghost position 
            # self.ghost_state_evader = np.array([0.5, 0., 0., 0., -0.4, 0., 0., 0., 1.0, 0.])  # Initial EVADER ghost position  TEMP TEMP
            # self.ghost_state_pursuer = np.array([-0.3, 0., 0., 0., 0.2, 0., 0., 0., 0.5, 0.])  # Initial PURSUER ghost position

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

        ## 7 - OPPOSITE ##
        elif INIT_SETUP == 7:
            self.ghost_state_evader = np.array([-1., 0., 0., 0., 1.8, 0., 0., 0., 0.7, 0.])  # Initial EVADER ghost position 
            self.ghost_state_pursuer = np.array([1., 0., 0., 0., 0., 0., 0., 0., 0.3, 0.])  # Initial PURSUER ghost position

        else:
            raise ValueError("INIT_SETUP must be an integer between 1 and 6")

        # Get robot parameters
        self._ros_parameters = self._param_to_dict(self._parameters)
        robots = self._ros_parameters.get('robots', {})
        self.get_logger().info(f"Robots: {robots}")
        self.nbr_flying_robots = min(len(robots), 2)
        if GHOST_AGENT == "none":
            self.nbr_robots = self.nbr_flying_robots
        else:
            self.nbr_robots = self.nbr_flying_robots + 1
        self.get_logger().info(f"Number of robots (including ghost): {self.nbr_robots}")

        # Initialize state history for RK4 integration
        self.dt = 1.0/self.controller_rate # Control period (50Hz)

        self.get_logger().info(f"Control period: {self.dt}")

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

        # Publish Arena Box for Visualization
        line_width = 0.1
        xmin, ymin, zmin = -4., -2., 0.2
        xmax, ymax, zmax = 4., 2., 2.0
        def _Point(x=0.0, y=0.0, z=0.0):
            point = Point()
            point.x, point.y, point.z = x, y, z
            return point    
            
        A = _Point(xmin, ymin, zmin)
        B = _Point(xmax, ymin, zmin)
        C = _Point(xmax, ymax, zmin)
        D = _Point(xmin, ymax, zmin)
        E = _Point(xmin, ymin, zmax)
        F = _Point(xmax, ymin, zmax)
        G = _Point(xmax, ymax, zmax)
        H = _Point(xmin, ymax, zmax)
        edges = [
            (A, B), (B, C), (C, D), (D, A),   # bottom rectangle
            (E, F), (F, G), (G, H), (H, E),   # top rectangle
            (A, E), (B, F), (C, G), (D, H),   # vertical edges
        ]
        arena_pts = []
        for p, q in edges:
            arena_pts.append(p)
            arena_pts.append(q)

        # self.marker_pub_arena = self.create_publisher(Marker, 'arena_box_marker', 10)
        self.arena_marker = Marker()
        self.arena_marker.ns = "arena_box"
        self.arena_marker.header.frame_id = "world"
        self.arena_marker.type = Marker.LINE_LIST
        self.arena_marker.action = Marker.ADD
        self.arena_marker.points = arena_pts
        self.arena_marker.scale.x = line_width
        self.arena_marker.scale.y = line_width
        self.arena_marker.scale.z = line_width
        self.arena_marker.pose.orientation.w = 1.0
        self.arena_marker.color.a = 0.5
        self.arena_marker.color.r = 1.0
        self.arena_marker.color.g = 1.0
        self.arena_marker.color.b = 0.0
        self.arena_marker.id = 2
        self.arena_marker.lifetime = rclpy.duration.Duration(seconds=0.0).to_msg()  # 0 means forever
        self.marker_pub.publish(self.arena_marker)
        # self.marker_pub_arena.publish(arena_marker)

        if WAYPOINT_CONTROL:
            if MODE != "deepreach":
                raise AssertionError("Need deepreach to integrate next waypoint .. unless you want to add the dynamics.")
            # if float(WP_INTEGRATION_HZN // self.dt) != WP_INTEGRATION_HZN / self.dt:
            #     self.get_logger().info(f"dt: {self.dt}, WP_INTEGRATION_HZN: {WP_INTEGRATION_HZN}")
            #     raise AssertionError("Fix your waypoint integration horizon to be a multiple of self.dt (=1/freq)")

            self.evader_wp_state = 1 * self.ghost_state_evader
            self.pursuer_wp_state =  1 * self.ghost_state_pursuer

            self.evader_wp_marker = Marker()
            self.evader_wp_marker.ns = "evader_waypoint"
            self.evader_wp_marker.header.frame_id = "world"
            self.evader_wp_marker.type = Marker.SPHERE
            self.evader_wp_marker.action = Marker.ADD
            self.evader_wp_marker.scale.x = 0.1
            self.evader_wp_marker.scale.y = 0.1
            self.evader_wp_marker.scale.z = 0.1
            self.evader_wp_marker.color.a = 0.4
            self.evader_wp_marker.color.r = 0.0
            self.evader_wp_marker.color.g = 1.0
            self.evader_wp_marker.color.b = 1.0
            self.evader_wp_marker.pose.position.x = float(self.evader_wp_state[0].item())
            self.evader_wp_marker.pose.position.y = float(self.evader_wp_state[4].item())
            self.evader_wp_marker.pose.position.z = float(self.evader_wp_state[8].item())
            self.evader_wp_marker.id = 2
            self.marker_pub.publish(self.evader_wp_marker)

            self.pursuer_wp_marker = Marker()
            self.pursuer_wp_marker.ns = "pursuer_waypoint"
            self.pursuer_wp_marker.header.frame_id = "world"
            self.pursuer_wp_marker.type = Marker.SPHERE
            self.pursuer_wp_marker.action = Marker.ADD
            self.pursuer_wp_marker.scale.x = 0.1
            self.pursuer_wp_marker.scale.y = 0.1
            self.pursuer_wp_marker.scale.z = 0.1
            self.pursuer_wp_marker.color.a = 0.4
            self.pursuer_wp_marker.color.r = 1.0
            self.pursuer_wp_marker.color.g = 0.0
            self.pursuer_wp_marker.color.b = 1.0
            self.pursuer_wp_marker.pose.position.x = float(self.pursuer_wp_state[0].item())
            self.pursuer_wp_marker.pose.position.y = float(self.pursuer_wp_state[4].item())
            self.pursuer_wp_marker.pose.position.z = float(self.pursuer_wp_state[8].item())
            self.pursuer_wp_marker.id = 3
            self.marker_pub.publish(self.pursuer_wp_marker)
        
        twoplayer_model_path = f"deepreach/saved_models/Drones20D/{TWOPLAYER_MODEL_NAME}"
        with open(os.path.join(twoplayer_model_path, "orig_opt.pickle"), 'rb') as f:
            self.orig_opt = pickle.load(f)

        
        # Initialize DeepReach components if using deepreach mode
        if MODE == "deepreach":

            dynamics_class = getattr(dynamics, self.orig_opt.dynamics_class)
            # Get the signature of the dynamics class constructor
            sig = inspect.signature(dynamics_class)
            
            # Build kwargs dict only for parameters that exist in orig_opt and are not None
            dynamics_kwargs = {}
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                    
                # Check if the parameter exists in orig_opt
                if hasattr(self.orig_opt, param_name):
                    value = getattr(self.orig_opt, param_name)
                    # Only add non-None values, or if the parameter has no default (is required)
                    if value is not None or param.default == inspect.Parameter.empty:
                        dynamics_kwargs[param_name] = value
                # If parameter has a default value and doesn't exist in orig_opt, skip it
                elif param.default == inspect.Parameter.empty:
                    # This is a required parameter that's missing from orig_opt
                    raise ValueError(f"Required parameter '{param_name}' not found in orig_opt for {dynamics_class.__name__}")
            self.dynamics = dynamics_class(**dynamics_kwargs)
            self.model = SingleBVPNet(in_features=self.dynamics.input_dim, out_features=1, type=self.orig_opt.model, mode=self.orig_opt.model_mode,
                             final_layer_factor=1., hidden_features=self.orig_opt.num_nl, num_hidden_layers=self.orig_opt.num_hl,
                             periodic_transform_fn=self.dynamics.periodic_transform_fn)

            checkpoint = torch.load(os.path.join(twoplayer_model_path, "training/checkpoints/model_final.pth"), map_location=device, weights_only=True)
            self.model.load_state_dict(checkpoint["model"])
            self.model.to(device)
            self.model.eval()
            self.get_logger().info("DeepReach 20D model loaded successfully, modelpath = " + twoplayer_model_path)
            
            if USE_FOLLOW_FILTER:
                twoplayer_follow_model_path = f"deepreach/saved_models/Drones20DFollow/{TWOPLAYER_MODEL_FOLLOW_NAME}"
                with open(os.path.join(twoplayer_follow_model_path, "orig_opt.pickle"), 'rb') as f:
                    self.orig_opt_follow = pickle.load(f)
                dynamics_class = getattr(dynamics, self.orig_opt_follow.dynamics_class)
                sig = inspect.signature(dynamics_class)
                
                # Build kwargs dict only for parameters that exist in orig_opt and are not None
                dynamics_kwargs = {}
                for param_name, param in sig.parameters.items():
                    if param_name == 'self':
                        continue
                        
                    # Check if the parameter exists in orig_opt
                    if hasattr(self.orig_opt_follow, param_name):
                        value = getattr(self.orig_opt_follow, param_name)
                        # Only add non-None values, or if the parameter has no default (is required)
                        if value is not None or param.default == inspect.Parameter.empty:
                            dynamics_kwargs[param_name] = value
                    # If parameter has a default value and doesn't exist in orig_opt, skip it
                    elif param.default == inspect.Parameter.empty:
                        # This is a required parameter that's missing from orig_opt
                        raise ValueError(f"Required parameter '{param_name}' not found in orig_opt for {dynamics_class.__name__}")
                self.dynamics_follow = dynamics_class(**dynamics_kwargs)
                
                self.model_follow = SingleBVPNet(in_features=self.dynamics_follow.input_dim, out_features=1, type=self.orig_opt_follow.model, mode=self.orig_opt_follow.model_mode,
                                final_layer_factor=1., hidden_features=self.orig_opt_follow.num_nl, num_hidden_layers=self.orig_opt_follow.num_hl,
                                periodic_transform_fn=self.dynamics_follow.periodic_transform_fn)

                checkpoint = torch.load(os.path.join(twoplayer_follow_model_path, "training/checkpoints/model_final.pth"), map_location=device, weights_only=True)
                self.model_follow.load_state_dict(checkpoint["model"])
                self.model_follow.to(device)
                self.model_follow.eval()
                self.get_logger().info("DeepReach 20D model follow loaded successfully, modelpath = " + twoplayer_follow_model_path)                

            if USE_PURSUER_ARENA_FILTER or USE_EVADER_ARENA_FILTER:
                safety_model_path = os.path.join('deepreach/saved_models/Drone10D', SINGLEAGENT_MODEL_NAME)
                self.get_logger().info("Loading arena containment model from " + safety_model_path)
                with open(os.path.join(safety_model_path, "orig_opt.pickle"), 'rb') as f:
                    self.orig_opt_arena = pickle.load(f)

                self.get_logger().info(f"Arena orig_opt: {self.orig_opt_arena}")
                
                dynamics_class = getattr(dynamics, self.orig_opt_arena.dynamics_class)
                sig = inspect.signature(dynamics_class)
                
                # Build kwargs dict only for parameters that exist in orig_opt and are not None
                dynamics_kwargs = {}
                for param_name, param in sig.parameters.items():
                    if param_name == 'self':
                        continue
                        
                    # Check if the parameter exists in orig_opt
                    if hasattr(self.orig_opt_arena, param_name):
                        value = getattr(self.orig_opt_arena, param_name)
                        # Only add non-None values, or if the parameter has no default (is required)
                        if value is not None or param.default == inspect.Parameter.empty:
                            dynamics_kwargs[param_name] = value
                    # If parameter has a default value and doesn't exist in orig_opt, skip it
                    elif param.default == inspect.Parameter.empty:
                        # This is a required parameter that's missing from orig_opt
                        raise ValueError(f"Required parameter '{param_name}' not found in orig_opt for {dynamics_class.__name__}")
                self.dynamics_arena = dynamics_class(**dynamics_kwargs)
                
                self.model_arena = SingleBVPNet(in_features=self.dynamics_arena.input_dim, out_features=1, type=self.orig_opt_arena.model, mode=self.orig_opt_arena.model_mode,
                                 final_layer_factor=1., hidden_features=self.orig_opt_arena.num_nl, num_hidden_layers=self.orig_opt_arena.num_hl,
                                 periodic_transform_fn=self.dynamics_arena.periodic_transform_fn)

                checkpoint = torch.load(os.path.join(safety_model_path, "training/checkpoints/model_final.pth"), map_location=device, weights_only=True)
                self.model_arena.load_state_dict(checkpoint["model"])
                self.model_arena.to(device)
                self.model_arena.eval()
                self.get_logger().info("Arena containment model loaded successfully, modelpath = " + safety_model_path)
            
        if LOAD_PRESOLVED_EVADER_TRAJ:
            traj_data = np.load(PRESOLVED_EVADER_FILE)
            self.presolved_evader_timestamps, self.presolved_evader_states = traj_data['timestamps'], traj_data['evader_full_states']

            # Convolve angles to smooth out noise
            window_size = 30 #assuming were using raw, unsmoothed data
            self.presolved_evader_states[:, 2] = np.convolve(self.presolved_evader_states[:, 2], np.ones(window_size)/window_size, mode='same') # pitch
            self.presolved_evader_states[:, 3] = np.convolve(self.presolved_evader_states[:, 3], np.ones(window_size)/window_size, mode='same') # omega_x
            self.presolved_evader_states[:, 6] = np.convolve(self.presolved_evader_states[:, 6], np.ones(window_size)/window_size, mode='same') # roll
            self.presolved_evader_states[:, 7] = np.convolve(self.presolved_evader_states[:, 7], np.ones(window_size)/window_size, mode='same') # omega_y

            # Clamp angles to learned state bounds
            angle_max, angular_vel_max = 0.35, 2.0
            self.presolved_evader_states[:, 2] = np.clip(self.presolved_evader_states[:, 2], -angle_max, angle_max)  # pitch
            self.presolved_evader_states[:, 3] = np.clip(self.presolved_evader_states[:, 3], -angular_vel_max, angular_vel_max)  # omega_x
            self.presolved_evader_states[:, 6] = np.clip(self.presolved_evader_states[:, 6], -angle_max, angle_max)  # roll
            self.presolved_evader_states[:, 7] = np.clip(self.presolved_evader_states[:, 7], -angular_vel_max, angular_vel_max)  # omega_y

            # Convert to tensor
            self.presolved_evader_states = torch.from_numpy(self.presolved_evader_states).float()
        
        self.start_controller()
        self.iteration = 0
        self.wp_iteration = 0

        # Initialize JSON logging
        self.log_data = []
        self.start_time = time.time()  # Track start time for relative timestamps
        self.log_filename = f"/mounted_volume/drone_experiment_data/20drones_{GHOST_AGENT}ghost_ic{INIT_SETUP}_{CONTROLLER_RATE}Hz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.get_logger().info(f"JSON logging enabled. Log file: {self.log_filename}")
        
        # Track first occurrence of events
        self.first_collision_warning_time = None
        self.first_out_of_bounds_time = None

        # For calibration
        self.calibrated = False
        self.calibration_counter = 0
        self.Gz = -9.81 if not MODE == "deepreach" else self.dynamics.Gz
        self.k_T = 0.83 if not MODE == "deepreach" else self.dynamics.k_T
        self.k_T_actual_evader = self.k_T
        self.k_T_actual_pursuer = self.k_T
        self.state_pos_buffer_evader = deque([], int(0.2 * 50.))
        self.state_pos_buffer_pursuer = deque([], int(0.2 * 50.))
        gain_matrix = np.zeros((4, 7))
        gain_matrix[0, 1] = -0.2  # y -> roll
        gain_matrix[0, 4] = -0.2  # v_y -> roll
        gain_matrix[1, 0] = 0.2  # x -> pitch
        gain_matrix[1, 3] = 0.2  # v_x -> pitch
        gain_matrix[2, 6] = 2.0  # yaw -> yaw_dot
        gain_matrix[3, 2] = -10.0  # z -> thrust
        gain_matrix[3, 5] = -10.0  # v_z -> thrust
        self.gain_matrix = gain_matrix
        self.u_hover_evader = np.array([0.0, 0.0, 0.0, 14]) 
        self.u_hover_pursuer = np.array([0.0, 0.0, 0.0, 11]) 
        if GHOST_AGENT  == "evader":
            self.goal_position_calibration = np.array([self.ghost_state_pursuer[[0,4,8]], self.ghost_state_pursuer[[0,4,8]]])
        elif GHOST_AGENT == "pursuer":
            self.goal_position_calibration = np.array([self.ghost_state_evader[[0,4,8]], self.ghost_state_evader[[0,4,8]]])
        else:
            self.goal_position_calibration = np.array([self.ghost_state_evader[[0,4,8]], self.ghost_state_pursuer[[0,4,8]]])

        self.last_safe_evader_pos = self.ghost_state_evader[[0,4,8]]
        self.last_safe_pursuer_pos = self.ghost_state_pursuer[[0,4,8]]
        self.emergency_lqr_override_evader_inuse = False
        self.emergency_lqr_override_pursuer_inuse = False
            
    def flight_status_callback(self, msg):
        if msg.data:
            if CALIBRATE_FIRST and not self.in_flight and not self.calibrated and not GHOST_AGENT == "both":
                self.get_logger().info("CALIBRATING CONTROLLER NOW...")
                self.calibration_timer = self.create_timer(4.0, self.calibrate_controller_callback)
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

    def get_control_and_disturbance(self, state):
        optimal_u, optimal_d, value, dv, ellx = self.infer_deepreach(state, LOOKBACK_TIME, self.model, self.dynamics)
        values = {"game": value}
        ## FOLLOW FILTER ##

        if USE_FOLLOW_FILTER and value.item() > FOLLOW_VALUE_THRESHOLD:
            _, optimal_d_follow, _, _, _ = self.infer_deepreach(state, LOOKBACK_TIME, self.model_follow, self.dynamics_follow)

            optimal_d = optimal_d_follow
            # self.get_logger().info(f"Using follow pursuer strategy (value = {value.item()})")

        ## ARENA FILTER ##

        if USE_EVADER_ARENA_FILTER:
            arena_u_evader, _, value_arena_evader, _, box_ellx_evader = self.infer_deepreach(state[0:10], LOOKBACK_TIME, self.model_arena, self.dynamics_arena, state_is_tensor=False)
            values["arena_evader"] = value_arena_evader
            if USE_SMOOTH_ARENA_FILTER:
                with torch.no_grad():
                    value_pos = torch.clamp(value_arena_evader - EVADER_ARENA_VALUE_THRESHOLD, min=0.0)
                    lambda_factor = 1 - torch.exp(-BETA_SMOOTHING * value_pos)
                    optimal_u = (lambda_factor * optimal_u + (1 - lambda_factor) * arena_u_evader)
            else:
                if value_arena_evader.item() < EVADER_ARENA_VALUE_THRESHOLD or box_ellx_evader.item() < 0.0:
                    optimal_u = arena_u_evader
                    # self.get_logger().info(f"Using arena safety filter for evader (value_arena_evader = {value_arena_evader.item()})")

        if USE_PURSUER_ARENA_FILTER:
            arena_u_pursuer, _, value_arena_pursuer, _, box_ellx_pursuer = self.infer_deepreach(state[10:20], LOOKBACK_TIME, self.model_arena, self.dynamics_arena, state_is_tensor=False)
            values["arena_pursuer"] = value_arena_pursuer
            
            if USE_SMOOTH_ARENA_FILTER:
                with torch.no_grad():
                    value_pos = torch.clamp(value_arena_pursuer - EVADER_ARENA_VALUE_THRESHOLD, min=0.0)
                    lambda_factor = 1 - torch.exp(-BETA_SMOOTHING * value_pos)
                    optimal_d = lambda_factor * optimal_d + (1 - lambda_factor) * arena_u_pursuer
            else:
                if value_arena_pursuer.item() < PURSUER_ARENA_VALUE_THRESHOLD or box_ellx_pursuer.item() < 0.0:
                    optimal_d = arena_u_pursuer  
    
        return optimal_u, optimal_d, values, ellx

    def infer_deepreach(self, state, eval_time, model, dynamics, state_is_tensor=True):

        time_tensor = torch.tensor([eval_time], dtype=torch.float32, device=device)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device) if not state_is_tensor else state
        state_tensor_bounded = dynamics.clip_state(state_tensor)
        deepreach_input = torch.cat([time_tensor, state_tensor_bounded]).unsqueeze(0)

        model_results = model(
            {"coords": dynamics.coord_to_input(deepreach_input)}
        ) 
        model_out = model_results["model_out"]
        model_in = model_results["model_in"]

        if model_out.dim() == 1:
            model_out = model_out.unsqueeze(0)
        
        dv = dynamics.io_to_dv(
            model_in, model_out.squeeze(dim=-1)
        ).detach()

        value = dynamics.io_to_value(
            model_in, model_out.squeeze(dim=-1)
        )

        boundary_value = dynamics.boundary_fn(state_tensor_bounded).detach()
        u = dynamics.optimal_control(state_tensor_bounded, dv[..., 1:])
        d = dynamics.optimal_disturbance(state_tensor_bounded, dv[..., 1:])
        return u, d, value, dv, boundary_value
    
    def calibrate_controller_callback(self):
        
        if GHOST_AGENT == "pursuer":
            avg_state_evader = np.mean(np.array(self.state_pos_buffer_evader), axis=0)
            self.get_logger().info(f"avg_state_evader: {avg_state_evader}")
            self.get_logger().info(f"goal: {self.goal_position_calibration}")
            deviation_z = avg_state_evader[0][2] - self.goal_position_calibration[0][2]
            thrust_offset = self.gain_matrix[3, 2] * deviation_z
            self.u_hover_evader[3] += thrust_offset
            self.calibration_counter += 1
            self.k_T_actual_evader = -self.Gz / self.u_hover_evader[3]

            self.get_logger().info(f"[CALIBRATION] -- Calibration deviation (evader): {deviation_z:.2f}")
            self.get_logger().info(f"[CALIBRATION] -- Thrust offset (evader): {thrust_offset:.2f}")
            self.get_logger().info(f"[CALIBRATION] -- New thrust target (evader): {self.u_hover_evader[3]:.2f}, k_T_actual (evader): {self.k_T_actual_evader:.2f}")

            if self.calibration_counter >= 4:
                self.get_logger().info(f"[CALIBRATION] DONE -- u_hover (evader): {self.u_hover_evader[3]:.2f}, k_T_actual (evader): {self.k_T_actual_evader:.2f}")
                self.calibration_timer.cancel()
                self.calibrated = True

        elif GHOST_AGENT == "evader":
            avg_state_pursuer = np.mean(np.array(self.state_pos_buffer_pursuer), axis=0)
            deviation_z = avg_state_pursuer[0][2] - self.goal_position_calibration[1][2]
            thrust_offset = self.gain_matrix[3, 2] * deviation_z
            self.u_hover_pursuer[3] += thrust_offset
            self.calibration_counter += 1
            self.k_T_actual_pursuer = -self.Gz / self.u_hover_pursuer[3]

            self.get_logger().info(f"[CALIBRATION] -- Calibration deviation (pursuer): {deviation_z:.2f}")
            self.get_logger().info(f"[CALIBRATION] -- Thrust offset (pursuer): {thrust_offset:.2f}")
            self.get_logger().info(f"[CALIBRATION] -- New thrust target (pursuer): {self.u_hover_pursuer[3]:.2f}, k_T_actual (pursuer): {self.k_T_actual_pursuer:.2f}")

            if self.calibration_counter >= 4:
                self.get_logger().info(f"[CALIBRATION] DONE -- u_hover (pursuer): {self.u_hover_pursuer[3]:.2f}, k_T_actual (pursuer): {self.k_T_actual_pursuer:.2f}")
                self.calibration_timer.cancel()
                self.calibrated = True

        elif GHOST_AGENT == "none":
            avg_state_evader = np.mean(np.array(self.state_pos_buffer_evader), axis=0)
            avg_state_pursuer = np.mean(np.array(self.state_pos_buffer_pursuer), axis=0)
            deviation_z_evader = avg_state_evader[2] - self.goal_position_calibration[0][2]
            thrust_offset_evader = self.gain_matrix[3, 2] * deviation_z_evader
            self.u_hover_evader[3] += thrust_offset_evader
            self.calibration_counter += 1
            self.k_T_actual_evader = -self.Gz / self.u_hover_evader[3]

            self.get_logger().info(f"[CALIBRATION] -- Calibration deviation (evader): {deviation_z_evader:.2f}")
            self.get_logger().info(f"[CALIBRATION] -- Thrust offset (evader): {thrust_offset_evader:.2f}")
            self.get_logger().info(f"[CALIBRATION] -- New thrust target (evader): {self.u_hover_evader[3]:.2f}, k_T_actual (evader): {self.k_T_actual_evader:.2f}")

            deviation_z_pursuer = avg_state_pursuer[2] - self.goal_position_calibration[1][2]
            thrust_offset_pursuer = self.gain_matrix[3, 2] * deviation_z_pursuer
            self.u_hover_pursuer[3] += thrust_offset_pursuer
            self.k_T_actual_pursuer = -self.Gz / self.u_hover_pursuer[3]

            self.get_logger().info(f"[CALIBRATION] -- Calibration deviation (pursuer): {deviation_z_pursuer:.2f}")
            self.get_logger().info(f"[CALIBRATION] -- Thrust offset (pursuer): {thrust_offset_pursuer:.2f}")
            self.get_logger().info(f"[CALIBRATION] -- New thrust target (pursuer): {self.u_hover_pursuer[3]:.2f}, k_T_actual (pursuer): {self.k_T_actual_pursuer:.2f}")

            if self.calibration_counter >= 10:
                self.get_logger().info(f"[CALIBRATION] DONE -- u_hover (evader): {self.u_hover_evader[3]:.2f}, k_T_actual (evader): {self.k_T_actual_evader:.2f}")
                self.get_logger().info(f"[CALIBRATION] DONE -- u_hover (pursuer): {self.u_hover_pursuer[3]:.2f}, k_T_actual (pursuer): {self.k_T_actual_pursuer:.2f}")
                self.calibration_timer.cancel()
                self.calibrated = True
    
    def __call__(self, state):
        if self.calibrated or not CALIBRATE_FIRST or GHOST_AGENT == "both":
            return self.call_pe(state)
        else:
            return self.call_lqr(state)

    def call_lqr(self, state):
        """
        LQR stabilization for calibration
        """
        states = np.array(state).reshape(self.nbr_flying_robots, -1)
        
        if GHOST_AGENT == "pursuer":
            self.state_pos_buffer_evader.append(states[0:3])
            u_hover = [self.u_hover_evader]
        elif GHOST_AGENT == "evader":
            self.state_pos_buffer_pursuer.append(states[0:3])
            u_hover = [self.u_hover_pursuer]
        elif GHOST_AGENT == "none":
            u_hover = [self.u_hover_evader, self.u_hover_pursuer]
            self.state_pos_buffer_evader.append(states[0][0:3])
            self.state_pos_buffer_pursuer.append(states[1][0:3])
        
        u = np.zeros((self.nbr_flying_robots, 4))

        for i, state in enumerate(states):
            
            euler_angles = rowan.to_euler(([state[9], state[6], state[7], state[8]]), "xyz")
            yaw = euler_angles[2]
            near_hover_state = np.concatenate([state[0:6], np.array([yaw])])
            u[i] = u_hover[i] + self.gain_matrix @ (near_hover_state - np.concatenate((self.goal_position_calibration[i], np.zeros(4))))
            u[i, :2] = np.clip(u[i, :2], -0.2, 0.2)
            u[i, 3] = np.clip(u[i, 3], 4.0, 16.0)  

        return u.flatten()
    
    def call_lqr_override(self, state, goal_position):
        """
        LQR stabilization for calibration
        """
        original_state = state.copy()
        states = np.array(state).reshape(self.nbr_flying_robots, -1)
        self.get_logger().info(" USING LQR CONTROLLER ")
        
        u_hover = [self.u_hover_evader, self.u_hover_pursuer]
        if GHOST_AGENT == "pursuer":
            states = np.array([states[0][0:6], self.ghost_state_pursuer[[0, 4, 8, 1, 5, 9]]])
        elif GHOST_AGENT == "evader":
            states = np.array([self.ghost_state_evader[[0, 4, 8, 1, 5, 9]], states[0][0:6]])
        elif GHOST_AGENT == "none":
            pass  # states is already sorted correctly
        elif GHOST_AGENT == "both":
            u_hover = np.array([self.u_hover_evader, self.u_hover_pursuer])
            self.get_logger().info(f"states RAW: {states}") # [x, y, z, vx, vy, vz, qx, qy, qz, qw, omega_x, omega_y, omega_z, ...]
            states = np.array([self.ghost_state_evader[[0, 4, 8, 1, 5, 9]], self.ghost_state_pursuer[[0, 4, 8, 1, 5, 9]]])
            self.get_logger().info(f"states DR: {states}") # [x1, v1_x, θ1_x, ω1_x, y1, v1_y, θ1_y, ω1_y, z1, v1_z]
            self.get_logger().info(f"goal_positions: {goal_position}") # [x1, v1_x, θ1_x, ω1_x, y1, v1_y, θ1_y, ω1_y, z1, v1_z]
            self.get_logger().info(f"u_hover: {u_hover}") # [x1, v1_x, θ1_x, ω1_x, y1, v1_y, θ1_y, ω1_y, z1, v1_z]
        else:
            raise ValueError(f"Unknown GHOST_AGENT {GHOST_AGENT} type for LQR")

        u = np.zeros((2, 4))

        for i, state in enumerate(states):
            if GHOST_AGENT == "pursuer" and i == 0:
                euler_angles = rowan.to_euler(([original_state[9], original_state[6], original_state[7], original_state[8]]), "xyz")
                yaw = euler_angles[2]
            elif GHOST_AGENT == "evader" and i == 1:
                euler_angles = rowan.to_euler(([original_state[9], original_state[6], original_state[7], original_state[8]]), "xyz")
                yaw = euler_angles[2]
            elif GHOST_AGENT == "none":
                euler_angles = rowan.to_euler(([state[9], state[6], state[7], state[8]]), "xyz")
                yaw = euler_angles[2]
            else:
                yaw = 0.
            near_hover_state = np.concatenate([state[0:6], np.array([yaw])])
            u[i] = u_hover[i] + self.gain_matrix @ (near_hover_state - np.concatenate((goal_position[i], np.zeros(4))))
            u[i, :2] = np.clip(u[i, :2], -0.2, 0.2)
            u[i, 3] = np.clip(u[i, 3], 4, 16.0)

        if self.emergency_lqr_override_evader_inuse:
            deviations = np.linalg.norm(states[0, 0:3] - goal_position[0])
            if np.all(deviations < LQR_OVERRIDE_EXIT_THRESH):
                self.get_logger().info("[EMERGENCY ARENA OVERRIDE] Evader LQR intervention succeeded, resuming DeepReach control")
                self.emergency_lqr_override_evader_inuse = False

        if self.emergency_lqr_override_pursuer_inuse:
            deviations = np.linalg.norm(states[1, 0:3] - goal_position[1])
            if np.all(deviations < LQR_OVERRIDE_EXIT_THRESH):
                self.get_logger().info("[EMERGENCY ARENA OVERRIDE] Pursuer LQR intervention succeeded, resuming DeepReach control")
                self.emergency_lqr_override_pursuer_inuse = False

        return u.flatten()        

    def call_pe(self, state):
        """
        Main control function for pursuit-evasion control
        
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
            # The euler angles here are follow compared to the drone convention
            roll = -euler_angles[0]   # θ_y  (post sign change: +roll = positive y acceleration)
            pitch = euler_angles[1]  # θ_x  (without sign change: +pitch = positive x acceleration)
                
        elif MODE == "deepreach":
            # Handle takeoff and hover phase
            if(self.nbr_robots != 2):
                raise ValueError("Pursuit evasion mode only supports 2 drones")

            # Extract full state information from robot states
            # Initialize 20D state with zeros for angles and angular velocities
            drone_20d_state = np.zeros(20)
            yawrates = np.zeros(len(states))
            for i, robot_state in enumerate(states):

                if GHOST_AGENT != "both":
                    # Full state format: [x, y, z, vx, vy, vz, qx, qy, qz, qw, omega_x, omega_y, omega_z, ...]
                    pos = robot_state[0:3]    # x, y, z
                    vel = robot_state[3:6]    # vx, vy, vz
                    quat_raw = robot_state[6:10]  # qx, qy, qz, qw
                    omega = robot_state[10:13] # omega_x, omega_y, omega_z
                    
                    # Convert quaternion from [qx, qy, qz, qw] to [qw, qx, qy, qz] format for rowan
                    quat = np.array([quat_raw[3], quat_raw[0], quat_raw[1], quat_raw[2]])  # [qw, qx, qy, qz]
                    # self.get_logger().info(f"agent {i} quat: {quat}, norm={np.linalg.norm(quat)}")
                    
                    # Convert quaternion to Euler angles to get roll and pitch
                    euler_angles = rowan.to_euler(quat, "xyz")
                    # The euler angles here are follow compared to the drone convention
                    roll = -euler_angles[0]   # θ_y  (post sign change: +roll = positive y acceleration)
                    pitch = euler_angles[1]  # θ_x  (without sign change: +pitch = positive x acceleration)
                    yaw = euler_angles[2]  # θ_z

                if GHOST_AGENT == "none":
                    start_iter = i * 10
                    drone_20d_state[start_iter + 0] = pos[0]  # x1
                    drone_20d_state[start_iter + 1] = vel[0]  # v1_x
                    drone_20d_state[start_iter + 2] = pitch  # θ1_x (pitch angle)
                    drone_20d_state[start_iter + 3] = omega[0]  # w1_x
                    drone_20d_state[start_iter + 4] = pos[1]  # y1
                    drone_20d_state[start_iter + 5] = vel[1]  # v1_y
                    drone_20d_state[start_iter + 6] = roll  # θ1_y (roll angle)
                    drone_20d_state[start_iter + 7] = omega[1]  # w1_y
                    drone_20d_state[start_iter + 8] = pos[2]  # z1
                    drone_20d_state[start_iter + 9] = vel[2]  # v1_z
                    yawrates[i] = 2.0 * yaw

                elif GHOST_AGENT == "pursuer":  # Live Drone 1 (evader)
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
                    yawrates[0] = 2.0 * yaw #FIXME -> bugs

                    if GHOST_PURSUER_SIMPLE:
                        # CIRCLE
                        drone_20d_state[10:20] = np.array([SIMPLE_RADIUS * np.cos(SIMPLE_FREQ * self.iteration), # x
                                                          SIMPLE_RADIUS * SIMPLE_FREQ * CONTROLLER_RATE * np.cos(SIMPLE_FREQ * self.iteration), # vx
                                                          0., 0., 
                                                          SIMPLE_RADIUS * np.sin(SIMPLE_FREQ * self.iteration), # y 
                                                          -SIMPLE_RADIUS * SIMPLE_FREQ * CONTROLLER_RATE * np.sin(SIMPLE_FREQ * self.iteration), # vy
                                                          0., 0., 
                                                          SIMPLE_HEIGHT, #z
                                                          0.])

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
                    yawrates[0] = 2.0 * yaw #FIXME -> bugs

                    if GHOST_EVADER_SIMPLE:
                        # CIRCLE
                        drone_20d_state[0:10] = np.array([SIMPLE_RADIUS * np.cos(SIMPLE_FREQ * self.iteration), # x
                                                          SIMPLE_RADIUS * SIMPLE_FREQ * CONTROLLER_RATE * np.cos(SIMPLE_FREQ * self.iteration), # vx
                                                          0., 0., 
                                                          SIMPLE_RADIUS * np.sin(SIMPLE_FREQ * self.iteration), # y 
                                                          -SIMPLE_RADIUS * SIMPLE_FREQ * CONTROLLER_RATE * np.sin(SIMPLE_FREQ * self.iteration), # vy
                                                          0., 0., 
                                                          SIMPLE_HEIGHT, #z
                                                          0.])
                        
                elif GHOST_AGENT == "both":
                    drone_20d_state[0:10] = self.ghost_state_evader
                    drone_20d_state[10:20] = self.ghost_state_pursuer

                    if GHOST_PURSUER_SIMPLE:
                        # CIRCLE
                        drone_20d_state[10:20] = np.array([SIMPLE_RADIUS * np.cos(SIMPLE_FREQ * self.iteration), # x
                                                          SIMPLE_RADIUS * SIMPLE_FREQ * CONTROLLER_RATE * np.cos(SIMPLE_FREQ * self.iteration), # vx
                                                          0., 0., 
                                                          SIMPLE_RADIUS * np.sin(SIMPLE_FREQ * self.iteration), # y 
                                                          -SIMPLE_RADIUS * SIMPLE_FREQ * CONTROLLER_RATE * np.sin(SIMPLE_FREQ * self.iteration), # vy
                                                          0., 0., 
                                                          SIMPLE_HEIGHT, #z
                                                          0.])
                        
                    if GHOST_EVADER_SIMPLE:
                        # CIRCLE
                        drone_20d_state[0:10] = np.array([SIMPLE_RADIUS * np.cos(SIMPLE_FREQ * self.iteration), # x
                                                          SIMPLE_RADIUS * SIMPLE_FREQ * CONTROLLER_RATE * np.cos(SIMPLE_FREQ * self.iteration), # vx
                                                          0., 0., 
                                                          SIMPLE_RADIUS * np.sin(SIMPLE_FREQ * self.iteration), # y 
                                                          -SIMPLE_RADIUS * SIMPLE_FREQ * CONTROLLER_RATE * np.sin(SIMPLE_FREQ * self.iteration), # vy
                                                          0., 0., 
                                                          SIMPLE_HEIGHT, #z
                                                          0.])

                else:
                    raise ValueError(f"Unknown GHOST_AGENT: {GHOST_AGENT}")
            
            ## Make Deepreach Tensor and Infer

            drone_20d_state_tensor = torch.tensor(drone_20d_state, dtype=torch.float32, device=device)
            optimal_u, optimal_d, values, ellx = self.get_control_and_disturbance(drone_20d_state_tensor)
            value = values["game"]
            if USE_EVADER_ARENA_FILTER:
                value_arena_evader = values["arena_evader"]
            if USE_PURSUER_ARENA_FILTER:
                value_arena_pursuer = values["arena_pursuer"]

            # Extract control inputs from DeepReach
            # Control: [S1_x, S1_y, T1_z] (evader)
            # Disturbance: [S2_x, S2_y, T2_z] (pursuer)

            # Scale controls to actual k_T (newer drones have been stronger)
            optimal_u[0, 2] = (self.k_T / self.k_T_actual_evader) * optimal_u[0, 2]
            optimal_d[0, 2] = (self.k_T / self.k_T_actual_pursuer) * optimal_d[0, 2] # FIXME for evader

            # Clamp controls for smoother flight
            max_torque = self.dynamics.max_torque
            max_thrust = self.dynamics.thrust_max

            if CLAMP_RPYT_CONTROLS and not WAYPOINT_CONTROL:

                raw_thrust_max = REAL_THRUST_MAX/16.
                raw_thrust_min = REAL_THRUST_MIN/16.
                raw_torque_mag = REAL_TORQUE_MAG/0.3
                 # NOTE: these could be defined wrt self.dynamics, but could change with new models
                
                min_vals = torch.tensor([-raw_torque_mag, -raw_torque_mag, raw_thrust_min], device='cuda:0')     # e.g., lower bound is 0
                max_vals = torch.tensor([ raw_torque_mag,  raw_torque_mag, raw_thrust_max], device='cuda:0')    # e.g., element-wise upper bounds

                optimal_u = torch.max(torch.min(optimal_u, max_vals), min_vals)
                # optimal_d = torch.max(torch.min(optimal_d, max_vals), min_vals)
            
            if self.in_flight:
                self.get_logger().info(f"DRONE_20d_STATE (EVADER[xp, yp]) {(drone_20d_state[0], drone_20d_state[4])}")
                self.get_logger().info(f"INITIAL OPTIMAL U {optimal_u}")
                self.get_logger().info(f"INITIAL OPTIMAL D {optimal_d}")
            
            evader_control = np.array([
                max_torque * optimal_u[0, 0].item(),  # S1_x
                max_torque * optimal_u[0, 1].item(),  # S1_y
                # self.dynamics.thrust_max * self.dynamics.k_T * optimal_u[0, 2].item()   # T1_z  # FIXME: Check whether this is correct
                max_thrust * optimal_u[0, 2].item()   # T1_z
            ])
            pursuer_control = np.array([
                max_torque * optimal_d[0, 0].item(),  # S2_x
                max_torque * optimal_d[0, 1].item(),  # S2_y
                # self.dynamics.thrust_max * self.dynamics.k_T * optimal_d[0, 2].item()   # T2_z
                max_thrust * optimal_d[0, 2].item()   # T2_z
            ])

            self.get_logger().info(f"REAL OPTIMAL U: {evader_control}")
            self.get_logger().info(f"max torque {self.dynamics.max_torque}")
            self.get_logger().info(f"max thrust {self.dynamics.thrust_max}")
            # self.get_logger().info(f"Pursuer control: {pursuer_control}")
            # self.get_logger().info(f"value: {value.item():.4f}")
            # self.get_logger().info(f"ellx: {ellx.item():.4f}")

            ## Emergency fallback to LQR if OOB
            if USE_EMERGENCY_ARENA_OVERRIDE and self.in_flight:

                # Check if either agent OOB
                if not self.emergency_lqr_override_evader_inuse:
                    if np.abs(drone_20d_state[0]).item() > ARENA_X_LIMIT or np.abs(drone_20d_state[4]).item() > ARENA_Y_LIMIT or drone_20d_state[8].item() < ARENA_Z_MIN or drone_20d_state[8].item() > ARENA_Z_MAX:
                        self.in_bounds_evader = False
                    else:
                        self.in_bounds_evader = True
                        # self.last_safe_evader_pos = [drone_20d_state[0], drone_20d_state[4], drone_20d_state[8]]
                        self.last_safe_evader_pos = [RESET_PROJ_FACTOR * drone_20d_state[0], RESET_PROJ_FACTOR * drone_20d_state[4], RESET_PROJ_FACTOR * (drone_20d_state[8] - (ARENA_Z_MAX + ARENA_Z_MIN)/2.) + (ARENA_Z_MAX + ARENA_Z_MIN)/2.]
                        # FIXME project back inwards by a factor to avoid deadlock on boundary
                
                if not self.emergency_lqr_override_pursuer_inuse:
                    if np.abs(drone_20d_state[10]).item() > ARENA_X_LIMIT or np.abs(drone_20d_state[14]).item() > ARENA_Y_LIMIT or drone_20d_state[18].item() < ARENA_Z_MIN or drone_20d_state[18].item() > ARENA_Z_MAX:
                        self.in_bounds_pursuer = False
                    else:
                        self.in_bounds_pursuer = True
                        # self.last_safe_pursuer_pos = [drone_20d_state[10], drone_20d_state[14], drone_20d_state[18]]
                        self.last_safe_pursuer_pos = [RESET_PROJ_FACTOR * drone_20d_state[10], RESET_PROJ_FACTOR * drone_20d_state[14], RESET_PROJ_FACTOR * (drone_20d_state[18] - (ARENA_Z_MAX + ARENA_Z_MIN)/2.) + (ARENA_Z_MAX + ARENA_Z_MIN)/2.]

                # Override to LQR if either agent OOB
                if not self.in_bounds_evader or not self.in_bounds_pursuer or self.emergency_lqr_override_evader_inuse or self.emergency_lqr_override_pursuer_inuse:

                    self.get_logger().info("[EMERGENCY ARENA OVERRIDE] Agent(s) exited arena domain, falling back to LQR; EVADER in? {}, PURSUER in? {}".format(self.in_bounds_evader, self.in_bounds_pursuer))
                    safe_goal_position = np.array([self.last_safe_evader_pos, self.last_safe_pursuer_pos])

                    # Update reset goal if OOB AND not already in LQR override
                    if not self.in_bounds_evader and not self.emergency_lqr_override_evader_inuse:
                        safe_goal_position[0] = self.last_safe_evader_pos
                        self.emergency_lqr_override_evader_inuse = True
                        self.get_logger().info(f"[EMERGENCY ARENA OVERRIDE] Evader exited resetting to {self.last_safe_evader_pos}")

                    if not self.in_bounds_pursuer and not self.emergency_lqr_override_pursuer_inuse:
                        safe_goal_position[1] = self.last_safe_pursuer_pos
                        if GHOST_AGENT == "evader":
                            safe_goal_position[0] = self.last_safe_pursuer_pos
                        self.emergency_lqr_override_pursuer_inuse = True
                        self.get_logger().info(f"[EMERGENCY ARENA OVERRIDE] Pursuer exited resetting to {self.last_safe_pursuer_pos}")

                    u_lqr_override_flat = self.call_lqr_override(state, safe_goal_position)
                    self.get_logger().info(f"[EMERGENCY ARENA OVERRIDE] LQR override control flat: {u_lqr_override_flat}")
                    
                    u_lqr_override = u_lqr_override_flat.reshape(2, 4)
                    self.get_logger().info(f"[EMERGENCY ARENA OVERRIDE] LQR override control: {u_lqr_override}")

                    if self.emergency_lqr_override_evader_inuse:
                        evader_control[:] = u_lqr_override[0][[1, 0, 3]]  # roll, pitch, thrust
                        evader_control[0] = -evader_control[0]  # SIGN CHANGE ? for drone convention (+pitch = negative x acceleration)
                        optimal_u = torch.tensor([(evader_control[0] / max_torque,
                                                   evader_control[1] / max_torque,
                                                   evader_control[2] / max_thrust)], device=device)

                    if self.emergency_lqr_override_pursuer_inuse:
                        pursuer_control[:] = u_lqr_override[1][[1, 0, 3]]  # roll, pitch, thrust
                        pursuer_control[0] = -pursuer_control[0]  # SIGN CHANGE ? for drone convention (+pitch = negative x acceleration)
                        optimal_d = torch.tensor([(pursuer_control[0] / max_torque,
                                                   pursuer_control[1] / max_torque,
                                                   pursuer_control[2] / max_thrust)], device=device)

            # Apply control limits for safety     # T2_z
            # FIXME: Add in that we want to control yaw again
            ## Convert DeepReach controls to Crazyflie format: [roll, pitch, yaw_rate, thrust]
            if GHOST_AGENT == "none":
                u[0, 0] = evader_control[1]  # roll  # drone convention (+roll = + y acceleration)
                u[0, 1] = -evader_control[0]  # pitch # SIGN CHANGE for drone convention (+pitch = - x acceleration)
                u[0, 2] = 0.0  # yaw_rate
                u[0, 3] = evader_control[2]  # thrust

                u[1, 0] = pursuer_control[1]  # roll  # drone convention (+roll = + y acceleration)
                u[1, 1] = -pursuer_control[0]  # pitch # SIGN CHANGE for drone convention (+pitch = - x acceleration)
                u[1, 2] = 0.0  # yaw_rate
                u[1, 3] = pursuer_control[2]  # thrust

            elif GHOST_AGENT == "pursuer":
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
                        next_state = drone_20d_state_tensor + GHOST_PURSUER_SLOW_FACTOR * self.dt * f.squeeze(0)
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
                        next_state = drone_20d_state_tensor + GHOST_EVADER_SLOW_FACTOR * self.dt * f.squeeze(0)
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

                    if LOAD_PRESOLVED_EVADER_TRAJ:
                        if self.iteration < len(self.presolved_evader_states):
                            self.get_logger().info(f"Loading presolved evader state with shape {self.presolved_evader_states[self.iteration].shape}")
                            self.ghost_state_evader = self.presolved_evader_states[self.iteration]
                        else:
                            self.ghost_state_evader = self.presolved_evader_states[-1]

                if self.iteration % 5 == 0:
                    self.evader_marker.pose.position.x = float(self.ghost_state_evader[0].item())
                    self.evader_marker.pose.position.y = float(self.ghost_state_evader[4].item())
                    self.evader_marker.pose.position.z = float(self.ghost_state_evader[8].item())
                    self.marker_pub.publish(self.evader_marker)
                    self.pursuer_marker.pose.position.x = float(self.ghost_state_pursuer[0].item())
                    self.pursuer_marker.pose.position.y = float(self.ghost_state_pursuer[4].item())
                    self.pursuer_marker.pose.position.z = float(self.ghost_state_pursuer[8].item())
                    self.marker_pub.publish(self.pursuer_marker)

            else:
                raise ValueError(f"Unknown GHOST_AGENT: {GHOST_AGENT}")
            
            # Apply final safety limits
            u[:, :2] = np.clip(u[:, :2], -0.3, 0.3)  # roll, pitch limits

            if GHOST_AGENT != "both":
                # We want to keep the yaw at 0, so yaw rate is an proporitional controller to keep yaw at 0
                u[:, 2] = yawrates

            u[:, 3] = np.clip(u[:, 3], 4.0, 16.0)    # thrust limits
                
        else:
            raise NotImplementedError(f"Mode {MODE} not implemented")

        ## WAYPOINT SOLVING AND PLANNING ##
                
        if WAYPOINT_CONTROL:
            xi, ui, di =  drone_20d_state_tensor, optimal_u, optimal_d
            
            ## Integrate from current state ##
            if self.in_flight: 
                # self.get_logger().info(f"xi[xp, yp] BEFORE: {(xi[0].item(), xi[4].item())}")
                # self.get_logger().info(f"thetax: {xi[2]}, thetay: {xi[6]}")
                f = self.dynamics.dsdt(xi, ui, di)
                xi = xi + INTEG_STRETCH_FACTOR * self.dt * f.squeeze(0) ## FIXME FE -> RK4
                # self.get_logger().info(f"xi: {xi}")

                # self.get_logger().info(f"thetax: {xi[2]}, thetay: {xi[6]}")
                for i in range(int(WP_INTEGRATION_HZN / self.dt) - 1):

                    ui, di, _, _ = self.get_control_and_disturbance(xi)
                    # self.get_logger().info(f"ui: {ui}")
                    # self.get_logger().info(f"di: {di}")
                    
                    f = self.dynamics.dsdt(xi, ui, di)
                    xi = xi + INTEG_STRETCH_FACTOR * self.dt * f.squeeze(0) ## FIXME FE -> RK4
                    # self.get_logger().info(f"theta_x: {xi[2]}, theta_y: {xi[6]}")
                #     self.get_logger().info(f"        ui         INTER: {[ui[0, j].item() for j in range(3)]}")
                #     self.get_logger().info(f"        xi[xp, yp] INTER: {(xi[0].item(), xi[4].item())}")
                # self.get_logger().info(f"xi[xp, yp] AFTER: {(xi[0].item(), xi[4].item())}")
                self.wp_iteration += 1

            # [x1, v1_x, θ1_x, ω1_x, y1, v1_y, θ1_y, ω1_y, z1, v1_z]
            # TEST CIRCLE WAYPOINTS
            # self.evader_wp_state = np.array([2. * np.cos(0.05 * self.iteration), 1. * np.cos(0.05 * self.iteration), 0., 0., 2. * np.sin(0.05 * self.iteration), -1. * np.sin(0.05 * self.iteration), 0., 0., 1., 0.])
            # self.evader_wp_state = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0.])
            # self.pursuer_wp_state = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0.])
            # if self.in_flight:  # Figure 8 implementation
            #     self.circle_iteration += 1
            #     self.evader_wp_state[0] = 2. * np.sin(0.03 * self.circle_iteration)
            #     self.evader_wp_state[4] = 2. * np.sin(2 * 0.03 * self.circle_iteration)

            if GHOST_AGENT == "pursuer":
                self.evader_wp_state = xi[0:10].cpu().numpy()  # Compute next evader state
                self.pursuer_wp_state = self.ghost_state_pursuer  # Compute next evader state

            elif GHOST_AGENT == "evader":
                self.evader_wp_state = self.ghost_state_evader  # Compute next evader state
                self.pursuer_wp_state = xi[10:20].cpu().numpy()  # Compute next evader state

            elif GHOST_AGENT == "both":
                self.evader_wp_state = self.ghost_state_evader  # Compute next evader state
                self.pursuer_wp_state = self.ghost_state_pursuer  # Compute next evader state

            elif GHOST_AGENT == "none":
                self.evader_wp_state = xi[0:10].cpu().numpy()  # Compute next evader state
                self.pursuer_wp_state = xi[10:20].cpu().numpy()  # Compute next evader state

            else:
                raise AssertionError(f"Ghost Agent {GHOST_AGENT} unrecognized!")

            # FIX ANGLES FOR NOW
            # self.evader_wp_state[2] = 0.
            # self.evader_wp_state[3] = 0.
            # self.evader_wp_state[6] = 0.
            # self.evader_wp_state[7] = 0.

            ## Update markers
            self.evader_wp_marker.pose.position.x = float(self.evader_wp_state[0].item())
            self.evader_wp_marker.pose.position.y = float(self.evader_wp_state[4].item())
            self.evader_wp_marker.pose.position.z = float(self.evader_wp_state[8].item())
            self.marker_pub.publish(self.evader_wp_marker)
            self.pursuer_wp_marker.pose.position.x = float(self.pursuer_wp_state[0].item())
            self.pursuer_wp_marker.pose.position.y = float(self.pursuer_wp_state[4].item())
            self.pursuer_wp_marker.pose.position.z = float(self.pursuer_wp_state[8].item())
            self.marker_pub.publish(self.pursuer_wp_marker)

            ## Convert to 
            desired_yaw, desired_yaw_dot = 0., 0.
            evader_quat = rowan.from_euler(-self.evader_wp_state[6], self.evader_wp_state[2], desired_yaw, "xyz") # TODO does this end in w?
            pursuer_quat = rowan.from_euler(-self.pursuer_wp_state[6], self.pursuer_wp_state[2], desired_yaw, "xyz") # TODO does this end in w?
            
            # self.get_logger().info(f"Evader wp state: {self.evader_wp_state}")
            # self.get_logger().info(f"evader quat: {evader_quat}, norm={np.linalg.norm(evader_quat)}")

            # if np.linalg.norm(evader_quat) != 1.:
            #     raise AssertionError(f"Sending Bad quaternion: quat={evader_quat}, norm={np.linalg.norm(evader_quat)}")

            ## Convert to ctrl_msg format
            u = np.zeros((self.nbr_flying_robots, 16))

            if GHOST_AGENT == "none":
                u[0, 0] = self.evader_wp_state[0]  # x
                u[0, 1] = self.evader_wp_state[4]  # y
                u[0, 2] = self.evader_wp_state[8]  # z
                u[0, 3] = self.evader_wp_state[1]  # vx
                u[0, 4] = self.evader_wp_state[5]  # vy
                u[0, 5] = self.evader_wp_state[9]  # vz
                u[0, 6:10] = evader_quat
                u[0, 10] = self.evader_wp_state[3]  # wx
                u[0, 11] = self.evader_wp_state[7]  # wy
                u[0, 12] = desired_yaw_dot  # wz

                u[1, 0] = self.pursuer_wp_state[0]  # x
                u[1, 1] = self.pursuer_wp_state[4]  # y
                u[1, 2] = self.pursuer_wp_state[8]  # z
                u[1, 3] = self.pursuer_wp_state[1]  # vx
                u[1, 4] = self.pursuer_wp_state[5]  # vy
                u[1, 5] = self.pursuer_wp_state[9]  # vz
                u[1, 6:10] = pursuer_quat
                u[1, 10] = self.pursuer_wp_state[3]  # wx
                u[1, 11] = self.pursuer_wp_state[7]  # wy
                u[1, 12] = desired_yaw_dot  # wz

            if GHOST_AGENT == "pursuer":
                u[0, 0] = self.evader_wp_state[0] # x
                u[0, 1] = self.evader_wp_state[4] # y
                u[0, 2] = self.evader_wp_state[8] # z
                u[0, 3] = self.evader_wp_state[1] # vx
                u[0, 4] = self.evader_wp_state[5] # vy
                u[0, 5] = self.evader_wp_state[9] # vz
                u[0, 6:10] = evader_quat
                u[0, 10] = self.evader_wp_state[3] # wx
                u[0, 11] = self.evader_wp_state[7] # wy
                u[0, 12] = desired_yaw_dot # wz

            if GHOST_AGENT == "evader":
                u[0, 0] = self.pursuer_wp_state[0] # x
                u[0, 1] = self.pursuer_wp_state[4] # y
                u[0, 2] = self.pursuer_wp_state[8] # z
                u[0, 3] = self.pursuer_wp_state[1] # vx
                u[0, 4] = self.pursuer_wp_state[5] # vy
                u[0, 5] = self.pursuer_wp_state[9] # vz
                u[0, 6:10] = pursuer_quat
                u[0, 10] = self.pursuer_wp_state[3] # wx
                u[0, 11] = self.pursuer_wp_state[7] # wy
                u[0, 12] = desired_yaw_dot # wz
            # [x1, v1_x, θ1_x, ω1_x, y1, v1_y, θ1_y, ω1_y, z1, v1_z]

            # TODO: could try estiamting accel u[..., 13:16] from integration finite-diff for smoother flight

        # self.get_logger().info(f"Control: {u}")
        # self.get_logger().info(f"Position: {state[0:3]}")
        # self.get_logger().info(f"Roll for DR: {roll:.2f}, Pitch for DR: {pitch:.2f}")

        ## WRITE TO JSON ##

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

            if(xydist < 0.25 and z_dist < 0.75):  # FIXME: I don't think this actually tracks the collision anymore
                # Track first collision warning time
                if self.first_collision_warning_time is None:
                    self.first_collision_warning_time = time.time() - self.start_time
                # self.get_logger().info(f"Collision warning! Distance: {xydist:.2f} m in xy and {z_dist:.2f} m in z direction")

            if np.any(actual_pos1 < box_min) or np.any(actual_pos1 > box_max):
                # Track first out of bounds time
                if self.first_out_of_bounds_time is None:
                    self.first_out_of_bounds_time = time.time() - self.start_time
                # self.get_logger().warn(
                #     f"Evader OUT OF BOUNDS: position {actual_pos1}"
                # )

            if self.iteration % 5 == 0:  # Log every 50 iterations (about once per second)
                pass
                # self.get_logger().info(f"DeepReach 20D Mode - Iteration {self.iteration}: Current positions for {self.nbr_robots} robots")

                # self.get_logger().info(f"Evader actual position: [{actual_pos1[0]:.2f}, {actual_pos1[1]:.2f}, {actual_pos1[2]:.2f}]")
                # self.get_logger().info(f"Evader actual velocity: [{actual_vel1[0]:.2f}, {actual_vel1[1]:.2f}, {actual_vel1[2]:.2f}]")
                # self.get_logger().info(f"Pursuer actual position: [{actual_pos2[0]:.2f}, {actual_pos2[1]:.2f}, {actual_pos2[2]:.2f}]")
                # self.get_logger().info(f"Pursuer actual velocity: [{actual_vel2[0]:.2f}, {actual_vel2[1]:.2f}, {actual_vel2[2]:.2f}]")

                # if self.first_collision_warning_time is not None or self.first_out_of_bounds_time is not None:
                    # self.get_logger().warn("Safety event detected - ending control loop")

            log_entry = {
                "timestamp": time.time() - self.start_time,  # Relative time in seconds
                "evader": {
                    "actual_position": actual_pos1.tolist(),
                    "actual_velocity": actual_vel1.tolist(),
                    "control": evader_control.tolist() if MODE == "deepreach" else None,
                    "full_state": drone_20d_state[0:10].tolist(),
                    "waypoint": self.evader_wp_state.tolist() if WAYPOINT_CONTROL else None
                },
                "pursuer": {
                    "actual_position": actual_pos2.tolist(),
                    "actual_velocity": actual_vel2.tolist(),
                    "control": pursuer_control.tolist() if MODE == "deepreach" else None,
                    "full_state": drone_20d_state[10:20].tolist(),
                    "waypoint": self.pursuer_wp_state.tolist() if WAYPOINT_CONTROL else None
                },
                "distances": {
                    "xy_distance": float(xydist),
                    "z_distance": float(z_dist),
                    "l_x": ellx.item(),
                    "V_PE": value.item(),
                    "V_A_pursuer": value_arena_pursuer.item() if USE_PURSUER_ARENA_FILTER else None,
                    "V_A_evader": value_arena_evader.item() if USE_EVADER_ARENA_FILTER else None,
                },
                "events": {
                    "first_collision_warning_time": self.first_collision_warning_time,
                    "first_out_of_bounds_time": self.first_out_of_bounds_time
                }
            }
            self.log_data.append(log_entry)

        # if self.in_flight:
        #     self.get_logger().info(f"WP ITER: {self.wp_iteration}")
            # if self.wp_iteration == 5:
            #     raise KeyboardInterrupt

        self.iteration += 1
        return u.flatten() # FIXME do we still want to flatten for WAYPOINT_CONTROL

    def save_log_file(self):
        """Save the logged data to a JSON file."""
        if self.log_data:
            try:
                with open(self.log_filename, 'w') as f:
                    json.dump({
                        "metadata": {
                            "timestamp": datetime.now().isoformat(),
                            "total_iterations": len(self.log_data),

                            "controller_rate": CONTROLLER_RATE,
                            "mode": MODE,
                            "ghost_agent": GHOST_AGENT,
                            "model_path": TWOPLAYER_MODEL_NAME,
                            
                            "use_pursuer_arena_filter": USE_PURSUER_ARENA_FILTER,
                            "use_evader_arena_filter": USE_EVADER_ARENA_FILTER,
                            "evader_arena_value_threshold": EVADER_ARENA_VALUE_THRESHOLD,
                            "pursuer_arena_value_threshold": PURSUER_ARENA_VALUE_THRESHOLD,

                            "use_follow_filter": USE_FOLLOW_FILTER,
                            "follow_value_threshold": FOLLOW_VALUE_THRESHOLD,

                            "waypoint_control": WAYPOINT_CONTROL,
                            "wp_integration_hzn": WP_INTEGRATION_HZN,
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