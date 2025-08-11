#!/usr/bin/env python3
import rclpy
import numpy as np
import torch
import rowan
import sys
import os
import time

# Add the scripts directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crazyflie_interface_py.template_controller import TemplateController
from deepreach.utils.modules import SingleBVPNet
from deepreach.dynamics import Drone6DWithDist

# Set device for PyTorch
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Model path - update this to your actual model path
MODEL_PATH = "/mounted_volume/ros2_ws/src/crazyflie_interface/scripts/drone_6d.pth"

# numpy logging only 2 digits
np.set_printoptions(precision=2, suppress=True, floatmode='fixed')

MODE = ["hover", "figure8", "circle", "deepreach"][3]  # Default to circle mode


# u: u[0] = p_x, u[1] = p_y, u[2] = p_z, u[3] = v_x, u[4] = v_y, u[5] = v_z, u[6:10] = quaternion, u[10] = omega_x, u[11] = omega_y, u[12] = omega_z, 
#    u[13] = acc_x, u[14] = acc_y, u[15] = acc_z
class FullStateController(TemplateController):
    def __init__(self, node_name='fullstate_controller'):
        # Set the full state control topic
        self.control_publisher_topic = 'cf_interface/control_full_state'
        
        super().__init__(node_name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True, controller_rate=10.0)
        
        # Get robot parameters
        self._ros_parameters = self._param_to_dict(self._parameters)
        robots = self._ros_parameters.get('robots', {})
        self.get_logger().info(f"Robots: {robots}")
        self.nbr_robots = len(robots)
        self.get_logger().info(f"Number of robots: {self.nbr_robots}")
        
        # Initialize DeepReach components if using deepreach mode
        if MODE == "deepreach":
            self.dynamics = Drone6DWithDist(thrust_max=12.0, disturbance_max=1.0, set_mode='avoid')

            self.model = SingleBVPNet(
                in_features=7,
                hidden_features=256,
                num_hidden_layers=3,
                out_features=1,
                type='sine',
                periodic_transform_fn=self.dynamics.periodic_transform_fn 
            )

            checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
            self.model.load_state_dict(checkpoint["model"])
            self.model.to(device)
            self.model.eval()
            self.get_logger().info("DeepReach model loaded successfully")
        
        # Takeoff detection
        self.takeoff_complete = False
        self.takeoff_height = 0.5  # Height threshold for takeoff completion
        self.takeoff_time = 3.0    # Time to wait after reaching height
        self.takeoff_start_time = None
        
        self.start_controller()
        self.iteration = 0
        
        # Initialize state history for RK4 integration
        self.dt = 1.0/self.controller_rate # Control period (50Hz)
        #self.dt = 0.5 # Control period (50Hz)

        self.get_logger().info(f"Control period: {self.dt}")
        
        # Initialize initial positions (will be set on first state update)
        self.initial_positions = None
        self.positions_initialized = False
        
        self.get_logger().info(f"FullState Controller initialized in {MODE} mode")

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

    def rk4_integrate(self, current_state, acceleration, dt, steps=1):
        """
        RK4 integrator to convert acceleration inputs to position and velocity
        
        Args:
            current_state: [x, y, z, vx, vy, vz] - current 6D state
            acceleration: [ax, ay, az] - acceleration inputs
            dt: time step
            steps: number of integration steps
            
        Returns:
            next_state: [x, y, z, vx, vy, vz] - integrated 6D state
        """
        def dynamics(state, acc):
            """Simple double integrator dynamics"""
            pos = state[:3]
            vel = state[3:6]
            return np.concatenate([vel, acc])
        
        state = current_state.copy()
        step_dt = dt / steps
        
        for _ in range(steps):
            # RK4 integration
            k1 = dynamics(state, acceleration)
            k2 = dynamics(state + 0.5 * step_dt * k1, acceleration)
            k3 = dynamics(state + 0.5 * step_dt * k2, acceleration)
            k4 = dynamics(state + step_dt * k3, acceleration)
            
            state = state + (step_dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        return state

    def __call__(self, state):
        
        """
        Main control function for full state control
        
        Args:
            state: Array containing state data for all robots
                   Format: [x, y, z, vx, vy, vz, qx, qy, qz, qw, ...] for each robot
        
        Returns:
            control: Flattened array of full state control inputs for all robots
                     Format: 16 elements per robot [pos(3), vel(3), quat(4), omega(3), acc(3)]
        """
        states = np.array(state).reshape(self.nbr_robots, -1)
        u = np.zeros((states.shape[0], 16))
        u[:, 9] = 1.0  # w -> zero rotation by default (unit quaternion)
        
        # Capture initial positions on first call
        if not self.positions_initialized and len(states) > 0:
            self.initial_positions = np.zeros((self.nbr_robots, 3))
            for i in range(self.nbr_robots):
                self.initial_positions[i] = states[i][0:3]  # x, y, z
            self.positions_initialized = True
            self.get_logger().info(f"Initial positions captured: {self.initial_positions}")
        
        # # Check takeoff status for first robot
        # if not self.takeoff_complete and len(states) > 0:
        #     pos = states[0][0:3]  # First robot position
        #     current_height = pos[2]
            
        #     # Check if we've reached takeoff height
        #     if current_height >= self.takeoff_height:
        #         if self.takeoff_start_time is None:
        #             self.takeoff_start_time = self.get_clock().now().nanoseconds / 1e9
        #             self.get_logger().info(f"Takeoff height reached ({current_height:.2f}m), waiting {self.takeoff_time}s for stabilization")
                
        #         # Check if we've waited long enough
        #         current_time = self.get_clock().now().nanoseconds / 1e9
        #         if current_time - self.takeoff_start_time >= self.takeoff_time:
        #             self.takeoff_complete = True
        #             self.get_logger().info("Takeoff complete! Starting FullState controller")
            
        #     # During takeoff, just send hover commands
        #     if not self.takeoff_complete:
        #         for i in range(self.nbr_robots):
        #             if self.positions_initialized:
        #                 for i, state in enumerate(states):
        #                     # Hover above initial position with offset height
        #                     hover_height = 1.0  # meters above initial position
        #                     u[i, 0:3] = np.array([
        #                         self.initial_positions[i, 0],  # x - same as initial
        #                         self.initial_positions[i, 1],  # y - same as initial  
        #                         self.initial_positions[i, 2] + hover_height  # z - initial + offset
        #                     ])
        #             else:
        #                 u[i, 0:3] = np.array([0.0, 0.0, 1.0])  # Hover at height 1.0m
        #         return u.flatten()
        
        if MODE == "hover":
            # Fallback if positions not yet initialized
            for i, state in enumerate(states):
                u[i, 0:3] = np.array([0.0, 0.0, 1.0])  # Default hover position
                
        elif MODE == "circle":
            for i, state in enumerate(states):
                angle = 2 * np.pi * self.iteration / 1000
                radius = 1.5
                u[i, 0:3] = np.array([radius * np.cos(angle), radius * np.sin(angle), 1.0 + 1.0 * i])
                
        elif MODE == "deepreach":
            # Process each robot with DeepReach
            for i, robot_state in enumerate(states):
                start_time = time.time()
                # Extract position and velocity
                pos = robot_state[0:3]  # x, y, z
                vel = robot_state[3:6]  # vx, vy, vz
                
                # Extract quaternion and convert to Euler angles
                quat = [robot_state[9], robot_state[6], robot_state[7], robot_state[8]]  # qw, qx, qy, qz                
                
                # Construct 10D state for DeepReach: [x, v_x, θ_x, ω_x, y, v_y, θ_y, ω_y, z, v_z]
                drone_6d_state = np.array([
                    pos[0],    # x
                    vel[0],    # v_x
                    pos[1],    # y
                    vel[1],    # v_y
                    pos[2],    # z
                    vel[2]     # v_z
                ])

                # Convert to tensor for DeepReach
                drone_6d_state_tensor = torch.tensor(drone_6d_state, dtype=torch.float32, device=device)
                
                # Add time dimension for DeepReach input: [time, state]
                time_tensor = torch.tensor([1.0], dtype=torch.float32, device=device)
                deepreach_input = torch.cat([time_tensor, drone_6d_state_tensor]).unsqueeze(0)  # [1, 11]

                traj_policy_results = self.model(
                    {"coords": self.dynamics.coord_to_input(deepreach_input)}
                )

                model_out = traj_policy_results["model_out"]
                model_in = traj_policy_results["model_in"]

                # Ensure proper shape for dynamics calculations
                if model_out.dim() == 1:
                    model_out = model_out.unsqueeze(0)  # [256] -> [1, 256]

                dv = self.dynamics.io_to_dv(
                    model_in,
                    model_out.squeeze(dim=-1),
                ).detach()

                value = self.dynamics.io_to_value(
                    model_in,
                    model_out.squeeze(dim=-1),
                ).detach()

                end_time = time.time()
                #self.get_logger().info(f"DeepReach time: {end_time - start_time}")

                # Use gradient to compute optimal control and disturbance
                optimal_u = self.dynamics.optimal_control(drone_6d_state_tensor, dv[..., 1:])
                d = self.dynamics.optimal_disturbance(drone_6d_state_tensor, dv[..., 1:])

                # Extract acceleration inputs from DeepReach
                acceleration = np.array([
                    self.dynamics.sideways_multiplier * optimal_u[0, 0].item(),  # ax
                    self.dynamics.sideways_multiplier * optimal_u[0, 1].item(),  # ay
                    self.dynamics.input_multiplier * optimal_u[0, 2].item() - 9.81   # az
                ])
                
                # Apply acceleration limits for safety
                max_acc = 2.0  # m/s^2
                acceleration = np.clip(acceleration, -max_acc, max_acc)
                
                # Get current 6D state for this robot
                current_6d_state = np.array([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]])
                
                # Use RK4 integrator to get next state (reduced steps for faster response)
                next_6d_state = self.rk4_integrate(current_6d_state, acceleration, self.dt, steps=1)
                
                # Extract integrated position and velocity
                integrated_pos = next_6d_state[:3]
                integrated_vel = next_6d_state[3:6]
                
                # Apply safety limits
                max_vel = 2.0  # m/s
                max_pos = 5.0  # m
                
                # Clip integrated values
                integrated_pos = np.clip(integrated_pos, -max_pos, max_pos)
                integrated_vel = np.clip(integrated_vel, -max_vel, max_vel)
                vel = np.array([1.0, 1.0, 1.0])
                # Set full state control outputs
                u[i, 0:3] = integrated_pos  # Position
                u[i, 3:6] = integrated_vel  # Velocity
                
                # Log DeepReach outputs
                self.get_logger().info(f"Robot {i} - DeepReach acceleration: {acceleration}", throttle_duration_sec=1.0)
                self.get_logger().info(f"Robot {i} - Integrated position: {integrated_pos}", throttle_duration_sec=1.0)
                self.get_logger().info(f"Robot {i} - Integrated velocity: {integrated_vel}", throttle_duration_sec=1.0)
                
        else:
            raise NotImplementedError(f"Mode {MODE} not implemented")
            
        self.iteration += 1
        
        # Log the actual positions of the robots
        if self.iteration % 50 == 0:  # Log every 50 iterations (about once per second)
            self.get_logger().info(f"FullState Mode - Iteration {self.iteration}: Current positions for {self.nbr_robots} robots")
            for i in range(self.nbr_robots):
                actual_pos = states[i, 0:3]  # Current actual position
                actual_vel = states[i, 3:6]  # Current actual velocity
                self.get_logger().info(f"Robot {i} actual position: [{actual_pos[0]:.2f}, {actual_pos[1]:.2f}, {actual_pos[2]:.2f}]")
                self.get_logger().info(f"Robot {i} actual velocity: [{actual_vel[0]:.2f}, {actual_vel[1]:.2f}, {actual_vel[2]:.2f}]")
                self.get_logger().info(f"Robot {i} actual acceleration: [{acceleration[0]:.2f}, {acceleration[1]:.2f}, {acceleration[2]:.2f}]")


      
        return u.flatten()


def main(args=None):
    rclpy.init(args=args)
    controller = FullStateController()
    rclpy.spin(controller)
    rclpy.shutdown()


if __name__ == "__main__":
    main() 