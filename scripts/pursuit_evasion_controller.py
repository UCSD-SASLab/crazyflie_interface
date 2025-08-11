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
from deepreach.dynamics import DronePursuitEvasion12D

# Set device for PyTorch
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Model path - update this to your actual model path
MODEL_PATH = "/mounted_volume/ros2_ws/src/crazyflie_interface/scripts/12d_drones.pth"

# numpy logging only 2 digits
np.set_printoptions(precision=2, suppress=True, floatmode='fixed')

MODE = ["hover", "deepreach"][1]  # Default to circle mode


# u: u[0] = p_x, u[1] = p_y, u[2] = p_z, u[3] = v_x, u[4] = v_y, u[5] = v_z, u[6:10] = quaternion, u[10] = omega_x, u[11] = omega_y, u[12] = omega_z, 
#    u[13] = acc_x, u[14] = acc_y, u[15] = acc_z
class FullStateController(TemplateController):
    def __init__(self, node_name='fullstate_controller'):
        # Set the full state control topic
        self.control_publisher_topic = 'cf_interface/control_full_state'
        
        super().__init__(node_name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        
        # Get robot parameters
        self._ros_parameters = self._param_to_dict(self._parameters)
        robots = self._ros_parameters.get('robots', {})
        self.get_logger().info(f"Robots: {robots}")
        self.nbr_robots = len(robots)
        self.get_logger().info(f"Number of robots: {self.nbr_robots}")
        
        # Initialize DeepReach components if using deepreach mode
        if MODE == "deepreach":
            self.dynamics = DronePursuitEvasion12D(collisionR = 0.25, thrust_max=14.0, set_mode='avoid')

            self.model = SingleBVPNet(
                in_features=13,
                hidden_features=512,
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
        
        self.start_controller()
        self.iteration = 0
        
        # Initialize state history for RK4 integration
        self.dt = 1.0/self.controller_rate # Control period (50Hz)
        #self.dt = 0.5 # Control period (50Hz)

        self.get_logger().info(f"Control period: {self.dt}")
      
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
        """RK4 integrator to convert acceleration inputs to position and velocity.
        
        Args:
            current_state: [x, vx, y, vy, z, vz] - interleaved position and velocity
            acceleration: [ax, ay, az] - acceleration inputs
            dt: time step
            steps: number of integration steps
            
        Returns:
            next_state: [x, vx, y, vy, z, vz] - integrated state in same format
        """
        def dynamics(state, acc):
            """Simple double integrator dynamics for [x, vx, y, vy, z, vz] format"""
            # Extract positions and velocities from interleaved format
            x, vx, y, vy, z, vz = state
            ax, ay, az = acc
            
            # Return derivatives: [dx/dt, dvx/dt, dy/dt, dvy/dt, dz/dt, dvz/dt]
            return np.array([vx, ax, vy, ay, vz, az])
        
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
        
        if MODE == "hover":
            # Fallback if positions not yet initialized
            for i, state in enumerate(states):
                u[i, 0:3] = np.array([0.0, 0.0, (i + 1) * 1.0])  # Default hover position
                
        elif MODE == "deepreach":
            # Handle takeoff and hover phase
            

            if(self.nbr_robots != 2):
                raise ValueError("Pursuit evasion mode only supports 2 drones")

            # 12d_state = [x1, y1, z1, vx1, vy1, vz1, x2, y2, z2, vx2, vy2, vz2]
            drone_12d_state = []
            
            for i, robot_state in enumerate(states):
                # Extract position and velocity from [x, y, z, vx, vy, vz] format
                pos = robot_state[0:3]  # x, y, z
                vel = robot_state[3:6]  # vx, vy, vz
                
                # Convert to DeepReach format: [x, vx, y, vy, z, vz] (interleaved)
                drone_6d_state = np.array([
                    pos[0],    # x
                    vel[0],    # vx
                    pos[1],    # y
                    vel[1],    # vy
                    pos[2],    # z
                    vel[2]     # vz
                ])
                
                drone_12d_state.append(drone_6d_state)
            
            # Combine both drone states into a single 12D state
            combined_12d_state = np.concatenate(drone_12d_state)  # [12] = [6] + [6]
            
            # Convert to tensor for DeepReach
            drone_12d_state_tensor = torch.tensor(combined_12d_state, dtype=torch.float32, device=device)
            
            # Add time dimension for DeepReach input: [time, state]
            time_tensor = torch.tensor([1.0], dtype=torch.float32, device=device)
            deepreach_input = torch.cat([time_tensor, drone_12d_state_tensor]).unsqueeze(0)  # [1, 13]
            
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
            
            # Use gradient to compute optimal control and disturbance
            optimal_u = self.dynamics.optimal_control(drone_12d_state_tensor, dv[..., 1:])
            optimal_d = self.dynamics.optimal_disturbance(drone_12d_state_tensor, dv[..., 1:])
            
            # Extract acceleration inputs from DeepReach
            acceleration1 = np.array([
                self.dynamics.sideways_multiplier * optimal_u[0, 0].item(),  # ax
                self.dynamics.sideways_multiplier * optimal_u[0, 1].item(),  # ay
                self.dynamics.input_multiplier * optimal_u[0, 2].item() - 9.81   # az
            ])
            
            acceleration2 = np.array([
                self.dynamics.sideways_multiplier * optimal_d[0, 0].item(),  # ax
                self.dynamics.sideways_multiplier * optimal_d[0, 1].item(),  # ay
                self.dynamics.input_multiplier * optimal_d[0, 2].item() - 9.81   # az
            ])

        
            
            # Apply acceleration limits for safety
            max_acc = 2.0  # m/s^2
            acceleration1 = np.clip(acceleration1, -max_acc, max_acc)
            acceleration2 = np.clip(acceleration2, -max_acc, max_acc)
            
            # Get current 6D state for each drone (already in DeepReach format: [x, vx, y, vy, z, vz])
            current_drone1_state = combined_12d_state[:6]   # First 6 elements: [x1, vx1, y1, vy1, z1, vz1]
            current_drone2_state = combined_12d_state[6:12] # Last 6 elements: [x2, vx2, y2, vy2, z2, vz2]

        
            # Use RK4 integrator to get next state
            next_drone1_state = self.rk4_integrate(current_drone1_state, acceleration1, self.dt, steps=1)
            next_drone2_state = self.rk4_integrate(current_drone2_state, acceleration2, self.dt, steps=1)

        
            # Extract integrated position and velocity from DeepReach format: [x, vx, y, vy, z, vz]
            integrated_pos1 = np.array([next_drone1_state[0], next_drone1_state[2], next_drone1_state[4]])  # [x, y, z]
            integrated_vel1 = np.array([next_drone1_state[1], next_drone1_state[3], next_drone1_state[5]])  # [vx, vy, vz]
            integrated_pos2 = np.array([next_drone2_state[0], next_drone2_state[2], next_drone2_state[4]])  # [x, y, z]
            integrated_vel2 = np.array([next_drone2_state[1], next_drone2_state[3], next_drone2_state[5]])  # [vx, vy, vz]
            
            # Apply safety limits
            max_vel = 2.0  # m/s
            max_pos = 3.0  # m
            
            # Clip integrated values
            integrated_pos1 = np.clip(integrated_pos1, -max_pos, max_pos)
            integrated_vel1 = np.clip(integrated_vel1, -max_vel, max_vel)
            integrated_pos2 = np.clip(integrated_pos2, -max_pos, max_pos)
            integrated_vel2 = np.clip(integrated_vel2, -max_vel, max_vel)
            
            # Set full state control outputs
            u[0, 0:3] = integrated_pos1  # Position
            u[0, 3:6] = integrated_vel1  # Velocity

            #u[0, 0:3] = [1,1,1]  # Position

            u[1, 0:3] = integrated_pos2  # Position
            u[1, 3:6] = integrated_vel2  # Velocity

            #self.get_logger().info("stationary pursuer")
            #u[1, 0:3] = np.array([current_drone2_state[0],current_drone2_state[2],current_drone2_state[4]])  # Position

            # u[1, 0:3] = np.array([current_drone1_state[0]+0.002,current_drone1_state[2],current_drone1_state[4]])  # Position
            # u[1, 3:6] = np.array([0.004,0.0,0.0])  # Velocity
            
            # Log DeepReach outputs
            self.get_logger().info(f"Evader - Integrated position: {integrated_pos1}")
            self.get_logger().info(f"Evader - Integrated velocity: {integrated_vel1}")
            self.get_logger().info(f"Pursuer - DeepReach acceleration: {acceleration2}")
            self.get_logger().info(f"Pursuer - Integrated position: {integrated_pos2}")
            self.get_logger().info(f"Pursuer - Integrated velocity: {integrated_vel2}")
                
        else:
            raise NotImplementedError(f"Mode {MODE} not implemented")
            
        self.iteration += 1
        
        # Log the actual positions of the robots
        if self.iteration % 50 == 0:  # Log every 50 iterations (about once per second)
            self.get_logger().info(f"FullState Mode - Iteration {self.iteration}: Current positions for {self.nbr_robots} robots")
            
            # Extract actual positions and velocities from [x, y, z, vx, vy, vz] format
            actual_pos1 = states[0, 0:3]  # Current actual position [x, y, z]
            actual_vel1 = states[0, 3:6]  # Current actual velocity [vx, vy, vz]
            actual_pos2 = states[1, 0:3]  # Current actual position [x, y, z]
            actual_vel2 = states[1, 3:6]  # Current actual velocity [vx, vy, vz]
            self.get_logger().info(f"Evader actual position: [{actual_pos1[0]:.2f}, {actual_pos1[1]:.2f}, {actual_pos1[2]:.2f}]")
            self.get_logger().info(f"Evader actual velocity: [{actual_vel1[0]:.2f}, {actual_vel1[1]:.2f}, {actual_vel1[2]:.2f}]")
            self.get_logger().info(f"Pursuer actual position: [{actual_pos2[0]:.2f}, {actual_pos2[1]:.2f}, {actual_pos2[2]:.2f}]")
            self.get_logger().info(f"Pursuer actual velocity: [{actual_vel2[0]:.2f}, {actual_vel2[1]:.2f}, {actual_vel2[2]:.2f}]")


      
        return u.flatten()


def main(args=None):
    rclpy.init(args=args)
    controller = FullStateController()
    rclpy.spin(controller)
    rclpy.shutdown()


if __name__ == "__main__":
    main() 