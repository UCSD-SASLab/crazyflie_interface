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

# Add the scripts directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crazyflie_interface_py.template_controller import TemplateController
from deepreach.utils.modules import SingleBVPNet
from deepreach.dynamics import DronePursuitEvasion12D, DronePursuitEvasion12DPure
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

# Model path - update this to your actual model path
MODEL_PATH = "/mounted_volume/ros2_ws/src/crazyflie_interface/scripts/dr_models/12d_aug27"

# numpy logging only 2 digits
np.set_printoptions(precision=2, suppress=True, floatmode='fixed')

MODE = ["hover", "deepreach"][1]  # Default to circle mode
GHOST_AGENT = ["pursuer", "evader", "both"][2]
GHOST_CONTROL_MODE = ["hover", "circle", "deepreach"][0]  # How to control the ghost agent


# u: u[0] = p_x, u[1] = p_y, u[2] = p_z, u[3] = v_x, u[4] = v_y, u[5] = v_z, u[6:10] = quaternion, u[10] = omega_x, u[11] = omega_y, u[12] = omega_z, 
#    u[13] = acc_x, u[14] = acc_y, u[15] = acc_z
class FullStateControllerGhost(TemplateController):
    def __init__(self, node_name='fullstate_controller_ghost'):
        # Set the full state control topic
        self.control_publisher_topic = 'cf_interface/control_full_state'
        
        super().__init__(node_name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

        self.ghost_state = np.zeros(16)  # Initialize ghost state
        self.ghost_state_evader[[0,1,2]] = np.array([-0.25, -0.25, 1.0])  # Initial ghost position
        self.ghost_state_pursuer = np.zeros(16)
        self.ghost_state_pursuer[[0,1,2]] = np.array([0.0, 0.0, 1.0])  # Initial ghost position
        # Get robot parameters
        self._ros_parameters = self._param_to_dict(self._parameters)
        robots = self._ros_parameters.get('robots', {})
        self.get_logger().info(f"Robots: {robots}")
        self.nbr_flying_robots = len(robots)
        self.nbr_robots = self.nbr_flying_robots + 1
        self.get_logger().info(f"Number of robots (including ghost): {self.nbr_robots}")
        # Create marker publisher for ghost visualization
        self.marker_pub = self.create_publisher(Marker, 'ghost_evader_marker', 1)
        
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
        self.evader_marker.pose.position.y = float(self.ghost_state_evader[1].item())
        self.evader_marker.pose.position.z = float(self.ghost_state_evader[2].item())
        self.evader_marker.id = 0
        self.marker_pub.publish(self.evader_marker)

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
        self.pursuer_marker.pose.position.y = float(self.ghost_state_pursuer[1].item())
        self.pursuer_marker.pose.position.z = float(self.ghost_state_pursuer[2].item())
        self.pursuer_marker.id = 1
        self.marker_pub.publish(self.pursuer_marker)

        with open(os.path.join(MODEL_PATH, "orig_opt.pickle"), 'rb') as f:
            self.orig_opt = pickle.load(f)
        
        # Initialize DeepReach components if using deepreach mode
        if MODE == "deepreach":
            #self.dynamics = DronePursuitEvasion12DPure(collisionR = 0.25, set_mode='avoid')
            self.dynamics = DronePursuitEvasion12D(collisionR = 0.25, set_mode='avoid')


            self.model = SingleBVPNet(
                in_features=13,
                hidden_features=512,
                num_hidden_layers=3,
                out_features=1,
                type='sine',
                periodic_transform_fn=self.dynamics.periodic_transform_fn 
            )
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
            self.get_logger().info("DeepReach model loaded successfully, modelpath = " + MODEL_PATH)
        
        self.start_controller()
        self.iteration = 0
        
        # Initialize state history for RK4 integration
        self.dt = 1.0/self.controller_rate # Control period (50Hz)
        #self.dt = 0.5 # Control period (50Hz)

        self.get_logger().info(f"Control period: {self.dt}")
        
        # Initialize JSON logging
        self.log_data = []
        self.start_time = time.time()  # Track start time for relative timestamps
        self.log_filename = f"/mounted_volume/drone_experiment_data/12drones_{GHOST_AGENT}ghost_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.get_logger().info(f"JSON logging enabled. Log file: {self.log_filename}")
        
        # Track first occurrence of events
        self.first_collision_warning_time = None
        self.first_out_of_bounds_time = None
        
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
        states = np.array(state).reshape(self.nbr_flying_robots, -1)
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
            drone_12d_state = np.zeros(12)
            
            for i, robot_state in enumerate(states):
                # Extract position and velocity from [x, y, z, vx, vy, vz] format
                pos = robot_state[0:3]  # x, y, z
                vel = robot_state[3:6]  # vx, vy, vz

                if GHOST_AGENT == "pursuer":  # Live Drone 1 (evader)
                    # [x1, v1_x, y1, v1_y, z1, v1_z]
                    drone_12d_state[0] = pos[0]
                    drone_12d_state[1] = vel[0]
                    drone_12d_state[2] = pos[1]
                    drone_12d_state[3] = vel[1]
                    drone_12d_state[4] = pos[2]
                    drone_12d_state[5] = vel[2]
                    # Create a simple ghost state for the pursuer
                    drone_12d_state[6:12] = self.ghost_state_pursuer
                elif GHOST_AGENT == "evader":  # Live Drone 2 (pursuer)
                    # [x2, v2_x, y2, v2_y, z2, v2_z]
                    drone_12d_state[6] = pos[0]
                    drone_12d_state[7] = vel[0]
                    drone_12d_state[8] = pos[1]
                    drone_12d_state[9] = vel[1]
                    drone_12d_state[10] = pos[2]
                    drone_12d_state[11] = vel[2]
                    # Create a simple ghost state for the evader
                    drone_12d_state[0:6] = self.ghost_state_evader
            
                elif GHOST_AGENT == "both":
                    drone_12d_state[0:6] = self.ghost_state_evader
                    drone_12d_state[6:12] = self.ghost_state_pursuer
                else:
                    raise ValueError(f"Unknown GHOST_AGENT: {GHOST_AGENT}")
            

                

            
            # Combine both drone states into a single 12D state
            combined_12d_state = np.concatenate(drone_12d_state)  # [12] = [6] + [6]
            
            # Convert to tensor for DeepReach
            drone_12d_state_tensor = torch.tensor(combined_12d_state, dtype=torch.float32, device=device)
            self.get_logger().info(f"Evader state: {drone_12d_state[0:6]}")
            self.get_logger().info(f"Pursuer state: {drone_12d_state[6:12]}")
            
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

            value = self.dynamics.io_to_value(
                model_in,
                model_out.squeeze(dim=-1),
            ).detach()

            ellx = self.dynamics.boundary_fn(drone_12d_state_tensor).detach()
            
            # Use gradient to compute optimal control and disturbance
            optimal_u = self.dynamics.optimal_control(drone_12d_state_tensor, dv[..., 1:])
            optimal_d = self.dynamics.optimal_disturbance(drone_12d_state_tensor, dv[..., 1:])
            
            # Extract acceleration inputs from DeepReach
            acceleration1 = np.array([ # evader control
                self.dynamics.sideways_multiplier * optimal_u[0, 0].item(),  # ax
                self.dynamics.sideways_multiplier * optimal_u[0, 1].item(),  # ay
                self.dynamics.input_multiplier * optimal_u[0, 2].item()  # az
            ])
            
            acceleration2 = np.array([ # pursuer control
                self.dynamics.sideways_multiplier * optimal_d[0, 0].item(),  # ax
                self.dynamics.sideways_multiplier * optimal_d[0, 1].item(),  # ay
                self.dynamics.input_multiplier * optimal_d[0, 2].item()   # az
            ])
         
            # Apply acceleration limits for safety
            max_acc = 2.0  # m/s^2
            acceleration1[2] = np.clip(acceleration1[2], -max_acc, max_acc)
            acceleration2[2] = np.clip(acceleration2[2], -max_acc, max_acc)

            acceleration1[0] = np.clip(acceleration1[0], -max_acc, max_acc)
            acceleration1[1] = np.clip(acceleration1[1], -max_acc, max_acc)
            acceleration2[0] = np.clip(acceleration2[0], -max_acc, max_acc)
            acceleration2[1] = np.clip(acceleration2[1], -max_acc, max_acc)
            
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
            max_z = 2.2
            # Clip integrated values - only limit Z position
            #integrated_pos1[2] = np.clip(integrated_pos1[2], 0.0, max_z)  # Only clip Z to [0, 2.5]
            integrated_vel1[2] = np.clip(integrated_vel1[2], -max_vel, max_vel)
            integrated_pos2[2] = np.clip(integrated_pos2[2], 0.0, max_z)  # Only clip Z to [0, 2.5]
            integrated_vel2[2] = np.clip(integrated_vel2[2], -max_vel, max_vel)
            
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
            # self.get_logger().info(f"Evader - Integrated position: {integrated_pos1}")
            # self.get_logger().info(f"Evader - Integrated velocity: {integrated_vel1}")
            # self.get_logger().info(f"Pursuer - DeepReach acceleration: {acceleration2}")
            # self.get_logger().info(f"Pursuer - Integrated position: {integrated_pos2}")
            # self.get_logger().info(f"Pursuer - Integrated velocity: {integrated_vel2}")
                
        else:
            raise NotImplementedError(f"Mode {MODE} not implemented")
            
        self.iteration += 1
        
        # Log the actual positions of the robots
        # Extract actual positions and velocities from [x, y, z, vx, vy, vz] format
        actual_pos1 = states[0, 0:3]  # Current actual position [x, y, z]
        actual_vel1 = states[0, 3:6]  # Current actual velocity [vx, vy, vz]
        actual_pos2 = states[1, 0:3]  # Current actual position [x, y, z]
        actual_vel2 = states[1, 3:6]  # Current actual velocity [vx, vy, vz]

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

        if self.iteration % 50 == 0:  # Log every 50 iterations (about once per second)
            self.get_logger().info(f"FullState Mode - Iteration {self.iteration}: Current positions for {self.nbr_robots} robots")

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
                "integrated_position": integrated_pos1.tolist(),
                "integrated_velocity": integrated_vel1.tolist(),
                "acceleration": acceleration1.tolist()
            },
            "pursuer": {
                "actual_position": actual_pos2.tolist(),
                "actual_velocity": actual_vel2.tolist(),
                "integrated_position": integrated_pos2.tolist(),
                "integrated_velocity": integrated_vel2.tolist(),
                "acceleration": acceleration2.tolist()
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
    controller = FullStateController()
    
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