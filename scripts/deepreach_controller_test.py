#!/usr/bin/env python3
"""DeepReach controller testing script for gym-pybullet-drones.

This script allows you to test the DeepReach controller with different trajectory types.
The controller can be tested in both hover mode and DeepReach pursuit-evasion mode.

Example
-------
In a terminal, run as:

    $ python deepreach_controller_test.py

"""
import os
import time
import argparse
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import torch
import sys
from deepreach.utils.modules import SingleBVPNet
from deepreach.dynamics import DronePursuitEvasion12D

# Add the scripts directory to the path for DeepReach imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

# Default parameters
DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 2  # DeepReach requires 2 drones
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 100
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

# Set device for PyTorch
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Model path - update this to your actual model path
MODEL_PATH = "/mounted_volume/gym-pybullet-drones/gym_pybullet_drones/examples/12d_drones.pth"

# numpy logging only 2 digits
np.set_printoptions(precision=2, suppress=True, floatmode='fixed')

MODE = ["hover", "deepreach"][1]  # Default to hover mode for testing


class DeepReachController:
    """DeepReach controller for pursuit-evasion game."""
    
    def __init__(self, model_path=MODEL_PATH):
        """Initialize DeepReach controller."""
        try:
            from deepreach.utils.modules import SingleBVPNet
            from deepreach.dynamics import DronePursuitEvasion12D
            
            self.dynamics = DronePursuitEvasion12D(collisionR=0.25, thrust_max=14.0, set_mode='avoid')
            
            self.model = SingleBVPNet(
                in_features=13,
                hidden_features=512,
                num_hidden_layers=3,
                out_features=1,
                type='sine',
                periodic_transform_fn=self.dynamics.periodic_transform_fn 
            )
            
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            self.model.load_state_dict(checkpoint["model"])
            self.model.to(device)
            self.model.eval()
            print("DeepReach model loaded successfully")
            
        except ImportError as e:
            print(f"Warning: DeepReach not available: {e}")
            print("Falling back to hover mode")
            self.model = None
            self.dynamics = None
        except FileNotFoundError:
            print(f"Warning: Model file not found at {model_path}")
            print("Falling back to hover mode")
            self.model = None
            self.dynamics = None
    
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
    
    def compute_control(self, states, dt):
        """Compute control for two drones using DeepReach.
        
        Returns:
            target_positions: Array of target positions for each drone
            target_velocities: Array of target velocities for each drone
        """
        if self.model is None or self.dynamics is None:
            # Fallback to hover
            print("Fallback to hover mode")
            target_positions = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.5]])
            target_velocities = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
            return target_positions, target_velocities
        
        # Process each robot with DeepReach
        if len(states) != 2:
            raise ValueError("DeepReach mode only supports 2 drones")
        
        # 12d_state = [x1, y1, z1, vx1, vy1, vz1, x2, y2, z2, vx2, vy2, vz2]
        drone_12d_state = []
        
        for i, robot_state in enumerate(states):
            # Extract position and velocity
            pos = robot_state[0:3]  # x, y, z
            vel = robot_state[3:6]  # vx, vy, vz
            
            # Construct 6D state: [x, v_x, y, v_y, z, v_z]
            drone_6d_state = np.array([
                pos[0],    # x
                vel[0],    # v_x
                pos[1],    # y
                vel[1],    # v_y
                pos[2],    # z
                vel[2]     # v_z
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
        
        # Get current 6D state for each drone
        current_drone1_state = combined_12d_state[:6]   # First 6 elements: [x1, vx1, y1, vy1, z1, vz1]
        current_drone2_state = combined_12d_state[6:12] # Last 6 elements: [x2, vx2, y2, vy2, z2, vz2]

       
        # Use RK4 integrator to get next state
        next_drone1_state = self.rk4_integrate(current_drone1_state, acceleration1, dt, steps=1)
        next_drone2_state = self.rk4_integrate(current_drone2_state, acceleration2, dt, steps=1)

    
        # Extract integrated position and velocity
        # next_drone_state format: [x, vx, y, vy, z, vz] (interleaved)
        integrated_pos1 = np.array([next_drone1_state[0], next_drone1_state[2], next_drone1_state[4]])  # [x, y, z]
        integrated_vel1 = np.array([next_drone1_state[1], next_drone1_state[3], next_drone1_state[5]])  # [vx, vy, vz]
        integrated_pos2 = np.array([next_drone2_state[0], next_drone2_state[2], next_drone2_state[4]])  # [x, y, z]
        integrated_vel2 = np.array([next_drone2_state[1], next_drone2_state[3], next_drone2_state[5]])  # [vx, vy, vz]
        
        # Apply safety limits
        max_vel = 2.0  # m/s
        max_pos = 5.0  # m
        
        # Clip integrated values
        integrated_pos1 = np.clip(integrated_pos1, -max_pos, max_pos)
        integrated_vel1 = np.clip(integrated_vel1, -max_vel, max_vel)
        integrated_pos2 = np.clip(integrated_pos2, -max_pos, max_pos)
        integrated_vel2 = np.clip(integrated_vel2, -max_vel, max_vel)
        
        # Return target positions and velocities for PID controller to follow
        target_positions = np.array([integrated_pos1, integrated_pos2])
        target_velocities = np.array([integrated_vel1, integrated_vel2])
        
        return target_positions, target_velocities

def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB,
        mode=MODE
        ):
    
    #### Initialize the simulation #############################
    # Initial positions - drones start at different heights
    H_STEP = 0.5
    INIT_XYZS = np.array([[0.25, 0, 0.5], [-0.25, 0, 0.5]])  # Drone 0 at (1,0,1), Drone 1 at (-1,0,1)
    INIT_RPYS = np.array([[0, 0, 0] for i in range(num_drones)])

    #### Create the environment ################################
    env = CtrlAviary(drone_model=drone,
                        num_drones=num_drones,
                        initial_xyzs=INIT_XYZS,
                        initial_rpys=INIT_RPYS,
                        physics=physics,
                        neighbourhood_radius=10,
                        pyb_freq=simulation_freq_hz,
                        ctrl_freq=control_freq_hz,
                        gui=gui,
                        record=record_video,
                        obstacles=obstacles,
                        user_debug_gui=user_debug_gui
                        )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )

            #### Initialize the controllers ############################
    if mode == "deepreach":
        # Use DeepReach controller
        deepreach_ctrl = DeepReachController()
        # Always initialize PID controllers for tracking
        if drone in [DroneModel.CF2X, DroneModel.CF2P]:
            ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]
    else:
        # Use PID controller
        print("Hover mode")
        if drone in [DroneModel.CF2X, DroneModel.CF2P]:
            ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]
        deepreach_ctrl = None


    #### Run the simulation ####################################
    action = np.zeros((num_drones,4))
    START = time.time()
    
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):
        current_time = i / env.CTRL_FREQ

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        #### Compute control for each drone ########################
        if mode == "deepreach":
            # Use DeepReach controller to get target positions and velocities
            states = np.array(obs).reshape(num_drones, -1)
            target_positions, target_velocities = deepreach_ctrl.compute_control(states, env.CTRL_TIMESTEP)
            
            # Use PID controller to track the DeepReach targets
            for j in range(num_drones):
                target_pos = target_positions[j]
                target_vel = target_velocities[j]

                current_pos = obs[j][0:3]
                current_vel = obs[j][3:6]
                
                # Check for collisions between drones
                collision_distance = 0.25  # Distance threshold for collision detection
                for k in range(j+1, num_drones):
                    other_pos = obs[k][0:3]
                    distance = np.linalg.norm(current_pos - other_pos)
                    if distance < collision_distance:
                        print(f"⚠️  COLLISION WARNING: Drones {j} and {k} are {distance:.3f}m apart (threshold: {collision_distance}m)")
                
                # Use PID controller to track the target position and velocity
                action[j, :], _, _ = ctrl[j].computeControlFromState(
                    control_timestep=env.CTRL_TIMESTEP,
                    state=obs[j],
                    target_pos=target_pos,
                    target_vel=target_vel,
                )

                if i % 100 == 0:
                    print(f"Drone {j} - Current: pos={current_pos}, vel={current_vel}")
                    print(f"Drone {j} - Target pos: {target_pos}, vel: {target_vel}")
                
              
                    
        else:
            # Hover mode - stay at initial positions
            for j in range(num_drones):
                target_pos = INIT_XYZS[j, :]
                target_vel = np.zeros(3)  # Zero velocity for hover
                
                # Compute control action
                action[j, :], _, _ = ctrl[j].computeControlFromState(
                    control_timestep=env.CTRL_TIMESTEP,
                    state=obs[j],
                    target_pos=target_pos,
                    target_vel=target_vel,
                    target_rpy=INIT_RPYS[j, :]
                )

                print(f"Control output: {action[j, :]}")

        #### Log the simulation ####################################
        for j in range(num_drones):
            if mode == "deepreach":
                target_pos = target_positions[j] if 'target_positions' in locals() else INIT_XYZS[j, :]
            else:
                target_pos = INIT_XYZS[j, :]
            
            logger.log(drone=j,
                       timestamp=current_time,
                       state=obs[j],
                       control=np.hstack([target_pos, INIT_RPYS[j, :], np.zeros(6)])
                       )

        #### Printout ##############################################
        #env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv(f"deepreach_controller_{mode}")

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()

if __name__ == "__main__":
    #### Define and parse arguments for the script ############
    parser = argparse.ArgumentParser(description='DeepReach controller testing script')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 2)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 15)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    parser.add_argument('--mode',               default=MODE, type=str,           help='Controller mode: hover, deepreach (default: hover)', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
