# Controller Run Guide

This is a short operator-facing guide for the iterative herd and delivery controllers added to `crazyflie_interface`.

Detailed implementation notes are in:

- [CONTROLLER_DESIGN_NOTES.md](/mounted_volume/ros2_ws/src/crazyflie_interface/CONTROLLER_DESIGN_NOTES.md:1)

## Files

Main controller entry points:

- [scripts/herd_controller.py](/mounted_volume/ros2_ws/src/crazyflie_interface/scripts/herd_controller.py:1)
- [scripts/delivery_controller.py](/mounted_volume/ros2_ws/src/crazyflie_interface/scripts/delivery_controller.py:1)

Optional dog tracker:

- [scripts/dog_tracker.py](/mounted_volume/ros2_ws/src/crazyflie_interface/scripts/dog_tracker.py:1)

Shared implementation:

- [crazyflie_interface_py/iterative_planning_controller.py](/mounted_volume/ros2_ws/src/crazyflie_interface/crazyflie_interface_py/iterative_planning_controller.py:1)

## Preconditions

Before running any controller:

1. The ROS environment must be sourced.
2. `crazyflie_interface` must be built and sourced.
3. `rraa-rl` and its Python dependencies must be installed in the same Python environment.
4. `jax` must be available in that environment.
5. `cf_interface.py` must be running in full-state mode, because these controllers publish on `cf_interface/control_full_state`.

The important mode in [cf_interface.py](/mounted_volume/ros2_ws/src/crazyflie_interface/scripts/cf_interface.py:18) is:

```python
CONTROL_MODE = "full_state"
```

## Standard Launch Order

Typical order:

1. Start the Crazyflie backend and `cf_interface.py`.
2. Confirm `cf_interface/state` is being published.
3. Start either the herd or delivery controller.
4. Take off through the usual `cf_interface/command` service path.

The controllers only emit active commands once `cf_interface/flight_status` reports that the system is in flight.

## Running The Herd Controller

Main script:

- [herd_controller.py](/mounted_volume/ros2_ws/src/crazyflie_interface/scripts/herd_controller.py:1)

Typical invocation:

```bash
ros2 run crazyflie_interface herd_controller --ros-args --params-file <your_robot_yaml>
```

Important knobs are currently class attributes in [iterative_planning_controller.py](/mounted_volume/ros2_ws/src/crazyflie_interface/crazyflie_interface_py/iterative_planning_controller.py:400), not ROS params.

The main herd-specific ones are:

- `use_herd_agents`
- `cf_herder_idx`
- `dog_herder_idx`
- `dog_face_nearest_herd`
- `dog_as_flying_agent`
- `publish_dog_plan`
- `simulate_controlled_positions`

### Common herd modes

All flying:

- `use_herd_agents = True`
- `cf_herder_idx = []`
- `dog_herder_idx = []`
- `dog_as_flying_agent = True`

Mixed flying herders:

- `use_herd_agents = True` or `False`
- `cf_herder_idx = [...]`
- `dog_herder_idx = [...]`
- `dog_as_flying_agent = True`

Separate dog planner output:

- `dog_herder_idx = [...]`
- `dog_as_flying_agent = False`
- `publish_dog_plan = True`

Face nearest herd agent:

- `dog_face_nearest_herd = True`

This preserves the old herd dog yaw behavior from `realmcf`.

## Running The Delivery Controller

Main script:

- [delivery_controller.py](/mounted_volume/ros2_ws/src/crazyflie_interface/scripts/delivery_controller.py:1)

Typical invocation:

```bash
ros2 run crazyflie_interface delivery_controller --ros-args --params-file <your_robot_yaml>
```

Important delivery-specific knobs are in [iterative_planning_controller.py](/mounted_volume/ros2_ws/src/crazyflie_interface/crazyflie_interface_py/iterative_planning_controller.py:532):

- `cf_herder_idx`
- `dog_herder_idx`
- `dog_as_flying_agent`
- `publish_dog_plan`
- `simulate_controlled_positions`

### Common delivery modes

All flying:

- `cf_herder_idx = [0, 1]` or whatever slots correspond to the flying agents
- `dog_herder_idx = []`
- `dog_as_flying_agent = True`

Separate dog planner output:

- `dog_herder_idx = [...]`
- `dog_as_flying_agent = False`
- `publish_dog_plan = True`

For delivery, dog yaw follows the path tangent.

## Default Dog Behavior

The default behavior is:

- `dog_as_flying_agent = True`

This means dog slots are treated as normal flying agents and included in the normal Crazyflie full-state command stream.

This was chosen intentionally so the default behavior is to use another Crazyflie instead of requiring a separate dog actuator stack.

## Enabling Separate Dog Output

If you want the planner to emit dog plans for another node instead of treating those dogs as flying agents:

1. Set `dog_as_flying_agent = False`.
2. Set `publish_dog_plan = True`.
3. Set `dog_herder_idx` to the desired herder indices.
4. Run the dog tracker or your own dog actuator node.

The planner then publishes:

- `dog_plan_00`, `dog_plan_01`, ...
- `dog_plan_path_00`, `dog_plan_path_01`, ...

The current plan message format on `dog_plan_*` is:

- first value: `delta_t`
- second value: `n_steps`
- remaining values: flattened `[x, y, yaw]`

## Running The Dog Tracker

Main script:

- [dog_tracker.py](/mounted_volume/ros2_ws/src/crazyflie_interface/scripts/dog_tracker.py:1)

Typical invocation:

```bash
ros2 run crazyflie_interface dog_tracker
```

Current defaults inside the script:

- subscribes to `dog_plan_00`
- subscribes to `/dog/odom`
- publishes `TwistStamped` to `/cmd_vel`

If your dog odometry topic or command topic differs, update the class attributes in [dog_tracker.py](/mounted_volume/ros2_ws/src/crazyflie_interface/scripts/dog_tracker.py:18).

## Ghost / Fully Simulated Mode

To iteratively simulate the controlled positions rather than seed planning from live CF positions:

- set `simulate_controlled_positions = True`

Effect:

- the controller keeps replanning every cycle
- but the controlled slots are advanced from the controller’s internal shadow state
- this is the closest analogue to the ghost behavior in `20d_drone_controller_ghost.py`

## RViz Topics

The controllers publish:

- `controller_plan_path_00`, `controller_plan_path_01`, ...
- `controller_plan_markers`

If dog plan publishing is enabled, they also publish:

- `dog_plan_path_00`, `dog_plan_path_01`, ...

The dog tracker publishes:

- `dog_tracker_path`

## Practical Editing Points

Until these are parameterized, the most likely lines you will edit are in:

- [iterative_planning_controller.py](/mounted_volume/ros2_ws/src/crazyflie_interface/crazyflie_interface_py/iterative_planning_controller.py:333)

For herd:

- checkpoint path
- `use_herd_agents`
- `cf_herder_idx`
- `dog_herder_idx`
- `dog_face_nearest_herd`

For delivery:

- checkpoint path
- `cf_herder_idx`
- `dog_herder_idx`

For either controller:

- `simulate_controlled_positions`
- `dog_as_flying_agent`
- `publish_dog_plan`
- `flight_height_m`
- `rollout_horizon_steps`

## Example Operational Patterns

Flying-only herd:

```text
Start cf_interface -> run herd_controller -> take off
```

Herd with one dog emitted separately:

```text
Set dog_as_flying_agent = False
Set publish_dog_plan = True
Set dog_herder_idx = [...]
Start cf_interface
Start herd_controller
Start dog_tracker
Take off CFs / start dog stack
```

Delivery with all agents mocked as Crazyflies:

```text
Set dog_as_flying_agent = True
Assign dog slots into dog_herder_idx or leave empty
Run delivery_controller
```

Iterative simulation only:

```text
Set simulate_controlled_positions = True
Run herd_controller or delivery_controller
```

## Current Limitations

- These controller settings are still class attributes, not ROS parameters.
- `dog_tracker.py` is currently written around a single dog topic by default.
- A full `colcon build` was not validated end-to-end because the workspace has a separate stale-interface issue unrelated to these scripts.
- Runtime verification depends on having `jax` installed in the same Python environment.
