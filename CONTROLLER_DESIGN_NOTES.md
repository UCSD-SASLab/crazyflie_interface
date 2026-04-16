# Iterative Herd And Delivery Controllers

This document summarizes the controller work added to `crazyflie_interface` to support online replanning for the herd and delivery tasks originally implemented in `realmcf/inp_herd.py` and `realmcf/inp_delivery.py`.

The goal of this work was not to port the old ROS topic structure directly. The goal was to keep planning inside `crazyflie_interface`, use the same `rraa-rl` models and rollout logic, and emit commands in the native `crazyflie_interface` control format.

## High-Level Outcome

Two new controller entry points were added:

- [scripts/herd_controller.py](/mounted_volume/ros2_ws/src/crazyflie_interface/scripts/herd_controller.py:1)
- [scripts/delivery_controller.py](/mounted_volume/ros2_ws/src/crazyflie_interface/scripts/delivery_controller.py:1)

Both are thin wrappers around a shared implementation module:

- [crazyflie_interface_py/iterative_planning_controller.py](/mounted_volume/ros2_ws/src/crazyflie_interface/crazyflie_interface_py/iterative_planning_controller.py:1)

An optional dog tracker node was also added:

- [scripts/dog_tracker.py](/mounted_volume/ros2_ws/src/crazyflie_interface/scripts/dog_tracker.py:1)

The install list in [CMakeLists.txt](/mounted_volume/ros2_ws/src/crazyflie_interface/CMakeLists.txt:23) was updated so these scripts are installed by the package.

## Core Design Decision

The old `realmcf` pipeline was:

1. Load checkpoint.
2. Generate a rollout offline in an input node.
3. Publish a batch trajectory on `traj_input` or `unitree_traj_input`.
4. Let downstream tracker nodes convert those trajectories into robot commands.

The new `crazyflie_interface` pipeline is:

1. Read current robot state from `cf_interface/state`.
2. Construct a planning state for the `rraa-rl` environment from those live observations.
3. Run a short rollout every planning cycle.
4. Smooth the short horizon.
5. Emit the next command directly as `cf_interface/control_full_state`.

This means:

- planning is now online and iterative
- control remains inside `crazyflie_interface`
- we no longer depend on the old `CFTraj` / `CFTrajArray` trajectory message path
- live feedback affects the high-level rollout, not just the low-level tracking

## What Was Preserved From The Original `realmcf` Scripts

The new controllers intentionally preserve the important environment-to-hardware adaptation logic from the original scripts.

### Shared preserved ideas

- Load the same `rraa-rl` checkpoint using `load_ckpt(...)`.
- Use a `Collector` and `agent.collect_eval_with_states(...)` for rollout generation.
- Work from environment state objects rather than reconstructing policy inputs manually.
- Produce a short horizon and smooth it before using it for hardware commands.

### Preserved scaling and timing logic

The original `realmcf` scripts converted simulation units into hardware units using:

- `sim_to_m = agent_radius_real / cfg.agent_radius`
- `vel_max_m_per_simtime = max(cfg.vel_maxs) * sim_to_m`
- `s_per_simtime = vel_max_m_per_simtime / vel_max_real`

That same logic is preserved in the new controllers.

For herd:

- `agent_radius_real_m = 0.11`
- `vel_max_real_mps = 1.1`
- `center_shift_m = [0.24, 0.01]`
- `smooth_coef = 1e-4`

For delivery:

- `agent_radius_real_m = 0.112`
- `vel_max_real_mps = 0.6`
- `center_shift_m = [0.24, 0.01] + [0.18, 0.10]`
- `smooth_coef = 2e-5`

These values were taken directly from the old `realmcf` input scripts.

### Preserved smoothing approach

The original scripts:

- densified the raw rollout in time
- used `make_smoothing_spline(...)`
- evaluated the smoothed path at a finer resolution

The new controllers preserve that same smoothing structure:

- `n_smooth_pts_per_step = 2`
- per-axis smoothing spline
- cubic spline evaluation of position, velocity, and acceleration

### Preserved herd dog yaw behavior

`inp_herd.py` had special dog yaw handling:

- for each dog position, compute the nearest herd agent
- set dog yaw to face that herd agent

That optional behavior is preserved in the new herd controller with:

- `dog_face_nearest_herd = True`

If enabled, dog yaw is computed against the nearest herd trajectory point over the smoothed horizon.

### Preserved delivery dog yaw behavior

`inp_delivery.py` used tangent yaw for dogs:

- compute yaw from the path tangent

That behavior is preserved in the delivery controller for dog paths.

## What Was Intentionally Changed

Several aspects of the old design were intentionally not preserved, because they were tied to the old ROS architecture rather than the planning logic itself.

### No `traj_input` / `unitree_traj_input` publishing in the main controller path

The new controllers do not publish `CFTrajArray` messages.

Reason:

- `crazyflie_interface` already has a native control loop and a native command output topic
- the request was to keep planning inside `crazyflie_interface`
- reintroducing the old trajectory message path would add an unnecessary second control stack

### No old `realmcf` tracker classes

The new code does not use:

- `CFTracker`
- `TrajManager`
- `UnitreeTracker`

Reason:

- those classes were part of the old `realmcf` message-passing architecture
- the request was to use robot positions directly as done in the ghost controllers
- the new controllers publish immediate full-state setpoints rather than maintaining a separate trajectory-tracking subsystem

### No one-shot future timestamp broadcast

The original input scripts:

- waited for user input
- stamped a trajectory to start 5 seconds in the future
- published the entire future rollout

The new controllers do not do that.

Reason:

- these controllers replan every cycle
- delayed one-shot broadcast is not compatible with iterative replanning

## New Controller Architecture

The shared implementation lives in `IterativePlanningControllerBase`.

Its cycle is:

1. Read live robot positions and velocities from `cf_interface/state`.
2. Optionally replace those with internal ghost positions if `simulate_controlled_positions = True`.
3. Overwrite the relevant parts of a valid `rraa-rl` template state.
4. Run a short rollout from that observed state.
5. Extract the planned future XY positions.
6. Convert sim units to meters using the original scaling logic.
7. Smooth the short horizon.
8. Query the short horizon at a near-future evaluation time.
9. Publish a 16D full-state command per flying robot.

The output format is the same full-state format used by `cf_interface/control_full_state`:

- position `(x, y, z)`
- velocity `(vx, vy, vz)`
- quaternion `(qx, qy, qz, qw)`
- angular velocity `(wx, wy, wz)`
- acceleration `(ax, ay, az)`

## Why Template State Overwrite Was Used

The `rraa-rl` environments have nontrivial state structure:

- temporal node index
- base state
- herder state
- herd state
- delivery target centers

Instead of trying to synthesize a new valid state object from scratch every cycle, the controller:

1. gets a valid template state from `env.get_eval_states(1, root_only=True)`
2. stores a mutable shadow copy of the relevant arrays
3. overwrites only the fields driven by hardware observations
4. rebuilds a JAX state object via `jdc.replace(...)`

Reason:

- this is much less brittle than creating environment state trees manually
- it preserves hidden or task-specific fields that are not directly observable from hardware
- it matches the environment’s expected internal structure

## How Live Observation Is Injected

### Herd

The herd controller maintains shadow copies of:

- `shadow_herd_state`
- `shadow_herder_state`
- `shadow_temporal_node_idx`

Live observed robot states are mapped into selected slots:

- `"herd"` slots if herd agents are flown directly
- `"herder"` slots for normal flying herders
- `"dog"` slots for dog herders, when dogs are being treated as flying agents

Observed XY positions are converted from meters to sim coordinates using:

- inverse center shift
- division by `sim_to_m`

Observed XY velocities are converted back into sim-time velocity units using:

- `vel_sim = vel_mps * s_per_simtime / sim_to_m`

### Delivery

The delivery controller maintains:

- `shadow_herd_state`
- `shadow_herder_state`
- `shadow_centers`
- `shadow_temporal_node_idx`

Live robot observations overwrite only the controlled herder slots. The dynamic target centers are not observed from ROS here; they are propagated from the planning rollout state.

## Ghost / Fully Simulated Option

The request explicitly asked to preserve a ghost-style option similar to `20d_drone_controller_ghost.py`.

This is implemented with:

- `simulate_controlled_positions = False` by default

If set to `True`:

- live robot states are ignored for the controlled slots
- the controller advances internal ghost positions and velocities from the planned command output
- replanning still happens each cycle, but it is seeded by the internal ghost state rather than real CF state

Reason:

- this supports pure iterative simulation inside the same controller path
- it matches the earlier “ghost” usage pattern without requiring a separate planner implementation

## Flying Robots Versus Dog Agents

This was a major design choice.

### Default behavior

Default behavior is:

- `dog_as_flying_agent = True`

This means dog-designated slots are simply treated as additional flying agents and included in the normal Crazyflie full-state output.

Reason:

- this matches the request that the default should be to use another Crazyflie agent
- it keeps the main planner/controller path simple

### Optional separate dog output

Optional behavior is:

- `dog_as_flying_agent = False`
- `publish_dog_plan = True`

In that mode:

- dog slots are not added to the CF output list
- the planner still computes short-horizon dog paths
- those dog plans are published on separate topics:
  - `dog_plan_00`, `dog_plan_01`, ...
  - `dog_plan_path_00`, `dog_plan_path_01`, ...

The format of `dog_plan_*` is currently:

- `Float32MultiArray`
- first entry: `delta_t`
- second entry: `n_steps`
- remaining entries: flattened `[x, y, yaw]` rows

Reason:

- keeps planning in `crazyflie_interface`
- allows a separate node to handle dog actuation, analogous to the old Unitree tracker setup
- avoids forcing Unitree or dog actuation logic into `cf_interface`

## Separate Dog Tracker Node

The new optional dog tracker is:

- [scripts/dog_tracker.py](/mounted_volume/ros2_ws/src/crazyflie_interface/scripts/dog_tracker.py:1)

Its job is:

1. Subscribe to `dog_plan_00`.
2. Subscribe to odometry on `/dog/odom`.
3. Build splines over the incoming dog plan.
4. Query the reference path at the current elapsed time.
5. Compute body-frame velocity and yaw-rate commands.
6. Publish `TwistStamped` on `/cmd_vel`.

This is intentionally similar in spirit to `realmcf/unitree_tracker.py`, but simplified for the current package.

Important limitation:

- `dog_tracker.py` is currently written for a single dog plan topic by default: `dog_plan_00`
- it is a minimal bridge, not yet a full multi-dog generalized tracker

## RViz Visualization Added

The controllers now publish RViz-friendly outputs directly.

### Flying-agent visualization

For each flying-agent output slot:

- `controller_plan_path_00`, `controller_plan_path_01`, ... as `nav_msgs/Path`
- `controller_plan_markers` as `visualization_msgs/MarkerArray`

The marker array includes:

- current commanded setpoint marker for each flying robot
- ghost position markers when `simulate_controlled_positions = True`

### Dog visualization

If dog plan publishing is enabled:

- `dog_plan_path_00`, `dog_plan_path_01`, ... are published from the main planner

The dog tracker also publishes:

- `dog_tracker_path`

Reason:

- the old `realmcf` scripts had RViz visualization of the planned environment and trajectories
- the new controller path needed an equivalent way to inspect the short-horizon plan online

## Naming Decisions

The intermediate implementation used names containing `realmcf` and `adaptive`.

That was removed from the final code path.

Current names:

- `HerdController`
- `DeliveryController`
- `IterativePlanningControllerBase`

Reason:

- the code now lives in `crazyflie_interface`
- the important distinction is iterative replanning, not the source repo name
- “adaptive” was considered too vague and redundant

The only remaining `realmcf` text in the code is inside checkpoint filesystem paths, because those model files live under a `realmcf` directory on disk.

## Important Implementation Choices And Tradeoffs

### Why full-state output instead of low-level RPYT

The controllers publish `cf_interface/control_full_state` rather than `cf_interface/control`.

Reason:

- the planner naturally produces geometric path information
- full-state setpoints are a better match for smoothed short-horizon rollout tracking
- this is closer in spirit to the old `realmcf` CF path-tracking approach

### Why immediate short-horizon query instead of whole-trajectory tracking

The controller queries a short near-future point from the smoothed horizon and publishes that as the next full-state command.

Reason:

- `crazyflie_interface` already runs a control loop
- we only need the next command, not a second internal trajectory manager
- replanning every cycle makes a separate long-horizon tracker less necessary

### Why current yaw is not deeply tracked

Yaw is currently derived from:

- path tangent, or
- nearest-herd-facing direction for herd dogs

Reason:

- the original `realmcf` inputs were primarily XY task planners
- there is no richer 3D attitude plan from the rollout
- this keeps yaw behavior explicit and simple

### Why target centers are not observed live in delivery

The new delivery controller does not currently subscribe to external target observations. It propagates target centers from the rollout/environment state.

Reason:

- that was enough to preserve the existing planning logic
- no target observation path was specified in `crazyflie_interface`

This is a known extension point if live target sensing is needed later.

## Known Limitations

- Runtime verification was not completed in this environment because `jax` is not installed here.
- Workspace `colcon build` is still blocked by a pre-existing ROS interface generation issue involving stale `cf_traj` artifacts in the build tree.
- The controller knobs such as `cf_herder_idx`, `dog_herder_idx`, `dog_as_flying_agent`, checkpoint paths, and ghost behavior are still class attributes, not ROS parameters.
- `dog_tracker.py` is currently a minimal single-dog bridge.
- Delivery target observations are not yet pulled from live ROS topics.

## Recommended Next Steps

If further work is desired, the highest-value next steps are:

1. Convert controller configuration into ROS parameters.
2. Generalize `dog_tracker.py` to multiple dogs.
3. Add a dedicated observed-target input path for the delivery task.
4. Clean the stale interface generation issue so `colcon build` is healthy again.
5. Add launch files that make the flying-only and flying-plus-dog modes explicit.

## Summary

The final controller design is:

- iterative online replanning inside `crazyflie_interface`
- direct reuse of `rraa-rl` rollout generation
- preserved environment scaling and smoothing from the original `realmcf` scripts
- direct full-state command output for Crazyflies
- optional external dog-plan publication
- optional separate dog tracker node
- preserved herd dog yaw behavior to face the nearest herd agent
- RViz visibility for both flying plans and optional dog plans

This gives a cleaner architecture for the current stack:

- planner stays in `crazyflie_interface`
- flying robots are commanded directly
- dog actuation can be kept separate when needed
