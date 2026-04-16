# RealmCF Migration Handoff

## Goal

This branch is migrating the subset of `realmcf` that is actually needed for sampled trajectory tracking into `crazyflie_interface`, so that `crazyflie_interface` becomes self-contained for this use case.

The intended architecture is:

- `crazyflie_interface` remains the only package that owns:
  - Crazyflie state normalization
  - low-level and full-state command forwarding
  - takeoff / land service interface
  - trajectory message definitions
  - trajectory interpolation and tracking utilities
  - tracker controllers and simple planner input nodes
- `realmcf` becomes reference material, not a required runtime dependency for the migrated control path

The migration deliberately does **not** try to port all of `realmcf`. The focus is only on the planner/tracker path that used:

- `realmcf_interfaces/CFTraj`
- `realmcf_interfaces/CFTrajArray`
- `realmcf/traj_manager.py`
- `realmcf/cf_tracker.py`
- simple input publishers such as `inp_lemniscate.py`

It explicitly avoids porting the `realmcf` transport/runtime layer built around `CrazyflieServer`, because `crazyflie_interface` already owns that role.


## Branch / Workspace Context

- Workspace root: `/mounted_volume/ros2_ws`
- Main package being modified: `/mounted_volume/ros2_ws/src/crazyflie_interface`
- Current branch in `crazyflie_interface` at time of writing: `WAS_rss26_realm_utils`

There is also a `realmcf` repo present at:

- `/mounted_volume/ros2_ws/src/realmcf`

That repo was used as the source/reference for the migration logic, but the new control path is being copied into `crazyflie_interface` so the latter can stand on its own.


## Why This Migration Was Needed

`realmcf` was not a clean drop-in dependency for this workspace.

Important findings:

- `realmcf` is a ROS 2 Python package, but it depends on a sibling package `realmcf_interfaces` that was not present in the workspace.
- `realmcf` also contained its own Crazyflie runtime/control stack based on `crazyflie_py.CrazyflieServer` and direct `cmdFullState()` calls.
- That overlaps with functionality already present in `crazyflie_interface`.
- The real piece that was missing from `crazyflie_interface` was not hardware control, but the planner-side sampled trajectory message and trajectory splicing/tracking logic.

So the correct migration target was:

- **Do migrate** the planner-side trajectory utilities and controllers.
- **Do not migrate** the old `realmcf` direct transport/runtime layer.


## What Was Implemented

### 1. New Self-Contained Trajectory Messages

Added to `crazyflie_interface`:

- [msg/CFTraj.msg](/mounted_volume/ros2_ws/src/crazyflie_interface/msg/CFTraj.msg:1)
- [msg/CFTrajArray.msg](/mounted_volume/ros2_ws/src/crazyflie_interface/msg/CFTrajArray.msg:1)

These are the local replacements for the old `realmcf_interfaces` messages.

Functional schema:

`CFTraj`

- `builtin_interfaces/Time stamp`
- `float32[] pos_traj`
- `float32[] yaw_traj`
- `int32 n_steps`
- `float32 delta_t`

`CFTrajArray`

- `builtin_interfaces/Time stamp`
- `int32[] cf_ids`
- `CFTraj[] trajs`

These match the actual data model that the `realmcf` input/tracker scripts were using.


### 2. New Reusable Trajectory Utility Layer

Added under:

- [crazyflie_interface_py/traj_manager.py](/mounted_volume/ros2_ws/src/crazyflie_interface/crazyflie_interface_py/traj_manager.py:1)
- [crazyflie_interface_py/tracker_utils.py](/mounted_volume/ros2_ws/src/crazyflie_interface/crazyflie_interface_py/tracker_utils.py:1)

This is the migrated core of the `realmcf` planner/tracker path.

`traj_manager.py` contains:

- `Traj`
- `TrajManagerCfg`
- `KinState`
- `TrajManager`

This handles:

- storing sampled position/yaw trajectories
- splicing in new trajectories with a lookahead
- smoothing continuity errors at splice boundaries
- querying interpolated position / velocity / acceleration / yaw / yaw rate

`tracker_utils.py` contains:

- `state_to_matrix()`
- `message_to_traj()`
- `solve_assignment()`
- `build_full_state_command()`
- `hold_current_positions()`

These utilities were written so that the controller logic can be tested without a live ROS graph.


### 3. New Tracker Controller Inside `crazyflie_interface`

Added:

- [scripts/realmcf_tracker_controller.py](/mounted_volume/ros2_ws/src/crazyflie_interface/scripts/realmcf_tracker_controller.py:1)

This is the main migrated control path.

Design:

- subclasses `TemplateController`
- subscribes to `cf_interface/state`
- subscribes to `traj_input`
- internally stores one `TrajManager` per robot
- performs initial trajectory assignment using XY nearest-cost matching
- publishes full-state setpoints to `cf_interface/control_full_state`

Important details:

- It is designed to work with the existing `crazyflie_interface` control path rather than calling `crazyflie_py` directly.
- It publishes 16 values per robot, matching the contract consumed by `cf_interface.py`.
- It currently uses position / velocity / yaw-rate-derived omega / acceleration as the effective full-state command fields.
- Quaternion is left as identity in practice, consistent with how the existing `cf_interface.py` callback currently handles full-state input.

Important robustness improvement made during implementation:

- The controller originally assumed robot count would always be supplied through ROS parameters.
- That assumption is not reliable under the current launch flow.
- The controller now lazily infers robot count from either:
  - the first incoming `cf_interface/state`, or
  - the first incoming `traj_input`
- This was necessary to make it usable under the existing package composition.


### 4. New Minimal Planner Input Node

Added:

- [scripts/input_lemniscate.py](/mounted_volume/ros2_ws/src/crazyflie_interface/scripts/input_lemniscate.py:1)

This is a migrated and simplified version of the old `realmcf` lemniscate input publisher.

It:

- publishes `CFTrajArray` on `traj_input`
- supports parameters for:
  - `cf_ids`
  - `delta_t`
  - `n_steps`
  - `speed`
  - `size`
  - `height`

This is meant to provide the first clean vertical slice for sim/debug before migrating the more complex `delivery` and `herd` planners.


### 5. New Launch Entry Point

Added:

- [launch/realmcf_tracker_launch.py](/mounted_volume/ros2_ws/src/crazyflie_interface/launch/realmcf_tracker_launch.py:1)

This launch file is intended as a first integration entry point for:

- existing `crazyflie_interface` bringup
- the new `realmcf_tracker_controller.py`
- optionally `input_lemniscate.py`

Arguments currently included:

- `backend`
- `uri`
- `run_input`

This launch file is useful for debugging, but it still needs live sim validation.


### 6. Build / Package Updates

Updated:

- [CMakeLists.txt](/mounted_volume/ros2_ws/src/crazyflie_interface/CMakeLists.txt:1)
- [package.xml](/mounted_volume/ros2_ws/src/crazyflie_interface/package.xml:1)
- [crazyflie_interface_py/__init__.py](/mounted_volume/ros2_ws/src/crazyflie_interface/crazyflie_interface_py/__init__.py:1)

Changes include:

- registering the new message files
- installing the new scripts
- exposing the new Python helper modules
- adding pytest-based test targets
- adding `builtin_interfaces`, `nav_msgs`, and `python3-scipy` package dependencies

Note:

- The initial implementation used `attrs`, but that module was not available in the environment.
- The migrated trajectory core was converted to stdlib `dataclasses` so the code is actually self-contained in the current environment.


## Tests Added

Added:

- [test/test_traj_manager.py](/mounted_volume/ros2_ws/src/crazyflie_interface/test/test_traj_manager.py:1)
- [test/test_tracker_utils.py](/mounted_volume/ros2_ws/src/crazyflie_interface/test/test_tracker_utils.py:1)

Coverage intent:

- `test_traj_manager.py`
  - verifies shape and query behavior for interpolated trajectories
  - verifies that a second trajectory can be spliced in and still produce valid future state

- `test_tracker_utils.py`
  - verifies flat message parsing into trajectory arrays
  - verifies state reshaping
  - verifies assignment logic
  - verifies output command shaping for full-state control

These tests are intentionally focused on the migrated logic and do not require a live ROS graph or a live Crazyflie backend.


## Verification Performed

### Focused Python Tests

Executed successfully with plugin autoload disabled:

```bash
source /mounted_volume/ros2_ws/install/setup.bash
export PYTHONPATH=/mounted_volume/ros2_ws/src/crazyflie_interface:$PYTHONPATH
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
pytest /mounted_volume/ros2_ws/src/crazyflie_interface/test/test_traj_manager.py /mounted_volume/ros2_ws/src/crazyflie_interface/test/test_tracker_utils.py -q
```

Observed result:

- `6 passed in 0.17s`


### Syntax Verification

Ran `py_compile` successfully on:

- `crazyflie_interface_py/traj_manager.py`
- `crazyflie_interface_py/tracker_utils.py`
- `scripts/realmcf_tracker_controller.py`
- `scripts/input_lemniscate.py`


### Package Build Status

A `colcon build --packages-select crazyflie_interface` was run.

Observed behavior:

- ROS interface generation and install logs for `crazyflie_interface` completed successfully.
- Generated Python artifacts for the new messages were observed in the install tree.
- However, the sandboxed `colcon` process did not return cleanly to the tool wrapper afterward.

Interpretation:

- There is good evidence that the package built far enough to generate and install the new interfaces.
- But there was **not** a cleanly observed full command completion through the tooling layer, so build verification is not as strong as the focused unit tests.

Treat the package as:

- likely buildable,
- but still requiring one clean manual `colcon build` confirmation in a fresh shell.


## Known Gaps / Important Caveats

### 1. No End-to-End Sim Validation Yet

The biggest remaining gap is that the new controller and launch path have **not yet been validated end-to-end in simulation**.

What still needs to be checked:

- `realmcf_tracker_controller.py` receives `cf_interface/state`
- `input_lemniscate.py` publishes `traj_input`
- tracker builds setpoints and publishes `cf_interface/control_full_state`
- `cf_interface.py` forwards those to the backend without unexpected dimension/parameter issues
- drones in sim actually move in the intended trajectory


### 2. Full-State Path Uses Existing `cf_interface.py` Semantics

The current `cf_interface.py` full-state callback already has specific behavior:

- it consumes 16 values per robot
- it clips z
- it currently overwrites quaternion to identity before publishing to `crazyflie_interfaces/FullState`

This means:

- the migrated tracker is aligned with the current behavior,
- but if orientation-aware full-state tracking becomes important later, this path may need a deeper redesign.


### 3. `cf_ids` in `CFTrajArray` Are Still Mostly Informational

This matches legacy `realmcf` behavior.

Current controller logic:

- uses assignment from current XY robot positions to first trajectory waypoint
- does **not** use `cf_ids` as the authoritative mapping

If deterministic explicit robot-to-traj mapping is preferred later, the controller can be changed to honor `cf_ids`.


### 4. Only the Lemniscate Planner Was Migrated

The first slice intentionally only migrated:

- trajectory messages
- trajectory utilities
- tracker controller
- `lemniscate` input node

The complex `delivery` and `herd` planners were **not** yet migrated into `crazyflie_interface`.

That is still future work.


### 5. `realmcf` Still Exists in Workspace

The old `realmcf` repo is still present and can still be referenced.

That is useful for source comparison, but it also means future sessions must be careful not to accidentally reintroduce runtime dependence on it.

The intent is:

- use `realmcf` only as reference code
- keep new runtime logic in `crazyflie_interface`


## Recommended Next Steps

### Immediate Next Step

Do a live sim validation of the new vertical slice.

Suggested order:

1. Clean build in a fresh shell:

```bash
cd /mounted_volume/ros2_ws
colcon build --packages-select crazyflie_interface
source install/setup.bash
```

2. Launch the tracker stack in sim:

```bash
ros2 launch crazyflie_interface realmcf_tracker_launch.py backend:=sim
```

3. Observe:

- whether `realmcf_tracker_controller` comes up cleanly
- whether `input_lemniscate` publishes
- whether `cf_interface/state` starts arriving
- whether `cf_interface/control_full_state` is being published
- whether sim drones move as expected

4. If motion does not happen, inspect:

- robot count inference in the tracker
- trajectory callback warnings
- whether takeoff happened
- whether the full-state callback in `cf_interface.py` is active and receiving commands


### Likely Follow-Up Changes After Sim Validation

If the vertical slice works:

- migrate `inp_delivery.py` into `crazyflie_interface/scripts/`
- migrate `inp_herd.py` into `crazyflie_interface/scripts/`
- copy only the minimal visualization helpers needed for those planners
- decide whether `cf_ids` should become authoritative rather than assignment-based

If the vertical slice does **not** work:

Start debugging in this order:

1. message flow
2. robot count inference
3. trajectory manager callback timing
4. full-state command dimensions
5. takeoff / flight gating in `cf_interface.py`


## Suggested Debug Commands

After sourcing the workspace:

```bash
ros2 topic list | sort
```

```bash
ros2 topic echo /cf_interface/state
```

```bash
ros2 topic echo /traj_input
```

```bash
ros2 topic echo /cf_interface/control_full_state
```

```bash
ros2 service call /cf_interface/command crazyflie_interface/srv/Command "{command: 'takeoff'}"
```

If needed:

```bash
ros2 node list
```

```bash
ros2 topic hz /cf_interface/state
```

```bash
ros2 topic hz /cf_interface/control_full_state
```


## Files Added / Changed In This Migration Slice

### New Files

- [msg/CFTraj.msg](/mounted_volume/ros2_ws/src/crazyflie_interface/msg/CFTraj.msg:1)
- [msg/CFTrajArray.msg](/mounted_volume/ros2_ws/src/crazyflie_interface/msg/CFTrajArray.msg:1)
- [crazyflie_interface_py/traj_manager.py](/mounted_volume/ros2_ws/src/crazyflie_interface/crazyflie_interface_py/traj_manager.py:1)
- [crazyflie_interface_py/tracker_utils.py](/mounted_volume/ros2_ws/src/crazyflie_interface/crazyflie_interface_py/tracker_utils.py:1)
- [scripts/realmcf_tracker_controller.py](/mounted_volume/ros2_ws/src/crazyflie_interface/scripts/realmcf_tracker_controller.py:1)
- [scripts/input_lemniscate.py](/mounted_volume/ros2_ws/src/crazyflie_interface/scripts/input_lemniscate.py:1)
- [launch/realmcf_tracker_launch.py](/mounted_volume/ros2_ws/src/crazyflie_interface/launch/realmcf_tracker_launch.py:1)
- [test/test_traj_manager.py](/mounted_volume/ros2_ws/src/crazyflie_interface/test/test_traj_manager.py:1)
- [test/test_tracker_utils.py](/mounted_volume/ros2_ws/src/crazyflie_interface/test/test_tracker_utils.py:1)

### Modified Files

- [CMakeLists.txt](/mounted_volume/ros2_ws/src/crazyflie_interface/CMakeLists.txt:1)
- [package.xml](/mounted_volume/ros2_ws/src/crazyflie_interface/package.xml:1)
- [crazyflie_interface_py/__init__.py](/mounted_volume/ros2_ws/src/crazyflie_interface/crazyflie_interface_py/__init__.py:1)


## Recommended Restart Prompt For Tomorrow

Use something close to this:

```text
Continue work on branch `WAS_rss26_realm_utils` in `/mounted_volume/ros2_ws/src/crazyflie_interface`.

First read:
- /mounted_volume/ros2_ws/src/crazyflie_interface/MIGRATION_NOTES.md
- git status
- git diff

Context:
- We are migrating the sampled-trajectory planner/tracker utilities from realmcf into crazyflie_interface so crazyflie_interface is self-contained.
- The first slice is implemented: local CFTraj/CFTrajArray messages, traj_manager/tracker_utils, realmcf_tracker_controller.py, input_lemniscate.py, launch file, and focused tests.
- The main next step is live sim validation of the new tracker path and then migrating delivery/herd planners if the vertical slice works.
```


## Bottom Line

This branch now contains the first usable self-contained migration slice.

What is solid:

- message definitions
- trajectory interpolation utilities
- assignment and full-state command shaping
- controller structure inside `crazyflie_interface`
- focused tests

What is still missing:

- live sim proof that the new tracker path behaves correctly under the existing `crazyflie_interface` runtime
- migration of the more complex `delivery` and `herd` input publishers

If resuming later, the correct mindset is:

- do **not** go back to using `realmcf` as a runtime dependency
- continue proving and extending the new `crazyflie_interface`-owned path
