# 2026-04-22 Crazyflie Interface Migration Status

## Summary

The `herd_controller` and `delivery_controller` migration from `rraa-rl` / `realmcf` into `ros2_ws/src/crazyflie_interface` is partially complete.

Current state:

- ROS 2 packaging/executable issues are resolved.
- Models now live inside the `crazyflie_interface` repo and are installed with the package.
- `herd_controller.py` launches and can load the migrated checkpoint.
- Sim launch wiring is working and RViz opens correctly.
- The pure `rraa-rl` reference rollout (`reference_env_default`) is available and is the current ground-truth debug artifact.
- The main remaining blocker is ROS/sim tracking behavior: simulated drones move toward staged targets but destabilize / descend instead of holding and proceeding cleanly.

## What Was Fixed

- Added install rules for `herd_controller.py` and `delivery_controller.py`.
- Cleaned stale `build/` and `install/` state so `ros2 run crazyflie_interface herd_controller.py` works.
- Moved model directories into:
  - `crazyflie_interface/models/20260128-134905_total_hardware`
  - `crazyflie_interface/models/20260129-154526_baker_reset-v4`
- Updated controller code to resolve models from package share instead of old `/home/realm/...` paths.
- Installed model directories into package share via `CMakeLists.txt`.
- Added sim config files for herd and delivery.
- Fixed launch forwarding so the selected `crazyflies_yaml` reaches the included Crazyswarm launch.
- Fixed several runtime/controller migration issues:
  - stale install-space paths
  - herd controller robot-count mapping mismatch
  - short-horizon spline crash
  - read-only shadow state arrays
  - RViz launch arg normalization
- Added extensive debug outputs:
  - reference rollout PNGs
  - reference GIFs
  - sim overlay plots
  - RViz path/marker topics

## Current Debugging Direction

We are currently treating `reference_env_default` as the expected behavior and using it to debug the ROS side, not the `rraa-rl` task semantics.

Current controller startup behavior:

- On first rollout, the controller computes `reference_env_default`.
- It extracts the first waypoint for each robot as a staging target.
- It holds those staging targets for `5` seconds before starting active control.
- It publishes debug arrays for:
  - observed XYZ: `/controller_debug_observed_xyz`
  - target XYZ: `/controller_debug_target_xyz`

## Current Problem

In sim, the drones appear to take off and move roughly toward the intended staging targets, but they do not stabilize there. Instead, they tend to descend / crash.

Most recent observations:

- staging targets are initialized correctly from `reference_env_default`
- staged XY error decreases initially, then stalls
- issue appears to be in ROS/sim tracking or full-state command interpretation, not in missing command publication

## Notes On Recent Changes

- The previous “press Enter to begin active control” behavior was removed because it interfered with terminal lifecycle / `Ctrl+C`.
- Startup now uses a fixed timed delay instead.
- Recent time-stamp related changes were rolled back from controller debug visualization messages and full-state command headers.
- `rraa-rl` experimental local transition-condition patches were discarded; `rraa-rl` was reset back to clean `os` branch state.

## Useful Commands

Launch sim:

```bash
senv
ros2 launch crazyflie_interface launch.py \
  backend:=sim \
  rviz:=true \
  crazyflies_yaml:=/mounted_volume/ros2_ws/install/crazyflie_interface/share/crazyflie_interface/config/herd_crazyflies_sim.yaml
```

Run herd controller:

```bash
senv
ros2 run crazyflie_interface herd_controller.py \
  --ros-args \
  --params-file /mounted_volume/ros2_ws/install/crazyflie_interface/share/crazyflie_interface/config/herd_controller_sim.yaml
```

Optional debug topics:

```bash
ros2 topic echo /controller_debug_observed_xyz
ros2 topic echo /controller_debug_target_xyz
```

## Immediate Next Steps

- Compare commanded XYZ vs observed XYZ during staging to determine whether Z is collapsing while the controller keeps commanding `z=0.6`.
- If needed, simplify staging commands further to isolate whether orientation / velocity / acceleration fields are destabilizing the sim backend.
- Once staging is stable, resume debugging `reference_observed` vs `reference_env_default`.
