from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from crazyflie_interface.msg import CFTraj

from crazyflie_interface_py.traj_manager import Traj


STATE_DIM = 13
FULL_STATE_CONTROL_DIM = 16


def state_to_matrix(state: np.ndarray | list[float], n_robots: int) -> np.ndarray:
    state_np = np.asarray(state, dtype=float)
    expected = n_robots * STATE_DIM
    if state_np.size != expected:
        raise ValueError(f"Expected flattened state of length {expected}, got {state_np.size}")
    return state_np.reshape(n_robots, STATE_DIM)


def message_to_traj(msg_traj: CFTraj, first_step_time_s: float) -> Traj:
    T = int(msg_traj.n_steps)
    if T <= 0:
        raise ValueError("Trajectory must contain at least one sample")

    T_pos = np.asarray(msg_traj.pos_traj, dtype=float)
    if T_pos.size != 3 * T:
        raise ValueError(f"Expected {3*T} position values, got {T_pos.size}")
    T_pos = T_pos.reshape(T, 3)

    T_yaw = np.asarray(msg_traj.yaw_traj, dtype=float)
    if T_yaw.size != T:
        raise ValueError(f"Expected {T} yaw values, got {T_yaw.size}")

    msg_start_time_s = float(msg_traj.stamp.sec) + float(msg_traj.stamp.nanosec) * 1e-9 - first_step_time_s
    T_time = np.arange(T, dtype=float) * float(msg_traj.delta_t) + msg_start_time_s
    return Traj(T_time=T_time, T_pos=T_pos, T_yaw=T_yaw)


def solve_assignment(start_points: list[np.ndarray], end_points: list[np.ndarray]) -> list[int]:
    n = len(start_points)
    if len(end_points) != n:
        raise ValueError("start_points and end_points must have the same length")

    start_points_np = np.stack(start_points, axis=0)
    end_points_np = np.stack(end_points, axis=0)
    cost_matrix = np.linalg.norm(start_points_np[:, None, :] - end_points_np[None, :, :], axis=-1)
    _, col_ind = linear_sum_assignment(cost_matrix)
    return col_ind.tolist()


def build_full_state_command(states: np.ndarray, queried_pts: list, override_height: float | None = None) -> np.ndarray:
    n_robots = states.shape[0]
    u = np.zeros((n_robots, FULL_STATE_CONTROL_DIM), dtype=float)
    u[:, 9] = 1.0

    for i, pt in enumerate(queried_pts):
        u[i, 0:3] = pt.pos
        u[i, 3:6] = pt.vel
        u[i, 10:13] = np.array([0.0, 0.0, pt.omega], dtype=float)
        u[i, 13:16] = pt.acc

    if override_height is not None:
        u[:, 2] = override_height
        u[:, 5] = 0.0

    return u.flatten()


def hold_current_positions(states: np.ndarray, height: float | None = None) -> np.ndarray:
    n_robots = states.shape[0]
    u = np.zeros((n_robots, FULL_STATE_CONTROL_DIM), dtype=float)
    u[:, 0:3] = states[:, 0:3]
    u[:, 3:6] = 0.0
    u[:, 9] = 1.0
    u[:, 10:13] = 0.0
    u[:, 13:16] = 0.0

    if height is not None:
        u[:, 2] = height

    return u.flatten()
