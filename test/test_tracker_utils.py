import types

import numpy as np

from crazyflie_interface_py.tracker_utils import (
    build_full_state_command,
    hold_current_positions,
    message_to_traj,
    solve_assignment,
    state_to_matrix,
)


def _mock_traj_msg(stamp_sec=10, stamp_nanosec=0, pos_traj=None, yaw_traj=None, n_steps=2, delta_t=0.1):
    if pos_traj is None:
        pos_traj = [0.0, 0.0, 0.5, 1.0, 0.0, 0.5]
    if yaw_traj is None:
        yaw_traj = [0.0, 0.2]
    return types.SimpleNamespace(
        stamp=types.SimpleNamespace(sec=stamp_sec, nanosec=stamp_nanosec),
        pos_traj=pos_traj,
        yaw_traj=yaw_traj,
        n_steps=n_steps,
        delta_t=delta_t,
    )


def test_message_to_traj_parses_flattened_positions():
    msg = _mock_traj_msg()
    traj = message_to_traj(msg, first_step_time_s=10.0)
    assert traj.T_pos.shape == (2, 3)
    assert traj.T_yaw.shape == (2,)
    assert np.allclose(traj.T_time, [0.0, 0.1])


def test_state_to_matrix_validates_length():
    state = np.arange(26, dtype=float)
    matrix = state_to_matrix(state, 2)
    assert matrix.shape == (2, 13)


def test_solve_assignment_returns_min_cost_permutation():
    start_points = [np.array([0.0, 0.0]), np.array([10.0, 0.0])]
    end_points = [np.array([9.5, 0.0]), np.array([0.5, 0.0])]
    assert solve_assignment(start_points, end_points) == [1, 0]


def test_build_full_state_command_and_hold_position_shapes():
    states = np.array(
        [
            [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    queried_pts = [
        types.SimpleNamespace(pos=np.array([0.1, 0.0, 0.6]), vel=np.zeros(3), acc=np.zeros(3), omega=0.2),
        types.SimpleNamespace(pos=np.array([1.1, 0.0, 0.6]), vel=np.zeros(3), acc=np.zeros(3), omega=-0.2),
    ]
    u = build_full_state_command(states, queried_pts, override_height=0.8)
    hold = hold_current_positions(states, height=0.7)
    assert u.shape == (32,)
    assert hold.shape == (32,)
    assert np.allclose(u[[2, 18]], [0.8, 0.8])
    assert np.allclose(hold[[2, 18]], [0.7, 0.7])
