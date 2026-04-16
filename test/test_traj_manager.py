import numpy as np

from crazyflie_interface_py.traj_manager import Traj, TrajManager


def test_traj_manager_query_returns_expected_shapes():
    manager = TrajManager(TrajManager.Cfg())
    T_time = np.array([0.0, 0.5, 1.0])
    T_pos = np.array([[0.0, 0.0, 0.5], [0.5, 0.0, 0.5], [1.0, 0.0, 0.5]])
    T_yaw = np.array([0.0, 0.1, 0.2])
    manager.add_trajectory(0.0, Traj(T_time=T_time, T_pos=T_pos, T_yaw=T_yaw))

    kin = manager.query(0.25)
    assert kin.pos.shape == (3,)
    assert kin.vel.shape == (3,)
    assert kin.acc.shape == (3,)
    assert np.isclose(kin.pos[0], 0.25, atol=1e-6)


def test_traj_manager_splices_new_trajectory_without_losing_future():
    cfg = TrajManager.Cfg(lookahead_dt=0.2, initial_err_frac=0.8, err_decay_halflife=0.2)
    manager = TrajManager(cfg)

    traj0 = Traj(
        T_time=np.array([0.0, 0.5, 1.0, 1.5]),
        T_pos=np.array([[0.0, 0.0, 0.5], [0.5, 0.0, 0.5], [1.0, 0.0, 0.5], [1.5, 0.0, 0.5]]),
        T_yaw=np.zeros(4),
    )
    manager.add_trajectory(0.0, traj0)

    traj1 = Traj(
        T_time=np.array([0.6, 1.1, 1.6, 2.1]),
        T_pos=np.array([[0.6, 0.2, 0.5], [1.1, 0.2, 0.5], [1.6, 0.2, 0.5], [2.1, 0.2, 0.5]]),
        T_yaw=np.array([0.0, 0.1, 0.2, 0.3]),
    )
    manager.add_trajectory(0.45, traj1)

    kin = manager.query(1.2)
    assert kin.pos.shape == (3,)
    assert kin.pos[0] > 0.8
    assert manager.T_time_spl_prev[-1] >= 2.0
