import numpy as np
from dataclasses import dataclass
from scipy.interpolate import CubicSpline


@dataclass(slots=True)
class Traj:
    T_time: np.ndarray
    T_pos: np.ndarray
    T_yaw: np.ndarray


@dataclass(slots=True)
class TrajManagerCfg:
    lookahead_dt: float = 0.2
    initial_err_frac: float = 0.8
    err_decay_halflife: float = 0.2


@dataclass(slots=True)
class KinState:
    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray
    yaw: float
    omega: float


class TrajManager:
    """Stores and smoothly splices sampled position/yaw trajectories."""

    Cfg = TrajManagerCfg

    def __init__(self, cfg: TrajManagerCfg):
        self.start_time: float | None = None
        self.pos_spl: CubicSpline | None = None
        self.yaw_spl: CubicSpline | None = None
        self.vel_spl: CubicSpline | None = None
        self.acc_spl: CubicSpline | None = None
        self.omega_spl: CubicSpline | None = None

        self.T_time_prev: np.ndarray | None = None
        self.T_pos_prev: np.ndarray | None = None
        self.T_yaw_prev: np.ndarray | None = None

        self.T_time_spl_prev: np.ndarray | None = None
        self.T_pos_spl_prev: np.ndarray | None = None
        self.T_yaw_spl_prev: np.ndarray | None = None

        self.cfg = cfg

    def as_dict(self):
        return {
            "T_time_spl_prev": self.T_time_spl_prev,
            "T_pos_spl_prev": self.T_pos_spl_prev,
            "T_yaw_spl_prev": self.T_yaw_spl_prev,
        }

    def add_trajectory(self, curr_time: float, traj: Traj):
        spline_opts = dict(bc_type="not-a-knot", extrapolate=False)

        if self.start_time is None:
            self.start_time = traj.T_time[0]

        T_time_shifted = traj.T_time - self.start_time
        T_pos = traj.T_pos
        T_yaw = traj.T_yaw

        if self.pos_spl is None:
            self.T_time_spl_prev = T_time_shifted
            self.T_pos_spl_prev = T_pos
            self.T_yaw_spl_prev = T_yaw

            self.pos_spl = CubicSpline(T_time_shifted, T_pos, axis=0, **spline_opts)
            self.yaw_spl = CubicSpline(T_time_shifted, T_yaw, axis=0, **spline_opts)
        else:
            curr_time_shifted = curr_time - self.start_time
            t_splice_min = curr_time_shifted + self.cfg.lookahead_dt
            idx_new = np.searchsorted(traj.T_time - self.start_time, t_splice_min, side="left")
            idx_new = min(idx_new, len(T_time_shifted) - 1)
            t_splice = T_time_shifted[idx_new]

            pos_splice_old = self.pos_spl(t_splice)
            yaw_splice_old = self.yaw_spl(t_splice)

            T_time_new = T_time_shifted[idx_new:]
            T_pos_new = traj.T_pos[idx_new:]
            T_yaw_new = traj.T_yaw[idx_new:]
            T_yaw_new = np.unwrap(np.concatenate([np.array([yaw_splice_old]), T_yaw_new]))[1:]

            pos_err = pos_splice_old - T_pos_new[0]
            yaw_err = yaw_splice_old - T_yaw_new[0]

            lam = np.log(2) / self.cfg.err_decay_halflife
            T_decay_frac = self.cfg.initial_err_frac * np.exp(-lam * (T_time_new - t_splice))
            T_pos_corrected = T_pos_new + pos_err * T_decay_frac[:, None]
            T_yaw_corrected = T_yaw_new + yaw_err * T_decay_frac

            idx_prev_end = np.searchsorted(self.T_time_prev, t_splice_min, side="right") - 1
            T_time_prev_part = self.T_time_prev[: idx_prev_end + 1]
            T_pos_prev_part = self.T_pos_prev[: idx_prev_end + 1]
            T_yaw_prev_part = self.T_yaw_prev[: idx_prev_end + 1]

            T_time_cat = np.concatenate([T_time_prev_part, T_time_new])
            T_pos_cat = np.concatenate([T_pos_prev_part, T_pos_corrected])
            T_yaw_cat = np.concatenate([T_yaw_prev_part, T_yaw_corrected])

            self.T_time_spl_prev = T_time_cat
            self.T_pos_spl_prev = T_pos_cat
            self.T_yaw_spl_prev = T_yaw_cat

            self.pos_spl = CubicSpline(T_time_cat, T_pos_cat, axis=0, **spline_opts)
            self.yaw_spl = CubicSpline(T_time_cat, T_yaw_cat, axis=0, **spline_opts)

        self.vel_spl = self.pos_spl.derivative()
        self.acc_spl = self.vel_spl.derivative()
        self.omega_spl = self.yaw_spl.derivative()

        self.T_time_prev = T_time_shifted
        self.T_pos_prev = T_pos
        self.T_yaw_prev = T_yaw

    def at_end(self, time: float):
        time_shifted = time - self.start_time
        return time_shifted >= self.T_time_spl_prev[-1]

    def query(self, time: float, clip: bool = True):
        time_shifted = time - self.start_time
        if clip:
            time_shifted = np.clip(time_shifted, self.T_time_spl_prev[0], self.T_time_spl_prev[-1])

        return KinState(
            pos=self.pos_spl(time_shifted),
            vel=self.vel_spl(time_shifted),
            acc=self.acc_spl(time_shifted),
            yaw=self.yaw_spl(time_shifted),
            omega=self.omega_spl(time_shifted),
        )
