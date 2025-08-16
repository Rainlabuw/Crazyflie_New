import numpy as np
from .util import global_constants as const
from .util.util_fcns import cbf


def compute_velocity(pos, error_history, name):
    kp = const.P_kp
    kd = const.P_kd
    v_max = const.v_max
    v_des = kp * error_history[name][-1, 0:3] + kd * (
            error_history[name][-1, 0:3] - error_history[name][-2, 0:3]) / 0.02

    v_des = cbf(pos, v_des, const.X_obs, 0.02)
    return v_des
