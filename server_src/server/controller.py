import numpy as np
from .util import global_constants as const
from .util.util_fcns import cbf


def compute_velocity(pos:np.ndarray, error_history, name:str,pos_all:dict):
    ## construct obs list
    obs_r = []
    obs_list = []
    for key in pos_all:
        if key != name:
            obs_list.append(pos_all[key][-1,0:3])
            obs_r.append(const.R_quad)
    for i in range(const.num_obs):
        static_obs_i = const.X_obs[i]
        static_obs_r = const.R_obs[i]
        obs_list.append(static_obs_i)
        obs_r.append(static_obs_r)
    kp = const.P_kp
    kd = const.P_kd
    # print("Current error",error_history[name][-1,0:3])
    # print("Last error",error_history[name][-2,0:3])
    # print("Error diff", error_history[name][-1, 0:3] - error_history[name][-2, 0:3])
    v_des = kp * error_history[name][-1, 0:3] + kd * (
            error_history[name][-1, 0:3] - error_history[name][-2, 0:3]) / 0.02

    v_des = cbf(pos, v_des, obs_list, obs_r,0.02)
    return v_des
