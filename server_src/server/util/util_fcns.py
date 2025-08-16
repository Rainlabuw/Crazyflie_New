import time
import numpy as np
from threading import Thread, Event
import threading
import cvxpy as cp
from numpy import linalg as LA
from scipy.spatial.transform import Rotation as rot
from .global_constants import x_max,x_min,y_max,y_min,z_max,z_min,R_obs,v_max
def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread

    return wrapper


def box_constratins(mover_i):
    if mover_i[0] >= x_max:
        mover_i[0] = x_max
    elif mover_i[0] <= x_min:
        mover_i[0] = x_min
    if mover_i[1] >= y_max:
        mover_i[1] = y_max
    elif mover_i[1] <= y_min:
        mover_i[1] = y_min
    if mover_i[2] >= z_max:
        mover_i[2] = z_max
    elif mover_i[2] <= z_min:
        mover_i[2] = z_min
    return mover_i


def q2e(q: np.ndarray):
    r = rot.from_quat(q)
    e = r.as_euler("xyz", degrees=True)
    return e

def e2q(e: np.ndarray):
    r = rot.from_euler("xyz", e, degrees=True)
    q = r.as_quat()
    return q



def cbf(X_t, v_des,X_obs,dt):
    u = cp.Variable(3)
    a = 1.2
    f = cp.norm(u - v_des, 2)
    h_t = LA.norm(X_t - X_obs,2)**2 - R_obs ** 2
    grad_h_i = 2 * (X_t - X_obs)
    f_t = u
    constraint = [2 * grad_h_i.T @ f_t >= -a * h_t]
    constraint.append(X_t[2] + u[2] * dt >= -0.001)
    constraint.append(cp.norm(u,2)<=v_max)
    problem = cp.Problem(cp.Minimize(f), constraint)
    problem.solve(solver=cp.CLARABEL)
    u_val = u.value
    return u_val

