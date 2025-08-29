import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
from numpy import linalg as LA
from scipy import signal


## dynamics
def descete_f(dt):
    A = np.array([[0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]])
    B = np.array([[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    n = len(A[0])
    m = len(B[0])
    C = np.eye(n)
    D = np.zeros((n, m))
    sys = signal.StateSpace(A, B, C, D)
    sysd = sys.to_discrete(dt)
    Ad = sysd.A
    Bd = sysd.B
    return [Ad, Bd]


## Initializatoin
def x_initial(x_ini_i, x_des_i, T):
    x_traj_i = np.zeros((T,9))
    for i in range(len(x_ini_i)):
        x_traj_i[:, i] = np.linspace(x_ini_i[i], x_des_i[i], T)
    return x_traj_i


class optmization_template:
    def __init__(self, x_ini: dict, x_des: dict, cf_list):
        self.x_ini = x_ini
        self.x_des = x_des
        self.Tf = 15
        self.T0 = 0
        self.T = 51
        self.t_traj = np.linspace(self.T0, self.Tf, self.T)
        self.dt = self.t_traj[1] - self.t_traj[0]
        self.n = 6  ## number of states
        self.m = 3  ## number of controls
        self.trust_region = 0.25
        self.max_iter = 1
        self.cf_list = cf_list
        self.N_agents = len(self.cf_list)  # Number of agents
        self.R = 0.15  # agent radius
        self.X_traj = {}
        self.ini()

    def ini(self):
        self.obs_r = np.array([0.2,0.2])
        self.obs_x = np.array([[0.5, 0.7, 0.4],[0.3, .0, .5]])
        self.num_obs = len(self.obs_r)
        for cf in self.cf_list:
            self.x_ini[cf] = np.hstack((self.x_ini[cf], np.zeros(6)))
            self.x_des[cf] = np.hstack((self.x_des[cf], np.zeros(6)))
            self.X_traj[cf] = x_initial(self.x_ini[cf], self.x_des[cf], self.T)


        ## get descrete LTI
        [self.Ad, self.Bd] = descete_f(self.dt)
        ## Cost list
        self.cost_list = np.zeros(self.max_iter)
        self.traj_gen()

    ## main traj fcn
    def traj_gen(self):
        ## Initialization (straight line)
        print("Traj generated")

        # ## Plotting initial traj
        # plot_traj(self.X_traj, self.T)

        ## Begin optimization loop
        for iter in range(self.max_iter):
            print("iteration",iter)
            # print(self.trust_region)
            self.x_traj_opt()
            # if iter % 1 == 0:
            #     self.plot_traj()
            # trust_region = trust_region / 2
            self.cost_list[iter] = self.cost_fcn()
            print("Actual cost: ", self.cost_list[iter])
            if iter >= 1:
                if self.cost_list[iter] > self.cost_list[iter - 1]:
                    self.trust_region = self.trust_region / 2

    def cost_fcn(self):
        n = self.n
        m = self.m
        T = self.T
        cost_iter = 0
        for name in self.cf_list:
            X_traj_i = self.X_traj[name]
            u_traj_i = X_traj_i[0:T - 1, n:n + m]
            for t in range(T - 1):
                cost_iter += LA.norm(u_traj_i[t, :], 2) ** 2
        return cost_iter



    def x_traj_opt(self):
        n = self.n
        m = self.m
        T = self.T
        s_val = {}
        s_bar_val = {}
        s_bar_val_new = {}
        diff = 0
        ## Initialize the perturbation variables
        for name in self.cf_list:
            s_val[name] = np.zeros((T, n + m))
            s_bar_val[name] = np.zeros((T, n + m))
            s_bar_val_new[name] = np.zeros((T, n + m)) + 1
            diff += LA.norm(s_bar_val[name] - s_bar_val_new[name], 2)
        # while diff > tol:
        for iter in range(1):
            ##########################################################
            for name in self.cf_list:
                X_des_i = self.x_des[name]
                x_des_i = X_des_i[0:n]
                X_traj_i = self.X_traj[name]
                x_traj_i = X_traj_i[0:T, 0:n]
                u_traj_i = X_traj_i[0:T - 1, n:n + m]  # extract the control
                # Primary variables
                s_i = cp.Variable((T, n + m))
                S_i = cp.Variable((T,self.num_obs)) ## for static obs
                S_i_cf = cp.Variable((T,self.N_agents))
                d_i = s_i[0:T, 0:n]
                w_i = s_i[0:T - 1, n:n + m]
                # Duplicate variables
                s_bar_val_i = s_bar_val[name]
                s_bar_val_i_vec = np.reshape(s_bar_val_i, (T * (n + m), 1),
                                             order="C")  ## vectorized duplicated variables
                ##########################################################

                L_rho = 1 * cp.sum_squares(u_traj_i + w_i) + 10000 * (cp.norm(S_i, 1) + cp.norm(S_i_cf, 1))
                constraints_s = [d_i[0, :] == np.zeros(n)]
                constraints_s.append(d_i[T - 1, :] + x_traj_i[T - 1, :] == x_des_i)
                for t in range(T - 1):
                    x_traj_t = x_traj_i[t, :]
                    x_traj_tp1 = x_traj_i[t + 1, :]
                    u_traj_t = u_traj_i[t, :]
                    d_t = d_i[t, :]
                    d_tp1 = d_i[t + 1, :]
                    w_t = w_i[t, :]
                    f_t = self.Ad @ x_traj_t + self.Bd @ u_traj_t
                    constraints_s.append(x_traj_tp1 + d_tp1 == f_t + self.Ad @ d_t + self.Bd @ w_t)
                    if name == "robot02" or name == "robot03":
                        constraints_s.append(cp.norm(w_t, 1) <= 0)
                    else:
                        constraints_s.append(cp.norm(w_t, 1) <= self.trust_region)

                    ## boundary constraints
                    # constraints_s.append(x_traj_t[0] + d_t[0] <= 22)
                    # constraints_s.append(x_traj_t[0] + d_t[0] >= -0.1)
                    constraints_s.append(x_traj_t[2] + d_t[2] >= -0.01)
                    # constraints_s.append(x_traj_t[1] + d_t[1] <= 20)

                    # loop through obstacles (static obs)
                    S_t = S_i[t]
                    S_t_cf = S_i_cf[t]
                    for j in range(len(self.obs_r)):
                        S_t_j = S_t[j]
                        obs_j = self.obs_x[j]
                        obs_r_j = self.obs_r[j]
                        h_j = obs_r_j ** 2 - LA.norm(x_traj_t[0:3] - obs_j)**2
                        grad_h = - 2 * (x_traj_t[0:3] - obs_j)
                        constraints_s.append(h_j + grad_h @ d_t[0:3] <= S_t_j)
                        constraints_s.append(S_t_j >= 0)
                    cf_count = 0
                    for name_j in self.cf_list:
                        S_t_cf_j = S_t_cf[cf_count]
                        x_cf_j = self.X_traj[name_j][t,0:3]
                        r = self.R
                        if name != name_j:
                            h_j = r ** 2 - LA.norm(x_traj_t[0:3] - x_cf_j) ** 2
                            grad_h = - 2 * (x_traj_t[0:3] - x_cf_j)
                            constraints_s.append(h_j + grad_h @ d_t[0:3] <= S_t_cf_j)
                        constraints_s.append(S_t_cf_j >= 0)
                        cf_count += 1




                problem = cp.Problem(cp.Minimize(L_rho), constraints_s)
                problem.solve(solver=cp.CLARABEL)
                s_val[name] = s_i.value
                self.X_traj[name] = self.X_traj[name] + s_val[name]

    print("update X")

