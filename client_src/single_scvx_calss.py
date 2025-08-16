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
def x_initial(x_ini, x_des,robots_name,T):
    x_traj = {}
    for name in robots_name:
        x_traj[name] = np.linspace(x_ini[name], x_des[name], T)
        # for t in range(T):
        #     x_traj[name][t,:] = x_ini[name]
    return x_traj




class optmization_template:
    def __init__(self,x_ini_0,x_des_0):
        x_ini_0 = np.hstack((x_ini_0,np.zeros(6)))
        x_des_0 = np.hstack((x_des_0, np.zeros(6)))
        self.Tf = 15
        self.T0 = 0
        self.T = 51
        self.t_traj = np.linspace(self.T0, self.Tf, self.T)
        self.dt = self.t_traj[1] - self.t_traj[0]
        self.n = 6  ## number of states
        self.m = 3  ## number of controls
        self.trust_region = 0.25
        self.max_iter = 1
        self.N_agents = 3  # Number of agents
        self.robots_name = ["robot01", "robot02", "robot03"]
        self.R = 0.3  # agent radius

        ## Specify the desired states
        self.x_ini = {}
        self. x_des = {}

        self.x_ini["robot01"] = x_ini_0
        # self.x_des["robot01"] = np.array([1.4, 1.0, 1.0,
        #                              0, 0, 0,
        #                              0, 0, 0])
        self.x_des["robot01"] = x_des_0
        self.x_ini["robot02"] = np.array([0.5, 0.6, 0.3,
                                     0, 0, 0,
                                     0, 0, 0])
        self.x_des["robot02"] = np.array([0.5, 0.6, 0.3,
                                     0, 0, 0,
                                     0, 0, 0])

        self.x_ini["robot03"] = np.array([3, .5, .5,
                                     0, 0, 0,
                                     0, 0, 0])
        self.x_des["robot03"] = np.array([3, .5, .5,
                                     0, 0, 0,
                                     0, 0, 0])

        ## get descrete LTI
        [self.Ad, self.Bd] = descete_f(self.dt)
        ## Cost list
        self.cost_list = np.zeros(self.max_iter)
        self.traj_gen()


    ## main traj fcn
    def traj_gen(self):
        ## Initialization (straight line)
        self.X_traj = x_initial(self.x_ini, self.x_des,self.robots_name,self.T)

        # ## Plotting initial traj
        # plot_traj(self.X_traj, self.T)

        # ## Begin optimization loop
        # for iter in range(self.max_iter):
        #     # print(self.trust_region)
        #     self.x_traj_opt()
        #     # if iter % 1 == 0:
        #     #     self.plot_traj()
        #     # trust_region = trust_region / 2
        #     self.cost_list[iter] = self.cost_fcn()
        #     print("Actual cost: ", self.cost_list[iter])
        #     if iter >= 1:
        #         if self.cost_list[iter] > self.cost_list[iter - 1]:
        #             self.trust_region = self.trust_region / 2

    def cost_fcn(self):
        n = self.n
        m = self.m
        T = self.T
        cost_iter = 0
        for name in self.robots_name:
            X_traj_i = self.X_traj[name]
            u_traj_i = X_traj_i[0:T - 1, n:n + m]
            for t in range(T - 1):
                cost_iter += LA.norm(u_traj_i[t, :], 2) ** 2
        return cost_iter

    ## Plotting
    def plot_traj(self):
        R = self.R
        # Create a sphere parameters
        phi, theta = np.linspace(0, np.pi, 20), np.linspace(0, 2 * np.pi, 20)
        phi, theta = np.meshgrid(phi, theta)

        # Create the 3D plot
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ## plotting the time traj
        for t in range(self.T):
            ax.clear()
            for name in self.robots_name:
                X_traj_i = self.X_traj[name]
                ax.plot(X_traj_i[:, 0], X_traj_i[:, 1], X_traj_i[:, 2])  ## plotting entire trajectories
                x_i = R * np.sin(phi) * np.cos(theta) + X_traj_i[t, 0]
                y_i = R * np.sin(phi) * np.sin(theta) + X_traj_i[t, 1]
                z_i = R * np.cos(phi) + X_traj_i[t, 2]
                # Plot the sphere
                ax.plot_surface(x_i, y_i, z_i, color='cyan', alpha=0.3, edgecolor='none')

            # Set the limits
            ax.set_xlim([-0.5, 2])
            ax.set_ylim([-0.5, 2])
            ax.set_zlim([0, 2])
            plt.pause(0.1)
        plt.close(fig)


    def x_traj_opt(self):
        n = self.n
        m = self.m
        T = self.T
        s_val = {}
        s_bar_val = {}
        s_bar_val_new = {}
        diff = 0
        ## Initialize the perturbation variables
        for name in self.robots_name:
            s_val[name] = np.zeros((T, n + m))
            s_bar_val[name] = np.zeros((T, n + m))
            s_bar_val_new[name] = np.zeros((T, n + m)) + 1
            diff += LA.norm(s_bar_val[name] - s_bar_val_new[name], 2)
        # while diff > tol:
        for iter in range(1):
            ##########################################################
            ## solve d - minimization individually (primal variables)
            for name in self.robots_name:
                X_des_i = self.x_des[name]
                x_des_i = X_des_i[0:n]
                X_traj_i = self.X_traj[name]
                x_traj_i = X_traj_i[0:T, 0:n]
                u_traj_i = X_traj_i[0:T - 1, n:n + m]  # extract the control
                # Primary variables
                s_i = cp.Variable((T, n + m))
                S_i = cp.Variable(T)
                s_i_vec = cp.reshape(s_i, (T * (n + m), 1), order="C")  ## vectorized primal variables
                d_i = s_i[0:T, 0:n]
                w_i = s_i[0:T - 1, n:n + m]
                # Duplicate variables
                s_bar_val_i = s_bar_val[name]
                s_bar_val_i_vec = np.reshape(s_bar_val_i, (T * (n + m), 1),
                                             order="C")  ## vectorized duplicated variables
                ##########################################################
                ## s - minimization (primal variables)
                # Construct the augmented Lagrangian
                # L_rho = 1*cp.sum_squares(u_traj_i + w_i) + r_i.T @ (s_i_vec - s_bar_val_i_vec) + rho / 2 * cp.square(
                #     cp.norm(s_i_vec - s_bar_val_i_vec, 2))
                L_rho = 1 * cp.sum_squares(u_traj_i + w_i) + 10000 * cp.norm(S_i, 1)
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
                    constraints_s.append(x_traj_t[0] + d_t[0] <= 22)
                    constraints_s.append(x_traj_t[0] + d_t[0] >= -1)
                    constraints_s.append(x_traj_t[1] + d_t[1] >= -1)
                    constraints_s.append(x_traj_t[1] + d_t[1] <= 20)

                    # loop through obstacles (test use)
                    S_t = S_i[t]
                    for obs_name in self.robots_name:
                        X_traj_j = self.X_traj[obs_name]
                        x_traj_j = X_traj_j[0:T, 0:n]
                        x_traj_j_t = x_traj_j[t, :]
                        if obs_name != name:  ## exclude itself
                            S = 2 * self.R - LA.norm(x_traj_t[0:3] - x_traj_j_t[0:3], 2)
                            S_grad = (x_traj_t[0:3] - x_traj_j_t[0:3]).T / cp.norm(x_traj_t[0:3] - x_traj_j_t[0:3], 2)

                            # S = R ** 2 -  LA.norm(x_traj_t[0:2] - x_traj_j_t[0:2], 2) ** 2
                            # S_grad = 2 * (x_traj_t[0:2] - x_traj_j_t[0:2]).T
                            constraints_s.append(
                                S - S_grad @ d_t[0:3] <= S_t
                            )
                            constraints_s.append(S_t >= 0)

                problem = cp.Problem(cp.Minimize(L_rho), constraints_s)
                problem.solve(solver=cp.CLARABEL)
                s_val[name] = s_i.value

        print("update X")
        for name in self.robots_name:
            if s_val[name].all() != None:
                self.X_traj[name] += np.array(s_val[name])

