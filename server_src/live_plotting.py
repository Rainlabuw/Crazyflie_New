import time
import numpy
import threading
from threading import Thread
import numpy as np
import math
from numpy import linalg as LA
import matplotlib.pyplot as plt

##
n = 7
data_length = 50



def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread

    return wrapper


class live_plotting_class:

    def __init__(self, cf_list):
        self.cf_list = cf_list
        self.curr_state = {}
        self.curr_vel_data = {}
        self.state_history = {}
        self.vel_history = {}
        self.init()
        self.history_updt()
        self.time_list = np.zeros(data_length)
        self.t0 = time.time()
        self.live_plot()
        # self.prin()




    def init(self):
        for cf in self.cf_list:
            self.curr_state[cf] = np.zeros(n)
            self.state_history[cf] = np.zeros((data_length, n))
            self.curr_vel_data[cf] = np.zeros(6) ## actual vel and vel cmd
            self.vel_history[cf] = np.zeros((data_length, 6))

    @threaded
    def history_updt(self):
        for cf in self.cf_list:
            self.state_history[cf][0:data_length-1, :] = self.state_history[cf][1:data_length,:]
            self.state_history[cf][data_length-1,:] = self.curr_state[cf]
            self.vel_history[cf][0:data_length-1, :] = self.vel_history[cf][1:data_length,:]
            self.vel_history[cf][data_length-1,:] = self.curr_vel_data[cf]


    @threaded
    def live_plot(self, blit=False):

        fig = plt.figure(figsize=(5, 5))
        ax0 = fig.add_subplot(2,1,1, projection='3d')

        # fig = plt.figure(figsize=(5, 5))
        ## plotting vel data
        ax1 = fig.add_subplot(2, 3, 4) # u
        ax2 = fig.add_subplot(2, 3, 5) # v
        ax3 = fig.add_subplot(2, 3, 6) # w

        pos_data_all = {}
        x_vel_all = {}
        y_vel_all = {}
        z_vel_all = {}
        x_vel_cmd_all = {}
        y_vel_cmd_all = {}
        z_vel_cmd_all = {}
        for cf in self.cf_list:
            pos_data_all[cf], = ax0.plot([], [], [], 'g.', lw=3)
            x_vel_all[cf], = ax1.plot([], [], 'r.', lw=3)
            y_vel_all[cf], = ax2.plot([], [], 'r.', lw=3)
            z_vel_all[cf], = ax3.plot([], [], 'r.', lw=3)
            x_vel_cmd_all[cf], = ax1.plot([], [], 'g.', lw=3)
            y_vel_cmd_all[cf], = ax2.plot([], [], 'g.', lw=3)
            z_vel_cmd_all[cf], = ax3.plot([], [], 'g.', lw=3)

        ax0.set_xlim([-1, 1])
        ax0.set_ylim([-1, 1])
        ax0.set_zlim([0, 2])
        ax0.set_xlabel('X')
        ax0.set_xlabel('Y')
        ax0.set_zlabel('z')
        ax1.legend("Drone")

        ax1.set_xlabel('time')
        ax1.set_ylabel('u(m/s)')
        ax1.legend(["vel","vel_cmd"])
        ax2.set_xlabel('time')
        ax2.set_ylabel('v(m/s)')
        ax2.legend(["vel","vel_cmd"])
        ax3.set_xlabel('time')
        ax3.set_ylabel('w(m/s)')
        ax3.legend(["vel","vel_cmd"])
        fig.canvas.draw()  # note that the first draw comes before setting data

        if blit:
            # cache the background
            ax1background = fig.canvas.copy_from_bbox(ax0.bbox)

        plt.show(block=False)

        # for i in np.arange(10000):
        while (True):
            time.sleep(0.01)
            self.time_list[0:data_length - 1] = self.time_list[1:data_length]
            self.time_list[data_length - 1] = time.time() - self.t0
            ax1.set_xlim([self.time_list[0], self.time_list[data_length - 1]])
            ax2.set_xlim([self.time_list[0], self.time_list[data_length - 1]])
            ax3.set_xlim([self.time_list[0], self.time_list[data_length - 1]])
            ax1.set_ylim([-1, 1]);ax2.set_ylim([-1, 1]);ax3.set_ylim([-1, 1])

            x_traj = {}
            y_traj = {}
            z_traj = {}
            x_vel = {}
            y_vel = {}
            z_vel = {}
            x_vel_cmd = {}
            y_vel_cmd = {}
            z_vel_cmd = {}
            for cf in self.cf_list:
                x_traj[cf] = self.state_history[cf][:, 0]
                y_traj[cf] = self.state_history[cf][:, 1]
                z_traj[cf] = self.state_history[cf][:, 2]
                pos_data_all[cf].set_data(x_traj[cf], y_traj[cf])
                pos_data_all[cf].set_3d_properties(z_traj[cf])

                x_vel[cf] = self.vel_history[cf][:, 0]
                x_vel_all[cf].set_data(self.time_list, x_vel[cf])
                y_vel[cf] = self.vel_history[cf][:, 1]
                y_vel_all[cf].set_data(self.time_list, y_vel[cf])
                z_vel[cf] = self.vel_history[cf][:, 2]
                z_vel_all[cf].set_data(self.time_list, z_vel[cf])

                x_vel_cmd[cf] = self.vel_history[cf][:, 3]
                x_vel_cmd_all[cf].set_data(self.time_list, x_vel_cmd[cf])
                y_vel_cmd[cf] = self.vel_history[cf][:, 4]
                y_vel_cmd_all[cf].set_data(self.time_list, y_vel_cmd[cf])
                z_vel_cmd[cf] = self.vel_history[cf][:, 5]
                z_vel_cmd_all[cf].set_data(self.time_list, z_vel_cmd[cf])

            # line1.set_data(x, y)
            # line1.set_3d_properties(z)

            if blit:
                # restore background
                fig.canvas.restore_region(ax1background)

                # redraw just the points
                # ax1.draw_artist(line1)

                # coords = plt.ginput(5)
                # fill in the axes rectangle
                fig.canvas.blit(ax0.bbox)


            else:

                fig.canvas.draw()

            fig.canvas.flush_events()

