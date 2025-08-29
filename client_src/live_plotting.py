import time
import numpy
import threading
from threading import Thread
import numpy as np
import math
from numpy import linalg as LA

import matplotlib.pyplot as plt


def threaded(fn):
    def wrapper(*a, **k):
        t = Thread(target=fn, args=a, kwargs=k, daemon=False)
        t.start();
        return t

    return wrapper


data_len = 50


class plotting_server:
    def __init__(self, cf_list):
        self.cf_list = cf_list
        self.num_cf = len(cf_list)
        print(cf_list)
        self.data = {}
        self.traj_all = {}
        self.target_click = {}
        self.history = {}
        self.target_flags = {}
        self.opt_flag = False
        self.click_count = 0
        self.t0 = time.time()
        self.time_list = np.zeros(data_len)

        # self.fake_gen()
        self.init()
        self.update()
        self.live_plot()
        plt.ion()

    def init(self):
        for cf in self.cf_list:
            self.data[cf] = np.zeros(6)
            self.history[cf] = np.zeros((data_len, 6))
            self.target_flags[cf] = False

    @threaded
    def update(self):
        while True:
            time.sleep(0.01)
            # shift history buffers
            for cf in self.cf_list:
                self.history[cf][:-1] = self.history[cf][1:]
                self.history[cf][-1] = self.data[cf]
                # print("plotting",self.data[cf][0:3])

    @threaded
    def live_plot(self, blit=False):

        fig = plt.figure(figsize=(10, 5))
        self.ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        self.ax = fig.add_subplot(1, 2, 2)

        # add two spheres at different centers
        u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:15j]
        r = 0.2  # sphere radius

        # define your two centers
        centers = [
            (0.5, 0.7, 0.4),  # sphere #1 at (0.5, 0, 0.5)
            (0.3, .0, .5)  # sphere #2 at (-0.5, 0.3, 0.2)
        ]

        for (cx, cy, cz) in centers:
            ## 3d spheres
            xs = cx + r * np.cos(u) * np.sin(v)
            ys = cy + r * np.sin(u) * np.sin(v)
            zs = cz + r * np.cos(v)
            self.ax1.plot_surface(xs, ys, zs,
                                  color='orange',
                                  alpha=0.4,
                                  linewidth=0,
                                  shade=True)
            ## 2d circles
            xs_2d = cx + r * np.cos(u)
            ys_2d = cy + r * np.sin(u)
            self.ax.plot(xs_2d,ys_2d)



        pos_data_all = {}
        target_all = {}
        traj_all = {}
        pos_data_all_2d = {}
        target_all_2d = {}
        for cf in self.cf_list:
            pos_data_all[cf], = self.ax1.plot([], [], [], 'g.', lw=3)
            target_all[cf], = self.ax1.plot([], [], [], 'r.', lw=3)
            traj_all[cf], = self.ax1.plot([], [], [], 'b.', lw=3)
            pos_data_all_2d[cf], = self.ax.plot([], [], 'g.', lw=3)
            target_all_2d[cf], = self.ax.plot([], [], 'r.', lw=3)

        # set up click‐handler
        fig.canvas.mpl_connect('button_press_event', self.onclick)

        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')

        fig.canvas.draw()  # note that the first draw comes before setting data

        if blit:
            # cache the background
            ax1background = fig.canvas.copy_from_bbox(self.ax1.bbox)

        plt.show(block=False)

        # for i in np.arange(10000):
        while (True):
            time.sleep(0.01)
            self.time_list[0:data_len - 1] = self.time_list[1:data_len]
            self.time_list[data_len - 1] = time.time() - self.t0
            # print(self.time_list)
            self.ax.set_xlim([-0.6, 1.5])
            self.ax.set_ylim([-0.6, 1.5])

            self.ax1.set_xlim3d(-0.6, 1.0)
            self.ax1.set_ylim3d(-0.6, 1.0)
            self.ax1.set_zlim3d(0.0, 1.0)

            x_traj = {}
            y_traj = {}
            z_traj = {}
            x_vel = {}
            for cf in self.cf_list:
                x_traj[cf] = self.history[cf][:, 0]
                y_traj[cf] = self.history[cf][:, 1]
                z_traj[cf] = self.history[cf][:, 2]
                ## plotting pos data
                pos_data_all[cf].set_data(x_traj[cf], y_traj[cf])
                pos_data_all[cf].set_3d_properties(z_traj[cf])
                pos_data_all_2d[cf].set_data(x_traj[cf], y_traj[cf])
                ## plotting trajectory
                if self.traj_all != {}:
                    traj_all[cf].set_data(self.traj_all[cf][:, 0],self.traj_all[cf][:, 1])
                    traj_all[cf].set_3d_properties(self.traj_all[cf][:,2])

                ## plotting desired position
                if self.target_flags[cf] == True: ## plot des pos
                    target_all[cf].set_data([self.target_click[cf][0]], [self.target_click[cf][1]])
                    target_all[cf].set_3d_properties([0.5])

                if self.target_flags[cf] == True:
                    target_all_2d[cf].set_data([self.target_click[cf][0]], [self.target_click[cf][1]])


            if blit:
                # restore background
                fig.canvas.restore_region(ax1background)

                # redraw just the points
                # ax1.draw_artist(line1)

                # coords = plt.ginput(5)
                # fill in the axes rectangle
                fig.canvas.blit(self.ax1.bbox)


            else:

                fig.canvas.draw()

            fig.canvas.flush_events()

    def onclick(self, event):
        # only record clicks on the 3D axes
        if event.inaxes is self.ax:
            x, y = event.xdata, event.ydata
            z = 0.0
            cf = self.cf_list[self.click_count % len(self.cf_list)]
            self.target_click[cf] = np.array([x, y, z])
            self.target_flags[cf] = True
            self.click_count += 1
            # send back to client
            if self.click_count == self.num_cf:
                self.opt_flag = True
                print(f"Clicked for {cf}: {self.target_click[cf]}")
                ## reset the click count
                self.click_count = 0

