import motioncapture
import time
import numpy as np
from .util import global_constants as const
from .util.util_fcns import threaded
from .util.util_fcns import e2q,q2e,box_constratins
from .controller import compute_velocity
class Vicon_class:
    def __init__(self,vicon_ip,crazyflies,print_t ,record_fcn,is_open,time_list_vicon):
        self.vicon_ip = vicon_ip
        self.crazyflies = crazyflies
        self.print_t = print_t
        self.is_open = is_open
        self.record = record_fcn
        self.time_list_vicon = time_list_vicon
        self.record_count = 0
        self._running = True
    def stop_logging(self):
        self._running = False

    @threaded
    def run(self):
        mocap = motioncapture.connect("vicon", {'hostname': self.vicon_ip})
        while self._running:
            t_vicon = time.time()
            self.time_list_vicon[0:9] = self.time_list_vicon[1:10]
            self.time_list_vicon[9] = float(t_vicon)
            dt = (self.time_list_vicon[9] - self.time_list_vicon[0]) / 10
            mocap.waitForNextFrame()
            pos_data = mocap.rigidBodies.items()
            for name, obj in pos_data:
                if self.record_count % self.print_t == 0:
                    self.Control_Flag = True  ## activeate the controller
                    # print("Received vicon data, dt :",dt,"state:    ",obj.position[0], obj.position[1], obj.position[2])
                self.record("vicon-", name, obj.position[0], obj.position[1], obj.position[2], obj.rotation.x,
                            obj.rotation.y, obj.rotation.z, obj.rotation.w)
                for cf in self.crazyflies:
                    if name == cf.object_name:
                        cf.cf.extpos.send_extpos(obj.position[0], obj.position[1], obj.position[2])
                        # cf.cf.extpos.send_extpose( obj.position[0], obj.position[1], obj.position[2], obj.rotation.x, obj.rotation.y, obj.rotation.z, obj.rotation.w )
            self.record_count += 1


class EKF_class:
    def __init__(self, crazyflies, data_history, record_fn, data_length):
        self.crazyflies     = crazyflies        # list of cf objects
        self.data_history   = data_history      # dict: name → (data_length×13) array
        self.record         = record_fn         # your CrazyflieServer.record method
        self.data_length    = data_length
        self.record_count_ekf = 0
        self.time_list_ekf  = np.zeros(10)
        self._running = True


    def stop_logging(self):
        self._running = False

    @threaded
    def run(self):
         while self._running:
            self._running = True
            self.record_count_ekf += 1
            t_ekf = time.time()
            self.time_list_ekf[0:9] = self.time_list_ekf[1:10]
            self.time_list_ekf[9] = float(t_ekf)
            dt = (self.time_list_ekf[9] - self.time_list_ekf[0]) / 10
            for cf in self.crazyflies:
                name = cf.object_name
                ## shift data
                self.data_history[name][0:const.data_length - 1] = self.data_history[name][1:const.data_length]

                current_data_i = np.zeros(10)  ## pos + att + vel
                state = np.zeros(7)
                for log_pos_entry in cf.cflog_pos:
                    pos_data = log_pos_entry[1]
                    # self.record( "kalman-" , cf.object_name, data['kalman.stateX'], data['kalman.stateY'], data['kalman.stateZ']);
                    state[0:3] = np.array([pos_data['stateEstimate.x'],
                                           pos_data['stateEstimate.y'],
                                           pos_data['stateEstimate.z']])
                    break

                for log_att_entry in cf.cflog_att:
                    att_data = log_att_entry[1]
                    state[3:7] = np.array([att_data['stateEstimate.qx'],
                                           att_data['stateEstimate.qy'],
                                           att_data['stateEstimate.qz'],
                                           att_data['stateEstimate.qw']])
                    break
                # print(state)
                for log_vel_entry in cf.cflog_vel:
                    vel_data = log_vel_entry[1]
                    vel = np.array([vel_data['stateEstimate.vx'],
                                    vel_data['stateEstimate.vy'],
                                    vel_data['stateEstimate.vz']])
                    break
                current_data_i[0:7] = state[0:7]
                current_data_i[7:10] = vel
                ## add new data (dont take pos)
                self.data_history[name][const.data_length - 1][3:10] = current_data_i[3:10]

                # if self.record_count_ekf % 50 == 0:
                #     print("EKF euler:   ",euler)
                #     print("EKF q:    ",state[3:7])
                self.record("kalman-", cf.object_name, state[0], state[1], state[2], state[3], state[4], state[5],
                            state[6])

class CFlib_class:
    def __init__(self,server):
        self.server = server

    def stop_bridge_comm(self):
        self.server.Control_Flag = False

    def pos_controller(self, cf):
        name = cf.object_name
        pos_history = self.server.data_history[name][:, 0:3]
        # print(pos_history[-1])
        if self.server.hover_flag[name] == True:
            wps = np.array([0.0, 0.0, 0.5])
            # print("Hovering")
        else:
            wps = self.server.mover[name][0:3]
            # print("Normal flight, wps:  ",wps)
        ## update the error data
        self.server.error_history[name][0:const.data_length - 1] = self.server.error_history[name][1:const.data_length]
        self.server.error_history[name][-1, 0:3] = wps - pos_history[-1]
        v_des = compute_velocity(pos_history[-1],self.server.error_history,name)
        print("vdes:    ",v_des)
        # cf.cf.commander.send_velocity_world_setpoint(v_des[0], v_des[1], v_des[2], 0)
        # print("cmd_vel: ",name,v_des)
        return v_des

    @threaded
    def run(self):
        if self.server.Control_Flag == True:
            print("Controller Activated")
        else:
            print("Controller failed")
        while (self.server.Control_Flag):
            t_control = time.time()
            t_elp = t_control - self.server.start_time
            self.server.time_list_control[0:9] = self.server.time_list_control[1:10]
            self.server.time_list_control[9] = float(t_control)
            dt_control = (self.server.time_list_control[9] - self.server.time_list_control[0]) / 10

            ## close the system and save
            if t_control - self.server.start_time >= const.duration:
                self.server.close()
                self.server.save_flag = False
                self.server.killswitch()
                self.server.Flag = False

            # if count % self.print_t ==0:
            #     print("control dt:  ", dt_control)
            # time.sleep(0.01)
            time.sleep(0.02)
            current_pos = self.server.curr_data.copy()
            for cf in self.server.crazyflies:
                self.server.mover_history[cf.object_name][0:const.data_length - 1] = self.server.mover_history[cf.object_name][
                                                                              1:const.data_length]
                self.server.mover_history[cf.object_name][-1] = self.server.mover[cf.object_name]
                current_pos_i = current_pos["vicon-" + cf.object_name][1:4]
                self.server.data_history[cf.object_name][-1, 0:3] = current_pos_i
                self.server.mover[cf.object_name] = box_constratins(self.server.mover[cf.object_name])
                print(self.server.mover[cf.object_name])
                q_i = self.server.mover[cf.object_name][3:7]
                euler_i = q2e(q_i)
                yaw_i = euler_i[-1]
                yaw_i = 0  ## set to 0 for test
                if self.server.mover[cf.object_name][2] == 0:  ## Hover (no cmd received or computing)
                    ## in optimization code, send zeros when computing to enable hover

                    if self.server.hover_flag[cf.object_name] == False:
                        self.server.hover[cf.object_name] = self.server.mover_history[cf.object_name][-2][0:3]
                        ## store the second last wps as hover point

                    self.server.hover_flag[cf.object_name] = True  ## enter hover mode
                    vel_des = self.pos_controller(cf)
                    self.server.data_history[cf.object_name][-1, 10:13] = vel_des
                    # print("vel_cmd: ",vel_des)
                    # if t_elp % 1 == 0:
                    # print("time", t_elp)
                else:
                    self.server.hover_flag[cf.object_name] = False
                    vel_des = self.pos_controller(cf)
                    self.server.data_history[cf.object_name][-1, 10:13] = vel_des
                    # print("vel_cmd: ",vel_des)
