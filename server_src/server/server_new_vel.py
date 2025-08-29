import time
import numpy
import threading
import cflib.crtp as crtp
import motioncapture
import numpy as np
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from .util.util_fcns import threaded, box_constratins, e2q, q2e
from .util import global_constants as const
from .classes import EKF_class, Vicon_class, CFlib_class


class TrackingObject:
    uri = "UNKNOWN"
    object_name = "UNKNOWN"

    def __init__(self, uri, object_name):
        self.uri = uri
        self.object_name = object_name


class CrazyflieServer:

    def __init__(self, crazyflies=[], vicon_ip="", kill_sleep=20):

        self.save_flag = True
        self.print_t = 50.0
        self.lock = threading.Lock()
        self.is_open = True
        self.Control_Flag = False
        self.log_data = "entity,time,x,y,z,p,r,t,w\n"
        self.curr_data = {}
        self.data_history = {}
        self.error_history = {}
        self.threads = []
        self.crazyflies = crazyflies
        ## create name list
        self.cf_list = []
        for cf in self.crazyflies:
            self.cf_list.append(cf.object_name)
        self.mover = {}
        self.mover_history = {}
        self.hover = {}
        self.hover_flag = {}
        ## initialize the data
        self.initialization()
        self.record_count = 0.0
        self.time_list_vicon = np.zeros(10)
        self.time_list_control = np.zeros(10)
        print("1) Connecting to Crazyflies_Justin_vel")
        crtp.init_drivers()
        log_config = LogConfig(name="pos_est", period_in_ms=10)
        log_config.add_variable('stateEstimate.x', 'float')
        log_config.add_variable('stateEstimate.y', 'float')
        log_config.add_variable('stateEstimate.z', 'float')

        # log_config2 = LogConfig(name="att_est", period_in_ms=10)
        # log_config2.add_variable('stateEstimate.qx', 'float')
        # log_config2.add_variable('stateEstimate.qy', 'float')
        # log_config2.add_variable('stateEstimate.qz', 'float')
        # log_config2.add_variable('stateEstimate.qw', 'float')

        log_config3 = LogConfig(name="euler_est", period_in_ms=10)
        log_config3.add_variable('stateEstimate.vx', 'float')
        log_config3.add_variable('stateEstimate.vy', 'float')
        log_config3.add_variable('stateEstimate.vz', 'float')

        for cf in self.crazyflies:
            print(f"Connecting crazyflie {cf.object_name} to radio: {cf.uri}")
            cf.cf = Crazyflie()
            cf.cflog_pos = SyncLogger(cf.cf, log_config)
            # cf.cflog_att = SyncLogger(cf.cf, log_config2)
            cf.cflog_vel = SyncLogger(cf.cf, log_config3)
            cf.cf.open_link(cf.uri)
        time.sleep(1.0)
        print("2) Initialzing to Telemetry System")
        print(f"Connecting to Vicon with IP: {vicon_ip}")
        self.vicon_ip = vicon_ip
        self.t0 = time.time()
        self.vicon_logger = Vicon_class(
            vicon_ip=self.vicon_ip,
            crazyflies=self.crazyflies,
            print_t=self.print_t,
            record_fcn=self.record,
            is_open=self.is_open,
            time_list_vicon=self.time_list_vicon
        )
        self.vicon_logger.run()
        # self.spawn_thread( self.vicon )
        time.sleep(1.0)
        print("3) Initializing Crazyflies Kalman Estimator")
        kvar_cfg = LogConfig(name="var", period_in_ms=500)
        kvar_cfg.add_variable('kalman.varPX', 'float')
        kvar_cfg.add_variable('kalman.varPY', 'float')
        kvar_cfg.add_variable('kalman.varPZ', 'float')

        for cf in self.crazyflies:
            cf.cf.param.set_value('stabilizer.estimator', '2')
            cf.cf.param.set_value('locSrv.extQuatStdDev', 0.06)
            cf.cf.param.set_value('kalman.resetEstimation', '1')
            cf.kvar_log = SyncLogger(cf.cf, kvar_cfg)
            cf.var_history = [[1000] * 10, [1000] * 10, [1000] * 10]
            cf.kvar_log.connect()
        print("4) Awaiting Kalman Estimator")
        var_active = True
        iteration = 0
        while var_active:
            var_active = False
            for cf in self.crazyflies:
                for log_entry in cf.kvar_log:
                    data = log_entry[1]
                    cf.var_history[0][iteration] = data['kalman.varPX'];
                    cf.var_history[1][iteration] = data['kalman.varPY'];
                    cf.var_history[2][iteration] = data['kalman.varPZ'];
                    threshold = 0.01
                    print(f"Kalman variance: {numpy.amax(cf.var_history) - numpy.amin(cf.var_history)}")
                    if (numpy.amax(cf.var_history) - numpy.amin(cf.var_history)) > threshold:
                        var_active = True
                    break
            iteration = (iteration + 1) % 10
        time.sleep(1.0)
        print("5) Arming Drones")
        for cf in self.crazyflies:
            cf.cf.param.set_value('kalman.resetEstimation', '0')
            cf.kvar_log.disconnect()
        time.sleep(1.0)
        for cf in self.crazyflies:
            cf.cflog_pos.connect()
            # cf.cflog_att.connect()
            cf.cflog_vel.connect()
            cf.cf.platform.send_arming_request(True)

        self.ekf_logging = EKF_class(
            crazyflies=self.crazyflies,
            data_history=self.data_history,
            record_fn=self.record,
            data_length=const.data_length
        )
        self.ekf_logging.run()
        time.sleep(1.0)
        self.kill_sleep = kill_sleep
        print("6) All ready!")
        time.sleep(3.0)
        self.start_time = time.time()
        self.Control_Flag = True
        self.bridge_comm = CFlib_class(server=self)
        self.bridge_comm.run()

    def initialization(self):
        for cf in self.cf_list:
            self.mover[cf] = np.zeros(const.m)
            self.mover[cf][-1] = 1.0  ## scaler quaternion
            self.mover_history[cf] = np.zeros((const.data_length, const.m))
            self.hover[cf] = np.zeros(3)
            self.hover_flag[cf] = True
            self.data_history[cf] = np.zeros((const.data_length, 13))
            self.error_history[cf] = np.zeros((const.data_length, 13))

    def record(self, source, name, x, y, z, p=0, r=0, q=0, w=1):
        t = time.time()
        t_elp = t - self.t0
        line = f"{source + name},{t_elp},{x},{y},{z},{p},{r},{q},{w}"
        self.lock.acquire()
        self.log_data += (line + "\n")
        self.curr_data[source + name] = [t_elp, x, y, z, p, r, q, w]
        self.lock.release()

    @threaded
    def killswitch(self):
        start_time = time.time()
        while self.is_open:
            time.sleep(1.0)
            if time.time() - start_time > self.kill_sleep:
                break
        if self.is_open:
            print(f"Shutting down crazyflies due to kill switch timeout")
            for cf in self.crazyflies:
                cf.cf.high_level_commander.stop()

    def close(self):
        if self.save_flag == True:
            print("7) Cleaning Up")
            self.is_open = False
            self.ekf_logging.stop_logging()  ## close the ekf logger
            self.vicon_logger.stop_logging()  ## close the vicon logging
            some_alive = True
            while some_alive:
                some_alive = False
                for thread in self.threads:
                    some_alive = some_alive or thread.is_alive()
            path = f"CF_demo_run_{round(time.time())}.csv";
            file = open(path, "w")
            file.write(self.log_data)
            file.close()
            print(f"Saved log data to: {path}")
            for cf in self.crazyflies:
                print(f"Shutting down crazyflie {cf.object_name}")
                cf.cf.high_level_commander.stop()
                cf.cf.close_link()
