import rclpy
import threading
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import PoseStamped
import numpy as np
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from example_interfaces.msg import String as RosString
from .live_plotting import live_plotting_class
from .server import CrazyflieServer, TrackingObject

class crazyfliebridge(Node):
    def __init__(self):
        super().__init__('crazyflie_bridge')
        # Initialize Crazyflie server FIRST
        crazyflies = [
            TrackingObject("radio://0/80/2M/E7E7E7E7E7", "crazyflie7"),
            # TrackingObject("radio://0/80/2M/E7E7E7E7E7", "crazyflie8"),
            # TrackingObject( "radio://0/80/2M/E7E7E7E7E9", "crazyflie9" ),
        ]
        self.cf_list = []  ## name list
        for cf in crazyflies:
            self.cf_list.append(cf.object_name)
        vicon_ip = "192.168.0.39"
        self.server = CrazyflieServer(crazyflies, vicon_ip, 300)
        self.plotting = live_plotting_class(self.cf_list)
        self.is_active = True
        print("bridge activated")
        self.pubs = {}  ## publish the states
        self.subs = {}  ## subscribe the cmd
        for name in self.cf_list:
            print("pub")
            ## data type, topic name, data length
            self.pubs[name] = self.create_publisher(Odometry,
                                                    name + "/odom",
                                                    10)

            self.subs[name] = self.create_subscription(PoseStamped,
                                                     name + "/cmd_pos",
                                                     self.subscriber,
                                                     1)
            print("sub")

        self.count = 0.0
        dt = 0.01
        self.timer = self.create_timer(dt, self.publisher)

    def publisher(self):
        self.count += 1
        current_data = self.server.curr_data.copy()
        for cf in self.server.crazyflies:
            vicon_key = "vicon-" + cf.object_name
            if vicon_key in current_data:
                odom_msg = Odometry()
                odom_msg.header.stamp = self.get_clock().now().to_msg()
                odom_msg.header.frame_id = cf.object_name
                pos_data = current_data[vicon_key]
                odom_msg.pose.pose.position.x = float(pos_data[1])
                odom_msg.pose.pose.position.y = float(pos_data[2])
                odom_msg.pose.pose.position.z = float(pos_data[3])
                odom_msg.pose.pose.orientation.x = float(pos_data[4])
                odom_msg.pose.pose.orientation.y = float(pos_data[5])
                odom_msg.pose.pose.orientation.z = float(pos_data[6])
                odom_msg.pose.pose.orientation.w = float(pos_data[7])

                ## vel data
                vel_i = self.server.data_history[cf.object_name][-1,7:13] ## actual vel + vel cmd
                self.plotting.curr_state[cf.object_name] = pos_data[1:8]
                self.plotting.curr_vel_data[cf.object_name] = vel_i

                self.plotting.history_updt() ## update the data to history
                self.pubs[cf.object_name].publish(odom_msg)
                if self.count % 10 == 0:
                    self.get_logger().info(f"Published odom for {cf.object_name}, (x,y,z,q) = ({pos_data})")

    def subscriber(self, msg):
        cf = msg.header.frame_id
        ## position
        self.server.mover[cf][0] = msg.pose.position.x
        self.server.mover[cf][1] = msg.pose.position.y
        self.server.mover[cf][2] = msg.pose.position.z
        ## quaternion
        self.server.mover[cf][3] = msg.pose.orientation.x
        self.server.mover[cf][4] = msg.pose.orientation.y
        self.server.mover[cf][5] = msg.pose.orientation.z
        self.server.mover[cf][6] = msg.pose.orientation.w

        if self.count % 10 == 0:
            self.get_logger().info(f"Recieved cmd for {cf}, (x,y,z,q) = ({self.server.mover[cf]})")


def main(args=None):
    rclpy.init(args=args)
    bridge = crazyfliebridge()
    while bridge.is_active: rclpy.spin(bridge)


if __name__ == '__main__':
    main()
