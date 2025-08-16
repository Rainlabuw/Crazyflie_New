import numpy as np
import rclpy
import threading
import time
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from example_interfaces.msg import String as RosString
from .client import client_main

class client_node(Node):
    def __init__(self):
        super().__init__('client')

        all_topics = self.get_topic_names_and_types()
        odom_topics = [name for name, types in all_topics
                       if name.endswith('/odom')
                       and 'nav_msgs/msg/Odometry' in types]
        ## create name list of quads
        # self.names = [t.lstrip('/').rsplit('/', 1)[0] for t in odom_topics]
        self.names = ["crazyflie7"]
        self.Client = client_main(self.names)
        self.pub_count = 0
        self.sub_count = 0
        self.subs = {}
        self.pubs = {}

        for cf in self.names:
            self.subs[cf] = self.create_subscription(Odometry,
                                                     cf + "/odom",
                                                     self.subscriber, 10)
            self.pubs[cf] = self.create_publisher(PoseStamped,
                                                  cf + "/cmd_pos", 1)
        self.dt = 0.01
        self.timer = self.create_timer(self.dt, self.publisher)

    def subscriber(self, msg):
        cf = msg.header.frame_id
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.Client.state[cf] = np.array([pos.x,pos.y,pos.z])
        self.sub_count += 1
        # if self.sub_count % 10 == 0:
        #     self.get_logger().info(f"Recieved cmd for {cf}, (x,y,z,q) = ({pos,q})")

    def publisher(self):

        for cf in self.names:
            msg = PoseStamped()
            ## set the time stamp for the frames
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = cf
            ## set the position
            # print(self.Client.cf_mover)
            msg.pose.position.x = self.Client.cf_mover[cf][0]
            msg.pose.position.y = self.Client.cf_mover[cf][1]
            msg.pose.position.z = self.Client.cf_mover[cf][2]
            ## set the attitude
            msg.pose.orientation.x = float(0.0)
            msg.pose.orientation.y = float(0.0)
            msg.pose.orientation.z = float(0.0)
            msg.pose.orientation.w = float(1.0)
            self.pubs[cf].publish(msg)
            # if self.pub_count % 10 == 0:
            #     self.get_logger().info(
            #         f"Pos cmd for {cf}, (x,y,z) = ({msg.pose.position.x,msg.pose.position.y,msg.pose.position.z})")
        self.pub_count += 1




def main(args=None):
    rclpy.init(args=args)
    node = client_node()
    rclpy.spin(node)
    # node.destroy_node()
    # rclpy.shutdown()


if __name__ == '__main__':
    main()
