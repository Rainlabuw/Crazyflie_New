import numpy as np
import rclpy
from threading import Thread, Event
import time
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from example_interfaces.msg import String as RosString
from .client import client_main


def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread

    return wrapper



class client_node(Node):
    def __init__(self):
        super().__init__('client')
        time.sleep(0.2)
        self.cf_name = "crazyflie"
        self.johnny_name = "Johnny"

        self.pub_count = 0
        self.sub_count = 0
        self.get_all_topic()


    @threaded
    def get_all_topic(self):
        time.sleep(1.0)
        all_topics = self.get_topic_names_and_types()
        odom_topics = [name for name, types in all_topics
                       if name.endswith('/odom')
                       and 'nav_msgs/msg/Odometry' in types]
        ## create name list of quads
        self.names = [t.lstrip('/').rsplit('/', 1)[0] for t in odom_topics]

        self.ini(odom_topics)

        while True:
            time.sleep(1.0)
            all_topics = self.get_topic_names_and_types()
            odom_topics = [name for name, types in all_topics
                           if name.endswith('/odom')
                           and 'nav_msgs/msg/Odometry' in types]
            ## create name list of quads
            self.names_new = [t.lstrip('/').rsplit('/', 1)[0] for t in odom_topics]
            ## reset the topics
            if self.names_new != self.names:
                print("Agent changed")
                self.names = self.names_new

                self.ini(odom_topics)
                print("Detected crazyflie: ", self.cf_list, "Detected Johnny: ", self.johnny_list)






    def ini(self,odom_topics):
        self.names = [t.lstrip('/').rsplit('/', 1)[0] for t in odom_topics]
        self.cf_list = []
        self.johnny_list = []
        for name in self.names:
            if self.cf_name in name:
                self.cf_list.append(name)
            elif self.johnny_name in name:
                self.johnny_list.append(name)
        self.Client = client_main(self.cf_list)
        self.cf_subs = {}
        self.johnny_subs = {}
        self.cf_pubs = {}
        self.johnny_pubs = {}
        ## for crazyflie
        for cf in self.cf_list:
            self.cf_subs[cf] = self.create_subscription(Odometry,
                                                        cf + "/odom",
                                                        self.subscriber, 10)
            # print(cf)
            self.cf_pubs[cf] = self.create_publisher(PoseStamped,
                                                     cf + "/cmd_pos", 1)

        ## for johnny
        for johnny in self.johnny_list:
            self.johnny_subs[johnny] = self.create_subscription(Odometry,
                                                                johnny + "/odom",
                                                                self.subscriber, 10)
        #     self.johnny_pubs[johnny] = self.create_publisher(PoseStamped,
        #                                                      johnny + "/cmd_pos", 1)

        self.dt = 0.01
        self.timer = self.create_timer(self.dt, self.publisher)


    def subscriber(self, msg):
        agent = msg.header.frame_id
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        if self.cf_name in agent:
            self.Client.state[agent] = np.array([pos.x, pos.y, pos.z])
            self.sub_count += 1
            # if self.sub_count % 10 == 0:
            #     self.get_logger().info(f"Recieved cmd for {agent}, (x,y,z,q) = ({pos, q})")
        elif self.johnny_name in agent:
            self.Client.johnny_data[agent] = np.array([pos.x,pos.y,pos.z])
            # if self.sub_count % 10 == 0:
            #     self.get_logger().info(f"Recieved pos for {agent}, (x,y,z,q) = ({pos})")



    def publisher(self):

        for agent in self.names:
            if self.cf_name in agent:
                msg = PoseStamped()
                ## set the time stamp for the frames
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = agent
                ## set the position
                # print(self.Client.cf_mover)
                msg.pose.position.x = self.Client.cf_mover[agent][0]
                msg.pose.position.y = self.Client.cf_mover[agent][1]
                msg.pose.position.z = self.Client.cf_mover[agent][2]
                ## set the attitude
                msg.pose.orientation.x = float(0.0)
                msg.pose.orientation.y = float(0.0)
                msg.pose.orientation.z = float(0.0)
                msg.pose.orientation.w = float(1.0)
                self.cf_pubs[agent].publish(msg)
                # if self.pub_count % 10 == 0:
                #     self.get_logger().info(
                #         f"Pos cmd for {agent}, (x,y,z) = ({msg.pose.position.x,msg.pose.position.y,msg.pose.position.z})")
                self.pub_count += 1




def main(args=None):
    rclpy.init(args=args)
    node = client_node()
    rclpy.spin(node)
    # node.destroy_node()
    # rclpy.shutdown()


if __name__ == '__main__':
    main()
