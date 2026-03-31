import numpy as np
import rclpy
import pygicp
from livox_ros_driver2.msg import CustomMsg
from geometry_msgs.msg import TwistStamped

current_scan = None

def factor(p, start=np.array([0.0, 0.0, 0.0])):
    norm = np.linalg.norm(p - start)
    return np.exp(-4*norm)

def livox_callback(msg):
    global current_scan
    current_scan = pygicp.downsample(np.array([[point.x, point.y, point.z] for point in msg.points], dtype=np.float32), 0.25)

rclpy.init()
node = rclpy.create_node("anti_point")
livox_sub = node.create_subscription(CustomMsg, '/livox/lidar', livox_callback, 10)
pub = node.create_publisher(TwistStamped, 'anti_vel', 10)


while current_scan is None:
    print("waiting for scan")
    rclpy.spin_once(node)

while rclpy.ok():
    resultant = -np.sum(np.array([factor(p) * p for p in current_scan]), axis=0)
    print("resultant: ", resultant)
    vel_msg = TwistStamped()
    vel_msg.header.frame_id = "map"
    vel_msg.twist.linear.x = float(resultant[0])
    vel_msg.twist.linear.y = float(resultant[1])
    vel_msg.twist.linear.z = float(resultant[2])
    print("publishing....")
    pub.publish(vel_msg)
    rclpy.spin_once(node)
