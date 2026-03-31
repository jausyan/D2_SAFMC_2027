import rclpy
from rclpy.node import Node
from scipy.interpolate import splprep, splev
from geometry_msgs.msg import PoseStamped, TwistStamped, Twist, PointStamped
from mavros_msgs.srv import CommandBool, SetMode
from mavros_msgs.msg import State
from nav_msgs.msg import Path
from livox_ros_driver2.msg import CustomMsg
from sensor_msgs.msg import PointCloud2, PointField, Imu
from utils import point_rotation_by_quarternion, factor
import time
import pygicp
import math
import numpy as np
import socket

class CobaCenteringNode(Node):
    def __init__(self, node_name='drone1', mavros_namespace='/mavros'):
        super().__init__(node_name)

        # Initialize UDP socket for communication
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)  # UDP
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.bind(("0.0.0.0", 5005))
        self.udp_buffer = np.array([], dtype=str)
        self.comm_timer = self.create_timer(0.5, self.update_udp_messages)
        rclpy.spin_once(self)

        self.mavros_namespace = mavros_namespace
        self.declare_parameter('input_array', [1.0, 2.0, 3.0, 4.0, 5.0])
        self.initial_guess = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ])

        self.zero_alt = 0.0
        self.current_velocity = None
        self.current_pose = None
        self.current_state = None
        self.current_imu = None
        self.start_gate_pcd_arr = None
        self.start_gate_center = None
        self.current_scan = None

        # ROS Publishers
        self.vel_pub = self.create_publisher(Twist, f'{self.mavros_namespace}/setpoint_velocity/cmd_vel_unstamped', 10)
        self.local_pos_pub = self.create_publisher(PoseStamped, f'{self.mavros_namespace}/setpoint_position/local', 10)
        # self.point_publisher = self.create_publisher(PointStamped, "/transformed_gate_center", 10)
        # self.pcd_publisher = self.create_publisher(PointCloud2, '/current_livox_scan', 10)
        # self.smooth_path_pub = self.create_publisher(Path, '/uav2/smooth_path', 10)

        # ROS Subscibers
        # self.vel_sub = self.create_subscription(TwistStamped, f'{self.mavros_namespace}/local_position/velocity_local', self.vel_callback, rclpy.qos.qos_profile_sensor_data)
        # self.imu_sub = self.create_subscription(Imu, f'{self.mavros_namespace}/imu/data', self.imu_callback, rclpy.qos.qos_profile_sensor_data)
        self.pose_sub = self.create_subscription(PoseStamped, f'{self.mavros_namespace}/local_position/pose', self.pose_callback, rclpy.qos.qos_profile_sensor_data)
        self.status_sub = self.create_subscription(State, f'{self.mavros_namespace}/state', self.status_callback, 10)
        self.livox_sub = self.create_subscription(CustomMsg, '/livox/lidar', self.livox_callback, 10)

        # ROS Clients
        self.setmode_client = self.create_client(SetMode, f'{self.mavros_namespace}/set_mode')
        self.arming_client = self.create_client(CommandBool, f'{self.mavros_namespace}/cmd/arming')

    
    def vel_callback(self, msg):
        self.current_velocity = msg
    
    def imu_callback(self, msg):
        self.current_imu = msg

    def pose_callback(self, msg):
        self.current_pose = msg

    def status_callback(self, msg):
        self.current_state = msg

    def livox_callback(self, msg):
        # if self.current_scan is None:
        self.current_scan = np.array([[point.x, point.y, point.z] for point in msg.points], dtype=np.float32)
        self.down_sampled_scan = pygicp.downsample(self.current_scan, 0.5)
        # self.down_sampled_scan = self.current_scan

        # self.current_scan = np.array([[point.x, point.y, point.z] for point in msg.points], dtype=np.float32)
        # pcd_msg = PointCloud2()
        # pcd_msg.width = self.current_scan.shape[0]
        # pcd_msg.height = 1
        # pcd_msg.fields = [
        #     PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        #     PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        #     PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        # ]
        # pcd_msg.is_bigendian = False
        # pcd_msg.point_step = 12
        # pcd_msg.row_step = self.current_scan.shape[0] * pcd_msg.point_step
        # pcd_msg.is_dense = True
        # pcd_msg.data = self.current_scan.tobytes()
        # pcd_msg.header.frame_id = "base_link"

        # self.pcd_publisher.publish(pcd_msg)


    def set_start_gate_pcd(self, npy_path, center=np.array([0.0, 0.0, 1.45])):
        self.get_logger().info(f"Loading start gate point cloud from: {npy_path}")
        self.start_gate_pcd_arr = np.load(npy_path)
        self.finish_gate_pcd_arr = self.start_gate_pcd_arr.copy()
        self.finish_gate_pcd_arr[:, 0] -= 5.0
        self.start_gate_pcd_arr = np.vstack((self.start_gate_pcd_arr, self.finish_gate_pcd_arr))
        self.start_gate_center = center
        print("Start gate point cloud loaded with shape:", self.start_gate_pcd_arr.shape)

    def takeoff(self, target_altitude=1.0):
        """
        Arms the drone and takes off to the specified altitude.
        """
        # Unsubscribe from livox
        self.destroy_subscription(self.livox_sub)
        self.livox_sub = None

        # Wait for FCU connection
        while rclpy.ok():
            rclpy.spin_once(self)
            if self.current_pose is not None:
                break
            self.get_logger().info("Waiting for FCU connection...")
            time.sleep(1)

        self.zero_alt = self.current_pose.pose.position.z

        # Takeoff to target altitude
        takeoff_pose = PoseStamped()
        takeoff_pose.pose.position.x = self.current_pose.pose.position.x
        takeoff_pose.pose.position.y = self.current_pose.pose.position.y
        takeoff_pose.pose.position.z = self.zero_alt + target_altitude
        takeoff_pose.pose.orientation = self.current_pose.pose.orientation

        for _ in range(100):
            self.local_pos_pub.publish(takeoff_pose)
            rclpy.spin_once(self)
            time.sleep(0.1)

        # Set to OFFBOARD mode
        while rclpy.ok():
            req = SetMode.Request()
            req.custom_mode = 'OFFBOARD'
            future = self.setmode_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            if future.result().mode_sent:
                self.get_logger().info("OFFBOARD enabled")
                break
            else:
                self.get_logger().info("Failed to set OFFBOARD mode, retrying...")
                time.sleep(1)

        # Arm the drone
        while rclpy.ok():
            req = CommandBool.Request()
            req.value = True
            future = self.arming_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            if future.result().success:
                self.get_logger().info("Vehicle armed")
                break
            else:
                self.get_logger().info("Failed to arm vehicle, retrying...")
                time.sleep(1)

        while rclpy.ok() and abs(self.current_pose.pose.position.z - takeoff_pose.pose.position.z) > 0.25:
            self.local_pos_pub.publish(takeoff_pose)
            rclpy.spin_once(self)
            time.sleep(0.1)

        self.get_logger().info(f"Takeoff to {target_altitude} meters complete.")
    
    def move_to(self, point, tolerance=0.1, hold_duration=0.0, method="position", max_speed=None):
        print("Moving to point:", point)
        if method == "position":
            target_pose = PoseStamped()
            target_pose.pose.position.x = point[0]
            target_pose.pose.position.y = point[1]
            target_pose.pose.position.z = self.zero_alt + point[2]
            print(f"target_z = {target_pose.pose.position.z}, current_z = {self.current_pose.pose.position.z}, zero_alt = {self.zero_alt}, point_z = {point[2]}")
            target_pose.pose.orientation = self.current_pose.pose.orientation
            start_time = None
            while rclpy.ok():
                self.local_pos_pub.publish(target_pose)
                rclpy.spin_once(self)
                if np.linalg.norm(np.array([self.current_pose.pose.position.x - point[0], self.current_pose.pose.position.y - point[1], self.current_pose.pose.position.z - self.zero_alt - point[2]])) > tolerance:
                    start_time = None
                else:
                    if start_time is None:
                        start_time = time.time()
                    elif time.time() - start_time >= hold_duration:
                        break
        elif method == "velocity":
            vel_cmd = Twist()
            start_time = None
            while rclpy.ok():
                direction = point - np.array([self.current_pose.pose.position.x,
                                            self.current_pose.pose.position.y,
                                            self.current_pose.pose.position.z - self.zero_alt])
                error = np.linalg.norm(direction)
                if error < tolerance:
                    if start_time is None:
                        start_time = time.time()
                    elif time.time() - start_time >= hold_duration:
                        break
                    vel_cmd.linear.x = 0.0
                    vel_cmd.linear.y = 0.0
                    vel_cmd.linear.z = 0.0
                else:
                    start_time = None
                    vel_cmd.linear.x = direction[0]
                    vel_cmd.linear.y = direction[1]
                    vel_cmd.linear.z = direction[2]
                    if max_speed is not None:
                        if error > max_speed:
                            vel_cmd.linear.x = vel_cmd.linear.x / error * max_speed
                            vel_cmd.linear.y = vel_cmd.linear.y / error * max_speed
                            vel_cmd.linear.z = vel_cmd.linear.z / error * max_speed
                self.vel_pub.publish(vel_cmd)
                rclpy.spin_once(self)

    def centering_gate(self, publish_scan=False):
        if self.start_gate_pcd_arr is None:
            self.get_logger().error("Start gate point cloud not set.")
            return None

        self.current_scan = None
        while self.current_scan is None and rclpy.ok():
            print("Waiting for current scan...")
            rclpy.spin_once(self)

        if publish_scan:
            print(self.current_scan.shape)
            pcd_msg = PointCloud2()
            pcd_msg.width = self.current_scan.shape[0]
            pcd_msg.height = 1
            pcd_msg.fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]
            pcd_msg.is_bigendian = False
            pcd_msg.point_step = 12
            pcd_msg.row_step = self.current_scan.shape[0] * pcd_msg.point_step
            pcd_msg.is_dense = True
            pcd_msg.data = self.current_scan.tobytes()
            pcd_msg.header.frame_id = "base_link"

            # self.pcd_publisher.publish(pcd_msg)

            # np.save("./gate_krti.npy", self.current_scan)

        matrix = pygicp.align_points(self.current_scan, self.start_gate_pcd_arr, method="GICP", initial_guess=self.initial_guess)

        gate_center_homogeneous = np.append(self.start_gate_center, 1.0)
        transformed_center = matrix @ gate_center_homogeneous

        return transformed_center[:3], matrix

    def land(self, descend_timeout=30.0):
        # Set to AUTO.LAND mode
        while rclpy.ok():
            req = SetMode.Request()
            req.custom_mode = 'AUTO.LAND'
            future = self.setmode_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            if future.result().mode_sent:
                self.get_logger().info("AUTO.LAND enabled")
                break
            else:
                self.get_logger().info("Failed to set AUTO.LAND mode, retrying...")
                time.sleep(1)

        return 0

    def _smooth_path(self, path, smoothing_factor=0.5):
        """
        Smooth path using cubic B-spline interpolation.
        
        Args:
            path: List of [x, y, z] waypoints
            smoothing_factor: Smoothing parameter (0=exact interpolation, higher=smoother)
        
        Returns:
            tck: B-spline representation
        """
        path = np.array(path)
        
        # Need at least 4 points for cubic spline
        if len(path) < 4:
            self.get_logger().warning("Path too short for cubic spline, using quadratic")
            k = min(2, len(path) - 1)
        else:
            k = 3
        
        # Fit B-spline to the path
        tck, u = splprep([path[:, 0], path[:, 1], path[:, 2]], 
                         s=smoothing_factor,
                         k=k)
        
        return np.hstack((np.array(splev(np.linspace(0, 1, 200), tck)).T, np.ones((200, 1))))
    
    def ramer_douglas_peucker(self ,path, epsilon=0.05):
        """
        Simplify a path using the Ramer-Douglas-Peucker algorithm.
        
        The algorithm recursively removes points that are within epsilon distance
        from the line segment connecting the endpoints, keeping only the points
        that are essential to maintain the path's shape.
        
        Args:
            path: numpy array of shape (n, 3) containing [x, y, z] coordinates
            epsilon: maximum perpendicular distance threshold for point removal
        
        Returns:
            simplified_path: numpy array of shape (m, 3) where m <= n
        """
        if len(path) < 3:
            return path
        
        # Find the point with maximum perpendicular distance from the line segment
        start = path[0]
        end = path[-1]
        
        # Vector from start to end
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)
        
        # Calculate perpendicular distances for all intermediate points
        dmax = 0.0
        index = 0
        
        if line_len > 1e-10:
            line_dir = line_vec / line_len
            
            for i in range(1, len(path) - 1):
                # Vector from start to current point
                point_vec = path[i] - start
                
                # Project onto line direction and clamp to segment
                t = np.clip(np.dot(point_vec, line_dir), 0, line_len)
                
                # Find closest point on line segment
                closest_point = start + t * line_dir
                
                # Calculate perpendicular distance
                d = np.linalg.norm(path[i] - closest_point)
                
                if d > dmax:
                    dmax = d
                    index = i
        else:
            # Degenerate case: start and end are the same
            for i in range(1, len(path) - 1):
                d = np.linalg.norm(path[i] - start)
                if d > dmax:
                    dmax = d
                    index = i
        
        # If max distance is greater than epsilon, recursively simplify
        if dmax > epsilon:
            # Recursive call on the two segments
            left_segment = self.ramer_douglas_peucker(path[:index + 1], epsilon)
            right_segment = self.ramer_douglas_peucker(path[index:], epsilon)
            
            # Combine results (remove duplicate point at index)
            result = np.vstack([left_segment[:-1], right_segment])
        else:
            # All points are within epsilon, keep only endpoints
            result = np.array([start, end])
        
        return result

    def follow_path(self, path, max_speed=10.0, step_tolerance=0.75):
        print(path)
        SLOW_DOWN_DISTANCE = 4.0
        for i, point in enumerate(path):
            while rclpy.ok():
                direction = point - np.array([self.current_pose.pose.position.x,
                                self.current_pose.pose.position.y,
                                self.current_pose.pose.position.z - self.zero_alt])
                # direction[2] = 0.0
                # direction = np.array([2.5, 0.0, 0.0])
                error = np.linalg.norm(direction)
                # print(point, np.array([self.current_pose.pose.position.x, self.current_pose.pose.position.y, self.current_pose.pose.position.z]))
                if error < step_tolerance:
                    break

                # neg_resultant = -np.sum(np.array([factor(p) * p for p in self.down_sampled_scan]), axis=0)
                # print("all: ", direction)
                # print("all: ", direction, neg_resultant, direction + neg_resultant)
                vel_cmd = Twist()
                vel_cmd.linear.y = direction[1]
                vel_cmd.linear.x = direction[0] / abs(15*vel_cmd.linear.y) if abs(vel_cmd.linear.y) >= 1.0 else direction[0]
                vel_cmd.linear.z = direction[2]
                # vel_cmd.linear.z = direction[2] * 0.5
                # if error > max_speed:
                #     vel_cmd.linear.x = vel_cmd.linear.x / error * max_speed
                #     vel_cmd.linear.y = vel_cmd.linear.y / error * max_speed
                #     vel_cmd.linear.z = vel_cmd.linear.z / error * max_speed

                if np.linalg.norm(path[-1] - point) < SLOW_DOWN_DISTANCE and i > path.shape[0] / 2:
                    speed_factor = np.linalg.norm(path[-1] - point) / SLOW_DOWN_DISTANCE
                    vel_cmd.linear.x *= speed_factor
                    vel_cmd.linear.y *= speed_factor
                    vel_cmd.linear.z *= speed_factor
                self.vel_pub.publish(vel_cmd)

                # pose_cmd = PoseStamped()
                # pose_cmd.pose.position.x = point[0]
                # pose_cmd.pose.position.y = point[1]
                # pose_cmd.pose.position.z = point[2]
                # self.local_pos_pub.publish(pose_cmd)

                rclpy.spin_once(self)
                direction = point - np.array([self.current_pose.pose.position.x,
                                            self.current_pose.pose.position.y,
                                            self.current_pose.pose.position.z])
        self.move_to(point, tolerance=0.5)
    
    def get_yaw_from_orientation(self, orientation):
        """Extract yaw angle from quaternion orientation."""
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    def spin_drone(self, duration, angle_step=10.0, step_duration=0.5):
        """Spin the drone because its funny."""
        # Wait for FCU connection
        while rclpy.ok():
            rclpy.spin_once(self)
            if self.current_pose is not None:
                break
            self.get_logger().info("Waiting for FCU connection...")
            time.sleep(1)

        self.zero_alt = self.current_pose.pose.position.z

        for _ in range(100):
            self.local_pos_pub.publish(self.current_pose)
            rclpy.spin_once(self)
            time.sleep(0.1)

        # Set to OFFBOARD mode
        while rclpy.ok():
            req = SetMode.Request()
            req.custom_mode = 'OFFBOARD'
            future = self.setmode_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            if future.result().mode_sent:
                self.get_logger().info("OFFBOARD enabled")
                break
            else:
                self.get_logger().info("Failed to set OFFBOARD mode, retrying...")
                time.sleep(1)

        # Arm the drone
        while rclpy.ok():
            req = CommandBool.Request()
            req.value = True
            future = self.arming_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            if future.result().success:
                self.get_logger().info("Vehicle armed")
                break
            else:
                self.get_logger().info("Failed to arm vehicle, retrying...")
                time.sleep(1)

        current_yaw = self.get_yaw_from_orientation(self.current_pose.pose.orientation)
        start_time = time.time()
        last_step_time = time.time()
        while rclpy.ok() and (time.time() - start_time) < duration:
            if (time.time() - last_step_time) >= step_duration:
                last_step_time = time.time()
                current_yaw += math.radians(angle_step)
            target_pose = PoseStamped()
            target_pose.pose.position.x = self.current_pose.pose.position.x
            target_pose.pose.position.y = self.current_pose.pose.position.y
            target_pose.pose.position.z = self.current_pose.pose.position.z
            # Convert yaw back to quaternion
            qz = math.sin(current_yaw / 2.0)
            qw = math.cos(current_yaw / 2.0)
            target_pose.pose.orientation.w = qw
            target_pose.pose.orientation.x = 0.0
            target_pose.pose.orientation.y = 0.0
            target_pose.pose.orientation.z = qz

            self.local_pos_pub.publish(target_pose)
            rclpy.spin_once(self)

    def broadcast_udp(self, data, port=5005):
        # self.sock.sendto(bytes(data, "utf-8"), ("192.168.50.255", port))
        # print("UDAH BERHASIL WEEE, -RIAN")
        self.udp_buffer = np.append(self.udp_buffer, data)
        self.sock.sendto(bytes(data, "utf-8"), ("192.168.50.255", port))
        self.get_logger().info(f"Broadcasted UDP message: {data}, current buffer: {self.udp_buffer}")
    
    def update_udp_messages(self):
        self.sock.settimeout(0.05)
        if len(self.udp_buffer) > 0:
            current_state_str = ",".join(self.udp_buffer)
            self.sock.sendto(bytes(current_state_str, "utf-8"), ("192.168.50.255", 5005))
        try:
            while True:
                data, addr = self.sock.recvfrom(1024)  # buffer size is 1024 bytes
                # self.udp_buffer = np.append(self.udp_buffer, data.decode("utf-8"))
                incoming = np.unique(np.array(data.decode("utf-8").split(",")))
                diff = np.setdiff1d(incoming, self.udp_buffer)
                if diff.size > 0:
                    self.udp_buffer = np.append(self.udp_buffer, diff)
                    self.get_logger().info(f"Received UDP message: {data.decode('utf-8')} from {addr}, newest buffer: {self.udp_buffer}")
        except socket.timeout:
            pass

    # def wait_for_udp_message(self, expected_data, port=5005, timeout=60.0):
    #     sock.settimeout(0.1)
    #     start_time = time.time()
    #     while time.time() - start_time < timeout:
    #         rclpy.spin_once(self)
    #         try:
    #             data, addr = sock.recvfrom(1024)  # buffer size is 1024 bytes
    #             if data.decode("utf-8") == expected_data:
    #                 self.get_logger().info(f"Received expected UDP message: {expected_data}")
    #                 break
    #         except socket.timeout:
    #             self.get_logger().warning("Still waiting for expected UDP message...")
    #             rclpy.spin_once(self)

    def wait_for_udp_message(self, expected_data, port=5005, timeout=60.0):
        start_time = time.time()
        while time.time() - start_time < timeout:
            rclpy.spin_once(self)
            if self.udp_buffer.size > 0:
                if expected_data in self.udp_buffer:
                    self.get_logger().info(f"Received expected UDP message: {expected_data}")
                    break
                else:
                    self.get_logger().warning(f"No expected data in the udp buffer, current buffer: {self.udp_buffer}")

if __name__ == '__main__':
    rclpy.init()
    node = CobaCenteringNode("rian_drone", "/rian")


    node.set_start_gate_pcd('/home/rian/project/slam_ws/start_gate_latihan.npy', center=np.array([2.5, 0.0, 1.3]))
    rclpy.spin_once(node)

    _ , mat = node.centering_gate(publish_scan=True)

    while node.current_pose is None:
        print("Waiting for current pose...")
        rclpy.spin_once(node)
    local_pose = node.current_pose
    center_area = mat @ np.array([0.0, 0.0, 1.3, 1.0])
    center_area = point_rotation_by_quarternion(center_area, [node.current_pose.pose.orientation.w, node.current_pose.pose.orientation.x, node.current_pose.pose.orientation.y, node.current_pose.pose.orientation.z])
    center_area[0] += node.current_pose.pose.position.x
    center_area[1] += node.current_pose.pose.position.y
    center_area[2] = 1.3
    comm_start_pose = mat @ np.array([2.5, 0.0, 1.3, 1.0])
    comm_start_pose = point_rotation_by_quarternion(comm_start_pose, [node.current_pose.pose.orientation.w, node.current_pose.pose.orientation.x, node.current_pose.pose.orientation.y, node.current_pose.pose.orientation.z])
    comm_start_pose[0] += node.current_pose.pose.position.x
    comm_start_pose[1] += node.current_pose.pose.position.y
    comm_start_pose[2] = 1.3
    land_area = mat @ np.array([0.0, 1.0, 1.3, 1.0])
    land_area = point_rotation_by_quarternion(land_area, [node.current_pose.pose.orientation.w, node.current_pose.pose.orientation.x, node.current_pose.pose.orientation.y, node.current_pose.pose.orientation.z])
    land_area[0] += node.current_pose.pose.position.x
    land_area[1] += node.current_pose.pose.position.y
    land_area[2] = 1.3
    
    point_msg = PointStamped()
    point_msg.header.stamp = node.get_clock().now().to_msg()
    point_msg.header.frame_id = "map"
    # node.point_publisher.publish(point_msg)

    # path = np.array([
    #     # [0.0, 0.0, 1.3, 1.0],
    #     [1.5, 0.0, 1.3, 1.0],
    #     [3.5, 0.0, 1.3, 1.0],
    #     [7.72, -3.22, 1.3, 1.0],
    #     [5.72, -5.22, 1.3, 1.0],
    #     [0.0, 0.0, 1.3, 1.0]
    # ])
    path = np.array([
        [2.5, 0.0, 1.3],
        [10.5, -0.95, 1.3],
        [14.5, -0.95, 1.3],
        [16.5, -0.95, 1.3],
        [19.0, -5.0, 1.3],
        [18.0, -7.5, 1.3],
        # [19.0, -9.05, 1.3],
        # [18.0, -9.05, 1.3],
        # [17.0, -9.05, 1.3],
        # [16.0, -9.05, 1.3],
        # [15.0, -9.05, 1.3],
        # [14.0, -9.05, 1.3],
        [16.0, -9.05, 1.3],
        [12.0, -9.05, 1.3],
        [10.0, -9.05, 1.3],
        [-2.875, -8.672, 1.3],
        [-4.25, -8.672, 1.3],
        [-4.375, -5.0, 1.3],
        # [-8.625, -10.58309519, 1.3],
        # [-12.0, 0.0, 1.3],
        # [-11.0, 0.0, 1.3],
        # [-10.0, 0.0, 1.3],
        # [-9.0, 0.0, 1.3],
        # [-8.0, 0.0, 1.3],
        # [-7.0, 0.0, 1.3],
        # [-6.0, 0.0, 1.3],
        # [-5.0, 0.0, 1.3],
        # [-4.0, 0.0, 1.3],
        # [-5.0, -3.0, 1.3],
        [-4.5, 0.0, 1.3],
        # [-2.5, 0.0, 1.3],
        # [0.0, 0.0, 1.3],
    ])

    smooth_path = node._smooth_path(path, smoothing_factor=0.0001)

    offset_smooth_path = np.zeros((200, 3))
    for i, point in enumerate(smooth_path):
        point = mat @ point
        point = point_rotation_by_quarternion(point, [local_pose.pose.orientation.w, local_pose.pose.orientation.x, local_pose.pose.orientation.y, local_pose.pose.orientation.z])
        point[0] += local_pose.pose.position.x
        point[1] += local_pose.pose.position.y
        point[2] += local_pose.pose.position.z
        offset_smooth_path[i] = point[:3]

    offset_smooth_path[:, 2] = 1.3
    simple = node.ramer_douglas_peucker(offset_smooth_path)
    simple[:, 2] = 1.3
    smooth_path_msg = Path()
    smooth_path_msg.header.frame_id = "map"
    for point in offset_smooth_path:
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "map"
        pose_stamped.pose.position.x = point[0]
        pose_stamped.pose.position.y = point[1]
        pose_stamped.pose.position.z = point[2]
        smooth_path_msg.poses.append(pose_stamped)
    # node.smooth_path_pub.publish(smooth_path_msg)

    print("taking off.....")
    # node.spin_drone(duration=5.0, angle_step=30.0, step_duration=0.075)
    # node.land()

    node.takeoff(1.3)
    node.broadcast_udp("1_rian_takeoff")
    node.wait_for_udp_message("3_rian_turn", port=5005, timeout=300.0)
    time.sleep(0.5)
    node.move_to(comm_start_pose[:3], method="position", max_speed=2.0, tolerance=1.0)
    node.broadcast_udp("4_verny_turn")
    node.follow_path(offset_smooth_path, step_tolerance=2.5)
    node.move_to(center_area[:3], method="position", max_speed=2.0, tolerance=0.75)
    node.move_to(land_area[:3], method="position", max_speed=1.5)
    node.land()

    # node.takeoff(1.3)
    # node.move_to(np.array([15.0, 0.0, 1.3]))
    # node.land()

    node.destroy_node()
    rclpy.shutdown()
