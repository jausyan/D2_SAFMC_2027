import time
import numpy as np
import rclpy
from utils.utils_ import (DroneNode, gate_to_world, load_config, ramer_douglas_peucker, smooth_path)

def main():
    cfg = load_config()
    alt = cfg["flight"]["cruise_altitude"]
    ctl = cfg["controller"]
    rclpy.init()
    node = DroneNode(cfg)
    # --- Localise the gate and precompute everything in the map frame ------
    node.set_start_gate_pcd()
    rclpy.spin_once(node)
    _, gate_mat = node.centering_gate()

    while node.current_pose is None:
        node.get_logger().info("Waiting for current pose...")
        rclpy.spin_once(node)
    pose = node.current_pose

    to_world = lambda p: gate_to_world(p, gate_mat, pose, altitude=alt)
    center_area = to_world(cfg["poses"]["center_area"])
    comm_start = to_world(cfg["poses"]["comm_start"])
    land_area = to_world(cfg["poses"]["land_area"])

    spline = smooth_path(cfg["path"]["waypoints"], cfg["path"]["smoothing_factor"], cfg["path"]["num_samples"])
    world_path = np.array([gate_to_world(p, gate_mat, pose, altitude=alt) for p in spline])
    # RDP-simplified variant is computed for inspection/visualisation only
    simple_path = ramer_douglas_peucker(world_path)  # noqa: F841

    # --- Mission sequence ---------------------------------------------------
    node.wait_command("GO-1")
    node.takeoff()
    node.pub_command("GO-2")
    time.sleep(0.5)
    node.move_to(comm_start, tolerance=ctl["comm_start"]["tolerance"])
    node.follow_path(world_path)
    node.move_to(center_area, tolerance=ctl["move_to"]["tolerance"])
    node.move_to(land_area)
    node.land()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
