"""Record a gate model from the lidar and save it as .npy (N x 3, float32).

Place the drone at the nominal start pose (the frame the gate centre in
config/config.yaml is expressed in), facing the gate, then run:

    python3 scripts/capture_gate.py data/start_gate_new.npy --scans 10 --voxel 0.05 \
        --crop 0 6 -2.5 2.5 0 3
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pygicp
import rclpy
from rclpy.node import Node

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.utils_ import load_config, subscribe_lidar  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--scans", type=int, default=10, help="scans to accumulate")
    ap.add_argument("--voxel", type=float, default=0.05, help="voxel size (m), 0 = off")
    ap.add_argument("--crop", type=float, nargs=6, metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                    help="keep only points inside this box (sensor frame)")
    args = ap.parse_args()

    cfg = load_config()
    rclpy.init()
    node = Node("capture_gate")
    scans = []
    sub = subscribe_lidar(node, cfg["lidar"], scans.append)
    while rclpy.ok() and len(scans) < args.scans:
        rclpy.spin_once(node, timeout_sec=1.0)
        node.get_logger().info(f"{len(scans)}/{args.scans} scans")

    pts = np.vstack(scans).astype(np.float32)
    if args.crop:
        x0, x1, y0, y1, z0, z1 = args.crop
        m = ((pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1)
             & (pts[:, 2] > z0) & (pts[:, 2] < z1))
        pts = pts[m]
    if args.voxel > 0:
        pts = pygicp.downsample(pts, args.voxel)
    pts = np.asarray(pts, dtype=np.float32)
    np.save(args.out, pts)
    print(f"saved {pts.shape} to {args.out}")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
