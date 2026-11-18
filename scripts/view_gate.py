"""View a gate model (.npy, N x 3). Optionally export a .pcd for pcl_viewer.

    python3 scripts/view_gate.py data/gate_sim.npy            # matplotlib window
    python3 scripts/view_gate.py data/gate_sim.npy --pcd      # also writes data/gate_sim.pcd
    pcl_viewer data/gate_sim.pcd
"""
import argparse
from pathlib import Path

import numpy as np


def write_pcd(path, pts):
    header = ("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\n"
              f"TYPE F F F\nCOUNT 1 1 1\nWIDTH {len(pts)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
              f"POINTS {len(pts)}\nDATA ascii\n")
    with open(path, "w") as f:
        f.write(header)
        np.savetxt(f, pts, fmt="%.5f")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npy")
    ap.add_argument("--pcd", action="store_true", help="also write a .pcd next to the .npy")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    pts = np.load(args.npy)
    print(f"{pts.shape} {pts.dtype}  min {pts.min(0)}  max {pts.max(0)}  centre {(pts.min(0) + pts.max(0)) / 2}")

    if args.pcd:
        out = Path(args.npy).with_suffix(".pcd")
        write_pcd(out, pts)
        print(f"wrote {out}")
    if args.no_show:
        return

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (a, b, name) in zip(axes, [(1, 2, "front (y-z)"), (0, 1, "top (x-y)"), (0, 2, "side (x-z)")]):
        ax.scatter(pts[:, a], pts[:, b], s=1, c=pts[:, 2], cmap="viridis")
        ax.set_xlabel("xyz"[a]); ax.set_ylabel("xyz"[b]); ax.set_title(name)
        ax.set_aspect("equal"); ax.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
