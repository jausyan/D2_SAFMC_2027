# SAFMC_2026
Safmc 2026 control program

```
mission.py          mission sequence only
utils/utils_.py     helpers + DroneNode (ROS 2 / MAVROS / GICP / /mission/order)
config/config.yaml  mission parameters, gate model, poses, waypoints
config/mavros_param.yaml  MAVROS parameters
data/               gate point clouds
scripts/            anti_point.py, capture_gate.py
setup/              SSH helper scripts
```

Run: `python3 mission.py`
