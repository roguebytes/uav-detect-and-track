# SITL bring-up — quad and fixed-wing

Run the detect-and-track node against a simulated camera in PX4 SITL (or ArduPilot SITL),
on **both** a multicopter and a fixed-wing airframe, before any real flight. Versions move
fast — treat these as the shape of the workflow and adjust to your installed releases.

## Prerequisites
- ROS 2 (Humble/Jazzy), `ros_gz_bridge`, `cv_bridge`
- PX4-Autopilot (with Gazebo) **or** ArduPilot SITL
- This repo's deps: `pip install -r ../requirements.txt`

## PX4 SITL + Gazebo

Multicopter with a camera (e.g. the `x500` + a camera-equipped world/model):
```bash
# Terminal 1 — PX4 SITL with a camera-equipped multicopter
make px4_sitl gz_x500_mono_cam        # model name varies by PX4 version

# Terminal 2 — bridge the Gazebo camera topic into ROS 2
ros2 run ros_gz_bridge parameter_bridge \
    /camera@sensor_msgs/msg/Image@gz.msgs.Image

# Terminal 3 — run the node on that topic
python ../ros2/detect_track_node.py --ros-args \
    -p image_topic:=/camera -p model:=yolo11n.pt
```

Fixed-wing — same node, just launch a plane model:
```bash
make px4_sitl gz_rc_cessna            # or another fixed-wing model with a camera
# (re-run the same bridge + node)
```

## ArduPilot SITL (alternative)
```bash
sim_vehicle.py -v ArduCopter -f gazebo-iris --console --map   # quad
sim_vehicle.py -v ArduPlane  -f gazebo-plane --console --map   # fixed-wing
# bridge the Gazebo camera to ROS 2 as above, then run the node.
```

## What to look for
- `~/target_offset` (PointStamped): normalised target offset, `z` = track id.
- `~/annotated` (Image): boxes, track IDs, centre crosshair, and the target vector.

This offset stream is the input to **Demo 2** (closed-loop orbit/loiter): map it to a
control action per airframe — quad yaw/reposition vs fixed-wing loiter — behind one interface.
See `../../demo_specs_autonomy.md`.
