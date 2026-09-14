# Setup

Tested on the development machine: Ubuntu 22.04.5, kernel 6.8, RTX 2070 (driver 580), 32 GB.
`scripts/setup.sh` performs every step below and is safe to rerun. The pinned versions are the
ones the results in the README were produced with.

| Component | Version | Why this one |
|---|---|---|
| ROS 2 | Humble | The LTS that pairs with Ubuntu 22.04 |
| Gazebo | Garden (gz-sim 7.9) | What PX4 1.15 targets. End of life since September 2024; Harmonic is the upgrade path (PX4 1.16 and a `GZ_VERSION=harmonic` ros_gz build) |
| ros_gz | humble branch, built from source with `GZ_VERSION=garden` | The apt `ros-humble-ros-gz` packages are built for Fortress and must not shadow this build |
| PX4 | v1.15.4 SITL | Installed and flown before the project started; the control action sits behind an interface so an ArduPilot adapter can follow |
| MAVROS | 2.14.0 from the 2026-08-07 Humble snapshot | The live repository lost the MAVROS binaries when 2.15.1 failed to build on the farm (September 2026). `scripts/install_mavros.sh` pins only the MAVROS family to the snapshot |
| Python | 3.10 venv, torch cu126, ultralytics 8.4.150 | See `requirements.txt` |

## Steps

1. **Packages.** ROS 2 Humble desktop, cv_bridge, vision_msgs, gz-garden with dev headers, colcon, ffmpeg.
2. **MAVROS.** `sudo bash scripts/install_mavros.sh`. It fetches the ROS snapshot signing key, adds the snapshot as an apt source pinned to the five MAVROS packages, upgrades diagnostic_updater (MAVROS links its shared library) and installs the GeographicLib datasets.
3. **ros_gz for Garden.** Clone the `humble` branch into `~/ws/src` and build with `GZ_VERSION=garden`.
4. **PX4.** Clone v1.15.4 with submodules, run its `Tools/setup/ubuntu.sh --no-nuttx --no-sim-tools`, then `make px4_sitl`. Nothing in the PX4 tree is modified: the launch file starts Gazebo itself, spawns this repo's camera model and attaches PX4 in standalone mode.
5. **Venv.** `python3 -m venv .venv`, `requirements.txt`, then the torch build matching the driver.
6. **Generated assets.** The 4096 px grass texture (about 20 MB, not committed) and the two worlds. The tiny gz-transport helper in `tools/gz_set_pose` is built into `build/`.
7. **ROS package.** `cd ros2_ws && colcon build --symlink-install`.
8. **Weights** (optional, needed for the real detector). See `models/README.md`.

## Environment

`source scripts/env.sh` before any launch. It sources Humble, then the Garden ros_gz workspace, then this repo's package; sets `GZ_SIM_RESOURCE_PATH` to `sim/models` and `sim/worlds` ahead of PX4's models; and exports `UAV_DT_PYTHON` so the perception node runs in the venv while the rest of ROS uses the system interpreter.

## Checks

```
source scripts/env.sh
python -m pytest              # 23 unit tests, numpy only
python smoke_test.py          # offline pipeline on a synthetic survey
scripts/smoke.sh              # headless sim, oracle detector, about 3 to 5 minutes
```

## Known machine issues

- Gazebo cameras render only while a subscriber exists. `ros2 topic hz` on a 12 MP topic can take longer than its window to print; use `scripts/grab_frames.py`.
- A stale `px4` process makes the next launch fail with "PX4 server already running". `scripts/smoke.sh` kills leftovers first.
- Without an NVIDIA driver, Gazebo renders on Mesa llvmpipe: the lite camera model works, the full 12 MP camera at 1 Hz does not. Installing the Ubuntu 535 HWE modules on kernel 6.8.0-138 transitions to driver 580 and removes the 535 user space; install `nvidia-driver-580` to match, and expect a black screen until the reboot.
