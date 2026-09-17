#!/usr/bin/env bash
# Author: Frank Loewenich
# One-shot setup on Ubuntu 22.04 for the simulation stack. Idempotent; rerun after a git pull.
# Steps that need root are printed and run with sudo; everything else runs as the user.
# See docs/setup.md for what each step does and the pinned versions.
set -eo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
PX4_TAG="${PX4_TAG:-v1.15.4}"
ROS_GZ_WS="${ROS_GZ_WS:-$HOME/ws}"

step() { echo; echo "==> $*"; }

step "apt: ROS 2 Humble, Gazebo Garden, build tools"
if ! grep -q packages.ros.org /etc/apt/sources.list.d/*.list 2>/dev/null; then
  sudo apt-get install -y curl gnupg lsb-release
  sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
fi
if ! grep -q packages.osrfoundation.org /etc/apt/sources.list.d/*.list 2>/dev/null; then
  sudo curl -sSL https://packages.osrfoundation.org/gazebo.gpg -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
fi
sudo apt-get update
sudo apt-get install -y ros-humble-desktop ros-humble-cv-bridge ros-humble-vision-msgs ros-humble-image-transport \
  gz-garden libgz-sim7-dev python3-colcon-common-extensions python3-venv python3-pip git cmake build-essential ffmpeg

step "MAVROS (pinned snapshot, see scripts/install_mavros.sh)"
dpkg -s ros-humble-mavros > /dev/null 2>&1 || sudo bash scripts/install_mavros.sh

step "ros_gz for Gazebo Garden from source in $ROS_GZ_WS"
if [ ! -f "$ROS_GZ_WS/install/setup.bash" ]; then
  mkdir -p "$ROS_GZ_WS/src"
  [ -d "$ROS_GZ_WS/src/ros_gz" ] || git clone -b humble https://github.com/gazebosim/ros_gz.git "$ROS_GZ_WS/src/ros_gz"
  (source /opt/ros/humble/setup.bash && cd "$ROS_GZ_WS" && GZ_VERSION=garden colcon build --symlink-install --cmake-args -DBUILD_TESTING=OFF)
fi

step "PX4 $PX4_TAG SITL in $PX4_DIR"
if [ ! -d "$PX4_DIR" ]; then
  git clone --recursive -b "$PX4_TAG" https://github.com/PX4/PX4-Autopilot.git "$PX4_DIR"
  bash "$PX4_DIR/Tools/setup/ubuntu.sh" --no-nuttx --no-sim-tools
fi
if [ ! -x "$PX4_DIR/build/px4_sitl_default/bin/px4" ]; then
  (cd "$PX4_DIR" && make px4_sitl)
fi

step "Python venv with the perception dependencies"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
if command -v nvidia-smi > /dev/null 2>&1; then
  .venv/bin/pip install -q --index-url https://download.pytorch.org/whl/cu126 torch torchvision
else
  .venv/bin/pip install -q --index-url https://download.pytorch.org/whl/cpu torch torchvision
fi

step "Generated assets and native helpers"
python3 tools/make_grass_texture.py
[ -f sim/worlds/bowl_field_sparse.sdf ] || python3 tools/make_world.py --name sparse --seed 1 --bowls 12
[ -f sim/worlds/bowl_field_dense.sdf ] || python3 tools/make_world.py --name dense --seed 2 --bowls 60
source scripts/env.sh
scripts/build_tools.sh

step "ROS 2 package"
(cd ros2_ws && colcon build --symlink-install)

step "Weights"
if [ -f models/scratch_best.pt ]; then
  (cd models && sha256sum -c scratch_best.pt.sha256 2>/dev/null) || echo "  models/scratch_best.pt present (checksum file missing, see models/README.md)"
else
  echo "  models/scratch_best.pt not found: the real detector needs it, see models/README.md. The smoke run does not."
fi

echo; echo "Setup complete. Next: source scripts/env.sh && scripts/smoke.sh"
