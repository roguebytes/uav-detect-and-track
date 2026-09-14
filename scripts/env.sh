#!/usr/bin/env bash
# Source this before launching anything: `source scripts/env.sh`
# Sets up ROS 2 Humble, the Garden build of ros_gz, PX4 and this repo's Gazebo assets.
# Pinned versions are listed in docs/setup.md.

_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source /opt/ros/humble/setup.bash
# ros_gz built for Gazebo Garden. Must be sourced after Humble so it shadows the apt Fortress bridge.
ROS_GZ_WS="${ROS_GZ_WS:-$HOME/ws}"
[ -f "$ROS_GZ_WS/install/setup.bash" ] && source "$ROS_GZ_WS/install/setup.bash"
# This repo's ROS 2 package, once built.
[ -f "$_repo/ros2_ws/install/setup.bash" ] && source "$_repo/ros2_ws/install/setup.bash"

export PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
export GZ_VERSION=garden
# Our models and worlds first, then PX4's (x500 base and friends).
export GZ_SIM_RESOURCE_PATH="$_repo/sim/models:$_repo/sim/worlds:$PX4_DIR/Tools/simulation/gz/models:$PX4_DIR/Tools/simulation/gz/worlds"
# PX4 1.15 looks for PX4_GZ_WORLD in this directory.
export PX4_GZ_WORLDS="$_repo/sim/worlds"

# Python venv with torch and ultralytics (created by scripts/setup.sh). ROS launches nodes with the
# system interpreter, so mission.launch.py runs the perception node with UAV_DT_PYTHON instead.
[ -f "$_repo/.venv/bin/activate" ] && source "$_repo/.venv/bin/activate"
export UAV_DT_PYTHON="${UAV_DT_PYTHON:-$_repo/.venv/bin/python}"
export UAV_DT_REPO="$_repo"
export PYTHONPATH="$_repo${PYTHONPATH:+:$PYTHONPATH}"
unset _repo
