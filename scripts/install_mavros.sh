#!/usr/bin/env bash
# Install MAVROS 2 for ROS 2 Humble on Ubuntu 22.04. Run with sudo.
#
# Why a snapshot: the MAVROS 2.15.1 binary build failed on the ROS build farm in September 2026
# (libmavconn failure), and the live Humble index no longer lists ros-humble-mavros. The dated
# snapshot below still carries MAVROS 2.14.0. The pin below applies only to the MAVROS family, so
# every other package keeps coming from the live repository. Remove the two files it writes once
# the live repository has MAVROS again.
set -euo pipefail
SNAP="${MAVROS_SNAPSHOT:-2026-08-07}"
# The snapshot repository is signed by the "ROS Snapshot builder" key (AD19BAB3CBF125EA), not the
# main ROS key. Fetch it from the Ubuntu keyserver into its own keyring.
KEY=/usr/share/keyrings/ros-snapshots-keyring.gpg
if [ ! -s "$KEY" ]; then
  tmp=$(mktemp -d); chmod 700 "$tmp"
  gpg --homedir "$tmp" --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys AD19BAB3CBF125EA
  gpg --homedir "$tmp" --export AD19BAB3CBF125EA > "$KEY"
  rm -rf "$tmp"
fi

cat > /etc/apt/sources.list.d/ros2-snapshot-mavros.list <<EOL
# MAVROS pinned to a ROS Humble snapshot (see scripts/install_mavros.sh in uav-detect-and-track)
deb [signed-by=$KEY] http://snapshots.ros.org/humble/$SNAP/ubuntu jammy main
EOL
cat > /etc/apt/preferences.d/ros2-snapshot-mavros <<'EOL'
Package: *
Pin: origin snapshots.ros.org
Pin-Priority: 100

Package: ros-humble-mavros ros-humble-mavros-extras ros-humble-libmavconn ros-humble-mavros-msgs ros-humble-mavlink
Pin: origin snapshots.ros.org
Pin-Priority: 1001
EOL

apt-get update
apt-get install -y --allow-downgrades ros-humble-mavros ros-humble-mavros-extras
bash /opt/ros/humble/lib/mavros/install_geographiclib_datasets.sh
echo "MAVROS installed:"; dpkg -l | grep -E '^ii +ros-humble-(mavros|libmavconn|mavlink)' | awk '{print "  " $2, $3}'
