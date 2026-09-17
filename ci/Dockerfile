# Author: Frank Loewenich
# CI image for the headless smoke run: ROS 2 Humble, Gazebo Garden, ros_gz for Garden, PX4 v1.15.4
# SITL, MAVROS from the pinned snapshot. Built once and cached; the smoke run itself takes minutes.
FROM ros:humble-ros-base-jammy

ARG DEBIAN_FRONTEND=noninteractive
ARG PX4_TAG=v1.15.4

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl gnupg lsb-release git cmake build-essential python3-pip python3-venv ffmpeg \
      ros-humble-cv-bridge ros-humble-vision-msgs ros-humble-image-transport ros-humble-nav-msgs \
      python3-colcon-common-extensions \
 && curl -sSL https://packages.osrfoundation.org/gazebo.gpg -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable jammy main" > /etc/apt/sources.list.d/gazebo-stable.list \
 && apt-get update && apt-get install -y --no-install-recommends gz-garden libgz-sim7-dev \
 && rm -rf /var/lib/apt/lists/*

# ros_gz for Garden
RUN mkdir -p /ws/src && git clone -q -b humble https://github.com/gazebosim/ros_gz.git /ws/src/ros_gz \
 && bash -c "source /opt/ros/humble/setup.bash && cd /ws && GZ_VERSION=garden colcon build --merge-install --cmake-args -DBUILD_TESTING=OFF" \
 && rm -rf /ws/build /ws/log

# PX4 SITL
RUN git clone -q --recursive -b ${PX4_TAG} https://github.com/PX4/PX4-Autopilot.git /px4 \
 && bash /px4/Tools/setup/ubuntu.sh --no-nuttx --no-sim-tools \
 && cd /px4 && make px4_sitl && rm -rf /px4/.git

# MAVROS from the pinned snapshot (see scripts/install_mavros.sh)
COPY scripts/install_mavros.sh /tmp/install_mavros.sh
RUN bash /tmp/install_mavros.sh && rm -rf /var/lib/apt/lists/*

# repo
WORKDIR /repo
COPY . /repo
RUN python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt \
 && .venv/bin/pip install -q --index-url https://download.pytorch.org/whl/cpu torch torchvision \
 && python3 tools/make_grass_texture.py --size 2048 \
 && bash -c "source scripts/env.sh && scripts/build_tools.sh && cd ros2_ws && colcon build --symlink-install"

ENV PX4_DIR=/px4 ROS_GZ_WS=/ws
CMD ["bash", "-lc", "source scripts/env.sh && MAX_VERIFY=1 scripts/smoke.sh"]
