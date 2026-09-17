"""ament_python packaging for uav_dt_ros: nodes, launch files and configuration."""

__author__ = "Frank Loewenich"
import os
from glob import glob

from setuptools import setup

package_name = "uav_dt_ros"

setup(
    name=package_name,
    version="0.2.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Frank Loewenich",
    maintainer_email="frank.loewenich@gmail.com",
    description="Launch files and nodes for the UAV survey-and-verify bowl detection demo.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "perception_node = uav_dt_ros.perception_node:main",
            "mission_node = uav_dt_ros.mission_node:main",
            "follow_cam_node = uav_dt_ros.follow_cam_node:main",
        ],
    },
)
