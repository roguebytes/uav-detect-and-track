"""Start the simulation stack: Gazebo (our world), the x500 with the nadir survey camera,
PX4 SITL attached to that model, MAVROS, the ros_gz bridges for the camera, clock and the ground-truth model odometry (/uav/gz_odom).

    ros2 launch uav_dt_ros sim.launch.py world:=bowl_field_sparse headless:=true

PX4 runs in standalone mode and attaches to the model we spawn by name, so nothing in the PX4
tree is modified. Requires `source scripts/env.sh` (sets PX4_DIR and GZ_SIM_RESOURCE_PATH).
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler, TimerAction, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def repo_root() -> str:
    # ros2_ws/install/uav_dt_ros/share/uav_dt_ros -> repo; fall back to the env var set by scripts/env.sh
    env = os.environ.get("UAV_DT_REPO")
    if env:
        return env
    here = os.path.abspath(__file__)
    for _ in range(8):
        here = os.path.dirname(here)
        if os.path.isdir(os.path.join(here, "sim", "worlds")):
            return here
    raise RuntimeError("cannot find the repo root; source scripts/env.sh")


def setup(context, *args, **kwargs):
    root = repo_root()
    px4_dir = os.environ.get("PX4_DIR", os.path.expanduser("~/PX4-Autopilot"))
    world = LaunchConfiguration("world").perform(context)
    headless = LaunchConfiguration("headless").perform(context).lower() == "true"
    model = LaunchConfiguration("model").perform(context)
    spawn_z = LaunchConfiguration("spawn_z").perform(context)

    world_file = os.path.join(root, "sim", "worlds", world + ".sdf")
    model_file = os.path.join(root, "sim", "models", model, "model.sdf")
    camera_hz = LaunchConfiguration("camera_hz").perform(context)
    if camera_hz:
        # spawn a copy of the model with the survey camera at this rate: at 5 m/s, 0.5 Hz still gives a
        # frame every 10 m under a 42 m footprint and halves the 12 MP render load on the GPU
        import re
        import tempfile
        src = open(model_file).read()
        src = re.sub(r"(<sensor name=\"survey_cam\".*?)<update_rate>[^<]*</update_rate>", rf"\1<update_rate>{camera_hz}</update_rate>", src, count=1, flags=re.S)
        tmp = tempfile.NamedTemporaryFile("w", suffix=".sdf", prefix=f"{model}_", delete=False)
        tmp.write(src)
        tmp.close()
        model_file = tmp.name
    px4_bin = os.path.join(px4_dir, "build", "px4_sitl_default", "bin", "px4")
    px4_rootfs = os.path.join(px4_dir, "build", "px4_sitl_default", "rootfs")
    px4_etc = os.path.join(px4_dir, "build", "px4_sitl_default", "etc")
    for f in (world_file, model_file, px4_bin):
        if not os.path.exists(f):
            raise RuntimeError(f"missing {f}")

    gz_args = ["gz", "sim", "-r", "-v", "1"]
    if headless:
        gz_args += ["-s", "--headless-rendering"]
    gz = ExecuteProcess(cmd=gz_args + [world_file], output="screen", name="gz_sim")

    spawn = Node(package="ros_gz_sim", executable="create", name="spawn_uav", output="screen",
                 arguments=["-world", world, "-file", model_file, "-name", model, "-z", spawn_z])
    # free-flying video camera, repositioned by follow_cam_node; lite variant when the UAV model is lite
    follow_model = "follow_cam_lite" if model.endswith("_lite") else "follow_cam"
    spawn_follow = Node(package="ros_gz_sim", executable="create", name="spawn_follow_cam", output="screen",
                        arguments=["-world", world, "-file", os.path.join(root, "sim", "models", follow_model, "model.sdf"),
                                   "-name", follow_model, "-x", "-10", "-z", "4"])
    follow = Node(package="uav_dt_ros", executable="follow_cam_node", name="follow_cam", output="screen",
                  parameters=[{"world": world, "entity": follow_model, "use_sim_time": True}],
                  additional_env={"UAV_DT_REPO": root})

    px4_env = dict(os.environ, PX4_GZ_STANDALONE="1", PX4_SIM_MODEL="gz_x500", PX4_GZ_MODEL_NAME=model,
                   PX4_GZ_WORLD=world, HEADLESS="1")
    px4 = ExecuteProcess(cmd=[px4_bin, "-d", px4_etc], cwd=px4_rootfs, env=px4_env, output="screen", name="px4")

    bridge = Node(package="ros_gz_bridge", executable="parameter_bridge", name="gz_bridge", output="screen",
                  arguments=[
                      "/uav/camera@sensor_msgs/msg/Image[gz.msgs.Image",
                      "/uav/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
                      "/follow_cam/image@sensor_msgs/msg/Image[gz.msgs.Image",
                      "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
                      f"/model/{model}/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry",
                  ],
                  remappings=[(f"/model/{model}/odometry", "/uav/gz_odom")])

    # No name override: mavros_node runs a router, a UAS node and one sub-node per plugin, and renaming
    # them all "mavros" makes plugin topics collide. Dict parameters apply to every node via /**.
    mavros = Node(package="mavros", executable="mavros_node", output="screen",
                  parameters=[{"fcu_url": "udp://:14540@127.0.0.1:14580", "gcs_url": "", "tgt_system": 1, "tgt_component": 1,
                               "fcu_protocol": "v2.0", "use_sim_time": True},   # stamps must match the camera's sim clock
                              os.path.join(root, "ros2_ws", "src", "uav_dt_ros", "config", "mavros_plugins.yaml")],
                  condition=IfCondition(LaunchConfiguration("mavros")))

    # order: gazebo, then spawn after 5 s, then PX4 once the spawn process exits, then bridge and MAVROS
    with_follow = LaunchConfiguration("follow_cam").perform(context).lower() == "true"
    after = [px4, bridge] + ([spawn_follow, follow] if with_follow else []) + [TimerAction(period=3.0, actions=[mavros])]
    after_spawn = RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=after))
    return [gz, TimerAction(period=5.0, actions=[spawn]), after_spawn]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("world", default_value="bowl_field_sparse", description="world name under sim/worlds"),
        DeclareLaunchArgument("model", default_value="x500_nadir_cam", description="model dir under sim/models"),
        DeclareLaunchArgument("headless", default_value="true", description="server only with software rendering"),
        DeclareLaunchArgument("mavros", default_value="true", description="start MAVROS"),
        DeclareLaunchArgument("spawn_z", default_value="0.3"),
        DeclareLaunchArgument("camera_hz", default_value="", description="override the survey camera's update rate, e.g. 0.5"),
        DeclareLaunchArgument("follow_cam", default_value="false",
                              description="spawn the live follow camera (third-person clips come from scripts/replay_follow.py instead)"),
        OpaqueFunction(function=setup),
    ])
