"""Start the perception and mission nodes on top of sim.launch.py.

    ros2 launch uav_dt_ros mission.launch.py detector:=yolo pose_source:=mavros world:=bowl_field_sparse
    ros2 launch uav_dt_ros mission.launch.py detector:=gt pose_source:=gz model:=x500_nadir_cam_lite   # CI

Logs go to runs/<run_name>/perception.jsonl and mission.jsonl for scripts/score.py.
"""
import os
import time

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node


def setup(context, *a, **k):
    L = lambda n: context.launch_configurations[n]  # noqa: E731
    root = os.environ.get("UAV_DT_REPO") or os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 5)))
    run_dir = os.path.join(root, "runs", L("run_name") or time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    manifest = os.path.join(root, "sim", "worlds", L("world") + ".json")
    lite = L("model").endswith("_lite")
    perception_params = {"detector": L("detector"), "pose_source": L("pose_source"),
                         "weights": os.path.join(root, "models", "scratch_best.pt"),
                         "world_manifest": manifest, "log_path": os.path.join(run_dir, "perception.jsonl"),
                         "device": L("device"), "half": L("half") == "true", "conf": float(L("conf")),
                         "gt_miss_rate": float(L("gt_miss_rate")), "use_sim_time": True,
                         "frame_stride": int(L("frame_stride"))}
    venv_python = os.environ.get("UAV_DT_PYTHON")
    if venv_python and os.path.exists(venv_python):
        # torch and ultralytics live in the repo venv; ros2 entry points use the system interpreter
        from launch.actions import ExecuteProcess
        ros_args = ["--ros-args", "-r", "__node:=perception"]
        for k, v in perception_params.items():
            if v == "":
                continue                      # rcl cannot parse an empty override; the node default applies
            ros_args += ["-p", f"{k}:={str(v).lower() if isinstance(v, bool) else v}"]
        perception = ExecuteProcess(cmd=[venv_python, "-m", "uav_dt_ros.perception_node", *ros_args],
                                    output="screen", name="perception")
    else:
        perception = Node(package="uav_dt_ros", executable="perception_node", name="perception", output="screen",
                          parameters=[perception_params])
    mission = Node(package="uav_dt_ros", executable="mission_node", name="mission", output="screen",
                   parameters=[{"field_w": float(L("field_w")), "field_h": float(L("field_h")),
                                "survey_alt": float(L("survey_alt")), "verify_alt": float(L("verify_alt")),
                                "dwell_s": float(L("dwell_s")), "max_verify": int(L("max_verify")),
                                "image_w": 1008 if lite else 4032, "image_h": 756 if lite else 3024, "use_sim_time": True,
                                "log_path": os.path.join(run_dir, "mission.jsonl")}])
    actions = [perception, mission]
    if L("record") == "true":
        from launch.actions import ExecuteProcess
        actions.append(ExecuteProcess(cmd=["python3", os.path.join(root, "scripts", "record_video.py"), "--out-dir", run_dir,
                                           "--fps", L("record_fps"), "/follow_cam/image", "/perception/annotated"],
                                      output="screen", name="record_video"))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("world", default_value="bowl_field_sparse"),
        DeclareLaunchArgument("model", default_value="x500_nadir_cam"),
        DeclareLaunchArgument("detector", default_value="yolo", description="yolo | gt"),
        DeclareLaunchArgument("pose_source", default_value="mavros", description="mavros | gz"),
        DeclareLaunchArgument("device", default_value="", description="torch device, e.g. 0 or cpu"),
        DeclareLaunchArgument("half", default_value="true", description="FP16 inference on the GPU: halves GPU time and heat"),
        DeclareLaunchArgument("frame_stride", default_value="1", description="process every Nth camera frame"),
        DeclareLaunchArgument("conf", default_value="0.25"),
        DeclareLaunchArgument("gt_miss_rate", default_value="0.0", description="oracle detector per-frame miss probability"),
        DeclareLaunchArgument("field_w", default_value="120"),
        DeclareLaunchArgument("field_h", default_value="80"),
        DeclareLaunchArgument("survey_alt", default_value="40"),
        DeclareLaunchArgument("verify_alt", default_value="11"),
        DeclareLaunchArgument("dwell_s", default_value="4"),
        DeclareLaunchArgument("max_verify", default_value="0", description="0 = verify every confirmed track"),
        DeclareLaunchArgument("run_name", default_value=""),
        DeclareLaunchArgument("record", default_value="false", description="record follow-camera and annotated video to the run dir"),
        DeclareLaunchArgument("record_fps", default_value="15"),
        OpaqueFunction(function=setup),
    ])
