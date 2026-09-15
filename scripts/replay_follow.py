#!/usr/bin/env python3
"""Cinematic replay: re-render the follow camera along a logged trajectory with nothing else loading the GPU.

    source scripts/env.sh
    python3 scripts/replay_follow.py --run runs/real_sparse --world bowl_field_sparse [--fps 30] [--start 0 --end 120]

Starts a headless Gazebo server on the world, spawns a static visual-only x500 and the follow camera,
then steps both along the trajectory at the camera's frame rate through tools/gz_set_pose while
scripts/record_video.py records /follow_cam/image. Output: <run>/replay_follow.mp4.

During a mission the 12 MP survey camera stalls Gazebo's render thread for over 100 ms every
second, so the live follow-camera recording skips and repeats frames. Here the only sensor is the
follow camera, so the render keeps up and every frame is a new frame.
"""
import argparse
import math
import os
import subprocess
import sys
import time

import numpy as np


def quat_yaw(qx, qy, qz, qw):
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def follow_pose(x, y, z, yaw, d=7.0, h=3.0, pitch=math.radians(20.0)):
    cx, cy, cz = x - d * math.cos(yaw), y - d * math.sin(yaw), z + h
    cy2, sy2, cp2, sp2 = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2)
    return cx, cy, cz, (-sy2 * sp2, cy2 * sp2, sy2 * cp2, cy2 * cp2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--world", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--start", type=float, default=0.0, help="trajectory time to start at (s from first pose)")
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--distance", type=float, default=7.0)
    ap.add_argument("--height", type=float, default=3.0)
    ap.add_argument("--pitch-deg", type=float, default=20.0)
    ap.add_argument("--yaw-smoothing", type=float, default=0.04)
    ap.add_argument("--quad-model", default="x500_visual", help="x500_visual (with camera frustum) or x500_visual_plain")
    a = ap.parse_args()
    repo = os.environ.get("UAV_DT_REPO") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    traj = np.loadtxt(os.path.join(a.run, "trajectory.csv"), skiprows=1)
    t = traj[:, 0] - traj[0, 0]
    t_end = a.end if a.end is not None else t[-1]
    print(f"trajectory: {len(traj)} poses over {t[-1]:.0f} s, replaying {a.start:.0f} to {t_end:.0f} s at {a.fps} fps")
    gz_set_pose = os.path.join(repo, "build", "gz_set_pose", "gz_set_pose")

    world_file = os.path.join(repo, "sim", "worlds", a.world + ".sdf")
    # leftovers from an earlier replay would fight this one for the world and the pose service
    subprocess.run(["pkill", "-x", "ruby"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-x", "gz_set_pose"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)
    procs = []
    gz_log = open(os.path.join(a.run, "replay_gz.log"), "w")
    try:
        procs.append(subprocess.Popen(["gz", "sim", "-s", "-r", "--headless-rendering", "-v", "2", world_file],
                                      stdout=gz_log, stderr=subprocess.STDOUT))
        for _ in range(60):                                   # wait for the world clock, up to 30 s
            time.sleep(0.5)
            if procs[0].poll() is not None:
                sys.exit(f"gz sim exited early, see {gz_log.name}")
            topics = subprocess.run(["gz", "topic", "-l"], capture_output=True, text=True).stdout
            if f"/world/{a.world}/clock" in topics:
                break
        else:
            sys.exit(f"gz sim did not come up, see {gz_log.name}")
        for name in (a.quad_model, "follow_cam"):
            subprocess.run(["ros2", "run", "ros_gz_sim", "create", "-world", a.world, "-name", name,
                            "-file", os.path.join(repo, "sim", "models", name, "model.sdf"), "-z", "0.3"], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append(subprocess.Popen(["ros2", "run", "ros_gz_bridge", "parameter_bridge",
                                       "/follow_cam/image@sensor_msgs/msg/Image[gz.msgs.Image"],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        time.sleep(3)
        setter = subprocess.Popen([gz_set_pose, a.world], stdin=subprocess.PIPE, text=True, bufsize=1)
        procs.append(setter)
        # place both at the start pose, then start recording
        idx = np.searchsorted(t, a.start)
        x, y, z, qx, qy, qz, qw = traj[idx, 1:]
        yaw_f = quat_yaw(qx, qy, qz, qw)
        cx, cy, cz, q = follow_pose(x, y, z, yaw_f, a.distance, a.height, math.radians(a.pitch_deg))
        setter.stdin.write(f"{a.quad_model} {x:.3f} {y:.3f} {z:.3f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
        setter.stdin.write(f"follow_cam {cx:.3f} {cy:.3f} {cz:.3f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")
        time.sleep(2)
        replay_dir = os.path.join(a.run, "replay")            # keep the live recording of the same topic intact
        os.makedirs(replay_dir, exist_ok=True)
        rec = subprocess.Popen(["python3", os.path.join(repo, "scripts", "record_video.py"), "--out-dir", replay_dir,
                                "--fps", str(a.fps), "/follow_cam/image"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append(rec)
        time.sleep(2)
        # step the trajectory in wall time at the recording frame rate (the sim runs at real time)
        wall0 = time.time()
        step = 1.0 / a.fps
        k = 0
        while True:
            tt = a.start + k * step
            if tt > t_end:
                break
            i = min(np.searchsorted(t, tt), len(t) - 1)
            i0 = max(i - 1, 0)
            f = 0.0 if t[i] == t[i0] else (tt - t[i0]) / (t[i] - t[i0])
            p = traj[i0, 1:4] + f * (traj[i, 1:4] - traj[i0, 1:4])
            qx, qy, qz, qw = traj[i, 4:8]
            yaw = quat_yaw(qx, qy, qz, qw)
            yaw_f += a.yaw_smoothing * math.atan2(math.sin(yaw - yaw_f), math.cos(yaw - yaw_f))
            cx, cy, cz, q = follow_pose(p[0], p[1], p[2], yaw_f, a.distance, a.height, math.radians(a.pitch_deg))
            setter.stdin.write(f"{a.quad_model} {p[0]:.3f} {p[1]:.3f} {p[2]:.3f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
            setter.stdin.write(f"follow_cam {cx:.3f} {cy:.3f} {cz:.3f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")
            if procs[0].poll() is not None:
                sys.exit(f"gz sim died during the replay, see {gz_log.name}")
            k += 1
            sleep_for = wall0 + k * step - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
            if k % (a.fps * 30) == 0:
                print(f"  {tt:.0f} s", flush=True)
        time.sleep(2)
        rec.send_signal(2)
        rec.wait(timeout=30)
        os.replace(os.path.join(replay_dir, "follow_cam_image.mp4"), os.path.join(a.run, "replay_follow.mp4"))
        print(f"wrote {os.path.join(a.run, 'replay_follow.mp4')}")
    finally:
        for p in procs:
            try:
                p.send_signal(2)
            except Exception:
                pass
        time.sleep(2)
        subprocess.run(["pkill", "-x", "ruby"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for p in procs:
            try:
                p.kill()
            except Exception:
                pass


if __name__ == "__main__":
    main()
