#!/usr/bin/env python3
"""Cinematic replay: re-render the follow camera along a logged trajectory with nothing else loading the GPU.

    source scripts/env.sh
    python3 scripts/replay_follow.py --run runs/real_sparse --world bowl_field_sparse [--fps 30] [--start 0 --end 120]

Starts a headless Gazebo server on the world, spawns a static visual-only x500 and the follow camera,
then steps both along the trajectory at the camera's frame rate through tools/gz_set_pose while
scripts/record_video.py records /follow_cam/image. Output: <run>/replay_follow.mp4.

During a mission the 12 MP survey camera stalls Gazebo's render thread for over 100 ms every
second, so the live follow-camera recording skips and repeats frames. Here the only sensor is the
follow camera and the world is stepped in lockstep: the physics step equals the camera period, so
for every video frame the script sets the quad and camera poses, steps the paused world once, and
waits for the camera to write that frame's PNG. Frame k therefore shows exactly pose k of
uav_dt.video.camera_track, which tools/annotate_follow.py uses to overlay geometry. Wall time is
whatever the renderer needs; the video plays at true speed.
"""
import argparse
import math
import os
import subprocess
import sys
import time

import signal

import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from uav_dt.video import camera_track  # noqa: E402




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
    ap.add_argument("--yaw-smoothing", type=float, default=0.15)
    ap.add_argument("--quad-model", default="x500_visual", help="x500_visual (with camera frustum) or x500_visual_plain")
    ap.add_argument("--hfov-deg", type=float, default=70.0, help="follow camera horizontal field of view")
    ap.add_argument("--out", default=None, help="output MP4 (default <run>/replay_follow.mp4)")
    a = ap.parse_args()
    repo = os.environ.get("UAV_DT_REPO") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    traj = np.loadtxt(os.path.join(a.run, "trajectory.csv"), skiprows=1)
    t = traj[:, 0] - traj[0, 0]
    t_end = a.end if a.end is not None else t[-1]
    print(f"trajectory: {len(traj)} poses over {t[-1]:.0f} s, replaying {a.start:.0f} to {t_end:.0f} s at {a.fps} fps")
    gz_set_pose = os.path.join(repo, "build", "gz_set_pose", "gz_set_pose")

    # a copy of the world whose physics step is one camera period, so one world step renders one frame
    src = open(os.path.join(repo, "sim", "worlds", a.world + ".sdf")).read()
    import re
    world_file = os.path.join(a.run, f"replay_{a.world}.sdf")
    src = re.sub(r"<max_step_size>[^<]*</max_step_size>", f"<max_step_size>{1.0 / a.fps:.9f}</max_step_size>", src, count=1)
    src = re.sub(r"<real_time_update_rate>[^<]*</real_time_update_rate>", "<real_time_update_rate>0</real_time_update_rate>", src, count=1)
    open(world_file, "w").write(src)
    # leftovers from an earlier replay would fight this one for the world and the pose service
    subprocess.run(["pkill", "-x", "ruby"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-x", "gz_set_pose"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)
    procs = []
    gz_log = open(os.path.join(a.run, "replay_gz.log"), "w")
    try:
        # started paused: every step is commanded explicitly
        procs.append(subprocess.Popen(["gz", "sim", "-s", *([] if os.environ.get("REPLAY_GLX") else ["--headless-rendering"]), "-v", "2", world_file],
                                      stdout=gz_log, stderr=subprocess.STDOUT, start_new_session=True))
        for _ in range(60):                                   # wait for the world clock, up to 30 s
            time.sleep(0.5)
            if procs[0].poll() is not None:
                sys.exit(f"gz sim exited early, see {gz_log.name}")
            topics = subprocess.run(["gz", "topic", "-l"], capture_output=True, text=True).stdout
            if f"/world/{a.world}/clock" in topics:
                break
        else:
            sys.exit(f"gz sim did not come up, see {gz_log.name}")
        # the follow camera saves every rendered frame to PNG: transport from Gazebo to a recorder drops
        # a third of the 2.7 MB frames on this machine, and a video assembled from files has no gaps
        frames_dir = os.path.join(a.run, "replay_frames")
        if os.path.isdir(frames_dir):
            for f in os.listdir(frames_dir):
                os.remove(os.path.join(frames_dir, f))
        os.makedirs(frames_dir, exist_ok=True)
        cam_sdf = open(os.path.join(repo, "sim", "models", "follow_cam", "model.sdf")).read()
        cam_sdf = cam_sdf.replace("</clip>", f"</clip>\n          <save enabled=\"true\"><path>{frames_dir}</path></save>")
        cam_sdf = cam_sdf.replace("<update_rate>30</update_rate>", f"<update_rate>{a.fps}</update_rate>")
        cam_sdf = re.sub(r"<horizontal_fov>[^<]*</horizontal_fov>", f"<horizontal_fov>{math.radians(a.hfov_deg):.6f}</horizontal_fov>", cam_sdf)
        cam_file = os.path.join(a.run, "replay_follow_cam.sdf")
        open(cam_file, "w").write(cam_sdf)
        for name, model_file in ((a.quad_model, os.path.join(repo, "sim", "models", a.quad_model, "model.sdf")), ("follow_cam", cam_file)):
            subprocess.run(["ros2", "run", "ros_gz_sim", "create", "-world", a.world, "-name", name,
                            "-file", model_file, "-z", "0.3"], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        setter = subprocess.Popen([gz_set_pose, a.world], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
                                  start_new_session=True)
        procs.append(setter)

        def cmd(line):
            setter.stdin.write(line + "\n")

        def step():
            cmd("step 1")
            while setter.stdout.readline().strip() != "stepped":
                pass

        track = camera_track(traj, a.start, t_end, a.fps, pose_hz=a.fps, distance=a.distance, height=a.height,
                             pitch_deg=a.pitch_deg, yaw_smoothing=a.yaw_smoothing)

        def place(frame):
            _, qp, qq, cp, cq = frame
            cmd(f"{a.quad_model} {qp[0]:.4f} {qp[1]:.4f} {qp[2]:.4f} {qq[0]:.6f} {qq[1]:.6f} {qq[2]:.6f} {qq[3]:.6f}")
            cmd(f"follow_cam {cp[0]:.4f} {cp[1]:.4f} {cp[2]:.4f} {cq[0]:.6f} {cq[1]:.6f} {cq[2]:.6f} {cq[3]:.6f}")
            cmd("ping")
            while setter.stdout.readline().strip() != "ok":
                pass

        # let the spawns and the first pose settle: a few steps, then discard whatever was rendered
        place(track[0])
        for _ in range(5):
            step()
        time.sleep(1)
        # Gazebo renders a camera only while something subscribes; the frames themselves come from the PNGs
        sub = subprocess.Popen(["gz", "topic", "-e", "-t", "/follow_cam/image"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               start_new_session=True)
        procs.append(sub)
        time.sleep(2)
        for _ in range(3):
            step()
        time.sleep(1)
        for f in os.listdir(frames_dir):                      # drop frames rendered before the start pose settled
            os.remove(os.path.join(frames_dir, f))

        def png_count():
            return sum(1 for f in os.listdir(frames_dir) if f.endswith(".png"))
        # lockstep: pose k, one world step, wait for PNG k
        wall_start = time.time()
        for k, frame in enumerate(track):
            place(frame)
            before = png_count()
            step()
            for _ in range(600):                              # up to 30 s for the render and PNG write
                if png_count() > before:
                    break
                time.sleep(0.05)
            else:
                print(f"  warning: no frame written for step {k}", flush=True)
            if procs[0].poll() is not None:
                sys.exit(f"gz sim died during the replay, see {gz_log.name}")
            if (k + 1) % (a.fps * 30) == 0:
                print(f"  {frame[0]:.0f} s  ({k + 1} frames, wall {time.time() - wall_start:.0f} s)", flush=True)
        time.sleep(1)
        sub.send_signal(2)
        # assemble the PNG sequence (Gazebo names them <scoped sensor>_<n>.png, in render order)
        pngs = sorted((f for f in os.listdir(frames_dir) if f.endswith(".png")), key=lambda f: int(f.rsplit("_", 1)[1][:-4]))
        for i, f in enumerate(pngs):
            os.replace(os.path.join(frames_dir, f), os.path.join(frames_dir, f"frame_{i:06d}.png"))
        out = a.out or os.path.join(a.run, "replay_follow.mp4")
        encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        codec = ["-c:v", "h264_nvenc", "-preset", "p6", "-cq", "22", "-b:v", "0"] if "h264_nvenc" in encoders else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(a.fps), "-i", os.path.join(frames_dir, "frame_%06d.png"),
                        *codec, "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], check=True)
        for f in os.listdir(frames_dir):
            os.remove(os.path.join(frames_dir, f))
        print(f"wrote {out}: {len(pngs)} frames ({len(track)} poses for {t_end - a.start:.0f} s at {a.fps} fps)")
    finally:
        # kill each child's whole process group: `ros2 run` and `gz` are wrappers whose real
        # processes otherwise outlive them and pile up (32 orphaned bridges once brought the machine down)
        for p in procs:
            try:
                os.killpg(p.pid, signal.SIGINT)
            except Exception:
                pass
        time.sleep(2)
        for p in procs:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:
                pass


if __name__ == "__main__":
    main()
