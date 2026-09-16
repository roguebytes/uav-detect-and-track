#!/usr/bin/env python3
"""Overlay the bowls inside the survey camera's field of view onto a replayed follow-camera clip.

    python3 tools/annotate_follow.py --run runs/real_sparse --world bowl_field_sparse \\
        --clip docs/results/clips/survey_follow.mp4 --out docs/results/clips/survey_follow_annotated.mp4 \\
        --start 6.2 --end 127.4 --distance 24 --height 11 --pitch-deg 36

For every frame the quad pose comes from the trajectory and the follow camera pose from the same
maths the replay used (uav_dt.video.camera_track). A bowl is "in view" when it projects inside the
survey camera's 4032x3024 image; it is drawn where it projects in the follow camera's image. Green
means the perception node reported a detection within a metre of that bowl within
--detect-window seconds of the frame time during the mission; amber means in view but not
detected at that moment. The input clip is not modified.
"""
import argparse
import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from uav_dt.geolocate import CameraModel, project  # noqa: E402
from uav_dt.video import camera_track  # noqa: E402

GREEN, AMBER, WHITE, BLACK = (80, 220, 80), (40, 170, 255), (240, 240, 240), (0, 0, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--world", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--distance", type=float, default=7.0)
    ap.add_argument("--height", type=float, default=3.0)
    ap.add_argument("--pitch-deg", type=float, default=20.0)
    ap.add_argument("--yaw-smoothing", type=float, default=0.15)
    ap.add_argument("--follow-hfov-deg", type=float, default=70.0)
    ap.add_argument("--detect-window", type=float, default=1.5)
    ap.add_argument("--frame-offset", type=int, default=0,
                    help="video frame i is paired with track frame i + offset (the camera's PNG sequence can lead the pose track by a few frames)")
    a = ap.parse_args()
    if os.path.abspath(a.out) == os.path.abspath(a.clip):
        sys.exit("refusing to overwrite the input clip")

    traj = np.loadtxt(os.path.join(a.run, "trajectory.csv"), skiprows=1)
    t0 = traj[0, 0]
    bowls = [(b["id"], b["x"], b["y"]) for b in json.load(open(os.path.join("sim", "worlds", a.world + ".json")))["bowls"]]
    # detection times per bowl, from the perception log
    det_times = {bid: [] for bid, _, _ in bowls}
    for line in open(os.path.join(a.run, "perception.jsonl")):
        fr = json.loads(line)
        for d in fr["detections"]:
            if d["ground"] is None:
                continue
            bid, _, _ = min(bowls, key=lambda b: math.hypot(b[1] - d["ground"][0], b[2] - d["ground"][1]))
            bx, by = next((x, y) for i, x, y in bowls if i == bid)
            if math.hypot(bx - d["ground"][0], by - d["ground"][1]) <= 1.0:
                det_times[bid].append(fr["t"] - t0)
    det_times = {k: np.array(sorted(v)) for k, v in det_times.items()}

    def detected(bid, tt):
        ts = det_times[bid]
        if len(ts) == 0:
            return False
        i = np.searchsorted(ts, tt)
        cand = [ts[j] for j in (i - 1, i) if 0 <= j < len(ts)]
        return any(abs(c - tt) <= a.detect_window for c in cand)

    track = camera_track(traj, a.start, a.end, a.fps, distance=a.distance, height=a.height, pitch_deg=a.pitch_deg,
                         yaw_smoothing=a.yaw_smoothing)
    survey_cam = CameraModel.dji_mini4pro_still()
    cap = cv2.VideoCapture(a.clip)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    follow_cam = CameraModel(W, H, math.radians(a.follow_hfov_deg), mount=np.eye(3))
    encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    codec = ["-c:v", "h264_nvenc", "-preset", "p6", "-cq", "22", "-b:v", "0"] if "h264_nvenc" in encoders else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    ff = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                           "-r", str(a.fps), "-i", "-", *codec, "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
    n = 0
    while n + a.frame_offset < len(track):
        ok, frame = cap.read()
        if not ok:
            break
        if n + a.frame_offset < 0:
            n += 1
            continue
        tt, qp, qq, cp, cq = track[n + a.frame_offset]
        in_view, found = 0, 0
        for bid, bx, by in bowls:
            uv = project(survey_cam, (bx, by, 0.0), qp, qq)
            if uv is None or not (0 <= uv[0] < survey_cam.width and 0 <= uv[1] < survey_cam.height):
                continue
            fuv = project(follow_cam, (bx, by, 0.03), cp, cq)
            if fuv is None or not (0 <= fuv[0] < W and 0 <= fuv[1] < H):
                continue
            in_view += 1
            dist = math.dist(cp, (bx, by, 0.0))
            half = int(min(40, max(9, follow_cam.fx * 0.5 / max(dist, 1.0))))
            u, v = int(fuv[0]), int(fuv[1])
            hit = detected(bid, tt)
            found += hit
            col = GREEN if hit else AMBER
            cv2.rectangle(frame, (u - half, v - half), (u + half, v + half), BLACK, 4)
            cv2.rectangle(frame, (u - half, v - half), (u + half, v + half), col, 2)
        hud = f"{in_view} target{'s' if in_view != 1 else ''} in the camera's view, {found} detected"
        cv2.putText(frame, hud, (16, H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLACK, 4, cv2.LINE_AA)
        cv2.putText(frame, hud, (16, H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
        for i, (name, col) in enumerate((("detected", GREEN), ("in view, not yet detected", AMBER))):
            y = 26 + 28 * i
            cv2.rectangle(frame, (18, y - 9), (36, y + 9), BLACK, 4)
            cv2.rectangle(frame, (18, y - 9), (36, y + 9), col, 2)
            cv2.putText(frame, name, (46, y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 4, cv2.LINE_AA)
            cv2.putText(frame, name, (46, y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)
        ff.stdin.write(frame.tobytes())
        n += 1
    ff.stdin.close()
    ff.wait()
    print(f"wrote {a.out}: {n} frames")


if __name__ == "__main__":
    main()
