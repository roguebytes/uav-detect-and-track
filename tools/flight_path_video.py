#!/usr/bin/env python3
"""Top-down flight-path animation from a run's trajectory log: survey and verify in different colours.

    python3 tools/flight_path_video.py --run runs/real_sparse --world bowl_field_sparse --out docs/results/clips/flight_path.mp4 [--speed 8]

Draws the field, the targets, the path so far (survey legs in blue, verification pass in orange,
transit and return in grey), the quad's current position, and an altitude profile along the bottom.
The mission's state changes come from mission.jsonl, refined by the flight itself: the survey is
blue only from the moment the quad reaches the first survey waypoint (the leg from the takeoff
point to the field corner is transit), and the verification pass is orange from the start of the
descent (the 40 m transit to the first candidate is transit). Rendered with OpenCV and encoded
with ffmpeg.
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
from uav_dt.geolocate import CameraModel  # noqa: E402
from uav_dt.mission.survey import lawnmower  # noqa: E402

COL = {"survey": (220, 120, 40), "verify": (40, 140, 255), "other": (150, 150, 150)}   # BGR
BG, GRASS, TEXT = (24, 24, 24), (46, 78, 40), (235, 235, 235)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--world", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--speed", type=float, default=8.0, help="playback speed factor")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--survey-alt", type=float, default=40.0)
    ap.add_argument("--side-overlap", type=float, default=0.1, help="as flown, to reproduce the first survey waypoint")
    ap.add_argument("--wp-tol", type=float, default=2.5)
    a = ap.parse_args()

    traj = np.loadtxt(os.path.join(a.run, "trajectory.csv"), skiprows=1)
    t, xyz = traj[:, 0] - traj[0, 0], traj[:, 1:4]
    mission = [json.loads(l) for l in open(os.path.join(a.run, "mission.jsonl"))]
    t0 = traj[0, 0]
    states = sorted((r["t"] - t0, r["state"]) for r in mission)
    manifest = json.load(open(os.path.join("sim", "worlds", a.world + ".json")))
    bowls = [(b["x"], b["y"]) for b in manifest["bowls"]]
    fw, fh = manifest["field_m"]

    def state_at(tt):
        cur = "init"
        for ts, st in states:
            if tt >= ts:
                cur = st
        return cur

    # the survey legs start when the quad reaches the first planned waypoint; the verification pass
    # starts with the descent from survey altitude
    wp0 = lawnmower(fw, fh, CameraModel.dji_mini4pro_still(), a.survey_alt, a.side_overlap)[0]
    t_survey_state = next((ts for ts, st in states if st == "survey"), None)
    t_verify_state = next((ts for ts, st in states if st == "verify"), None)
    t_legs = next((t[i] for i in range(len(t)) if t_survey_state is not None and t[i] >= t_survey_state
                   and math.hypot(xyz[i, 0] - wp0[0], xyz[i, 1] - wp0[1]) < a.wp_tol), t_survey_state)
    # the descent begins where the quad first sinks faster than 0.5 m/s (over a 1 s window) after
    # the verify state starts; the 40 m transit to the first candidate stays transit
    t_descent = t_verify_state
    if t_verify_state is not None:
        i0 = int(np.searchsorted(t, t_verify_state))
        for i in range(i0, len(t)):
            j = int(np.searchsorted(t, t[i] + 1.0))
            if j < len(t) and (xyz[j, 2] - xyz[i, 2]) / max(t[j] - t[i], 1e-6) < -0.5:
                t_descent = t[i]
                break

    def phase(tt):
        st = state_at(tt)
        if st == "survey" and t_legs is not None and tt >= t_legs:
            return "survey"
        if st == "verify" and t_descent is not None and tt >= t_descent:
            return "verify"
        return "other"

    W = a.width
    map_h = int(W * (fh + 20) / (fw + 20))
    prof_h = 160
    H = map_h + prof_h
    margin = 10.0                                              # metres of grass beyond the field
    sx = W / (fw + 2 * margin)
    sy = map_h / (fh + 2 * margin)

    def to_px(x, y):
        return int((x + fw / 2 + margin) * sx), int(map_h - (y + fh / 2 + margin) * sy)

    base = np.full((H, W, 3), BG, np.uint8)
    cv2.rectangle(base, to_px(-fw / 2 - margin, fh / 2 + margin), to_px(fw / 2 + margin, -fh / 2 - margin), GRASS, -1)
    cv2.rectangle(base, to_px(-fw / 2, fh / 2), to_px(fw / 2, -fh / 2), (90, 130, 80), 1)
    for bx, by in bowls:
        cv2.circle(base, to_px(bx, by), 5, (245, 245, 245), -1)
        cv2.circle(base, to_px(bx, by), 5, (60, 60, 60), 1)
    # legend
    for i, (name, col) in enumerate([("survey at 40 m", COL["survey"]), ("verification pass at 11 m", COL["verify"]), ("transit / return", COL["other"])]):
        y = 24 + 24 * i
        cv2.line(base, (18, y), (58, y), col, 4)
        cv2.putText(base, name, (68, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT, 1, cv2.LINE_AA)
    cv2.putText(base, f"{len(bowls)} targets, {fw:.0f} x {fh:.0f} m", (18, 24 + 24 * 3 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    # altitude profile axes
    px0, px1, py0, py1 = 60, W - 20, map_h + 20, H - 30
    cv2.rectangle(base, (px0, py0), (px1, py1), (60, 60, 60), 1)
    for alt in (11, 40):
        y = int(py1 - (alt / 45.0) * (py1 - py0))
        cv2.line(base, (px0, y), (px1, y), (70, 70, 70), 1)
        cv2.putText(base, f"{alt} m", (8, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    t_end = t[-1]

    def prof_px(tt, z):
        return int(px0 + tt / t_end * (px1 - px0)), int(py1 - (z / 45.0) * (py1 - py0))

    tmp = os.path.join(a.run, "flight_path_frames")
    os.makedirs(tmp, exist_ok=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    n_frames = int(t_end / a.speed * a.fps)
    path = base.copy()
    last_i = 0
    for k in range(n_frames + 1):
        tt = min(k * a.speed / a.fps, t_end)
        i = min(np.searchsorted(t, tt), len(t) - 1)
        for j in range(last_i, i):                             # accumulate the path drawn so far
            col = COL[phase(t[j])]
            cv2.line(path, to_px(*xyz[j, :2]), to_px(*xyz[j + 1, :2]), col, 2, cv2.LINE_AA)
            cv2.line(path, prof_px(t[j], xyz[j, 2]), prof_px(t[j + 1], xyz[j + 1, 2]), col, 2, cv2.LINE_AA)
        last_i = i
        frame = path.copy()
        x, y, z = xyz[i]
        cv2.circle(frame, to_px(x, y), 8, (255, 255, 255), -1)
        cv2.circle(frame, to_px(x, y), 8, (0, 0, 0), 2)
        cv2.circle(frame, prof_px(tt, z), 5, (255, 255, 255), -1)
        ph = phase(tt)
        cv2.putText(frame, f"t = {tt:5.0f} s   altitude {z:4.1f} m   {ph if ph != 'other' else 'transit'}", (px0, H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT, 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(tmp, f"f{k:05d}.png"), frame)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(a.fps), "-i", os.path.join(tmp, "f%05d.png"),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out], check=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    print(f"wrote {a.out}: {n_frames + 1} frames, {t_end:.0f} s of flight at {a.speed}x")


if __name__ == "__main__":
    main()
