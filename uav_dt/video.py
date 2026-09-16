"""Shared geometry for the replay renderer and the clip annotator: the follow camera's pose per frame.

Both must agree exactly, so the pose sequence is computed here from the trajectory and the camera
parameters, stepping at pose_hz as the replay does and sampling at the video frame times.
"""
from __future__ import annotations

import math

import numpy as np


def quat_yaw(qx, qy, qz, qw) -> float:
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def follow_pose(x, y, z, yaw, d, h, pitch):
    """Camera position and (x, y, z, w) orientation: d behind the nose, h above, pitched down, no roll."""
    cx, cy, cz = x - d * math.cos(yaw), y - d * math.sin(yaw), z + h
    cy2, sy2, cp2, sp2 = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2)
    return (cx, cy, cz), (-sy2 * sp2, cy2 * sp2, sy2 * cp2, cy2 * cp2)


def interpolate_pose(traj, t, tt):
    """Quad position (linear) and orientation (nearest sample) at trajectory time tt."""
    i = min(int(np.searchsorted(t, tt)), len(t) - 1)
    i0 = max(i - 1, 0)
    f = 0.0 if t[i] == t[i0] else (tt - t[i0]) / (t[i] - t[i0])
    p = traj[i0, 1:4] + f * (traj[i, 1:4] - traj[i0, 1:4])
    return p, tuple(traj[i, 4:8])


def camera_track(traj, start, end, fps, pose_hz=None, distance=7.0, height=3.0, pitch_deg=20.0, yaw_smoothing=0.15):
    """Per video frame: (frame time, quad position, quad quaternion, camera position, camera quaternion).

    Mirrors scripts/replay_follow.py: the yaw filter runs at pose_hz from the start pose, and each
    video frame uses the most recent pose tick at or before its own time."""
    pose_hz = pose_hz or fps
    t = traj[:, 0] - traj[0, 0]
    pitch = math.radians(pitch_deg)
    idx = min(int(np.searchsorted(t, start)), len(t) - 1)
    yaw_f = quat_yaw(*traj[idx, 4:8])
    step = 1.0 / pose_hz
    frames, k, next_frame = [], 0, 0
    frame_dt = 1.0 / fps
    last = None
    while True:
        tt = start + k * step
        if tt > end:
            break
        p, q = interpolate_pose(traj, t, tt)
        yaw = quat_yaw(*q)
        yaw_f += yaw_smoothing * math.atan2(math.sin(yaw - yaw_f), math.cos(yaw - yaw_f))
        cam_p, cam_q = follow_pose(p[0], p[1], p[2], yaw_f, distance, height, pitch)
        last = (tt, tuple(p), q, cam_p, cam_q)
        while start + next_frame * frame_dt <= tt + 1e-9:
            frames.append((start + next_frame * frame_dt,) + last[1:])
            next_frame += 1
        k += 1
    return frames
