# UAV detect-and-track: survey high, verify low

A ROS 2 and Gazebo simulation of a small quadrotor that surveys a grass field at 40 m, detects
white 16 cm bowls with an onboard YOLOv9-C detector, tracks and geolocates them, then descends
once to 11 m and hops between the candidates to verify each one. It mirrors the flight profile
from my Remote Sensing paper on time-efficient UAV search strategies, in which surveying high and
verifying low cut mission cost by 46% in sparse fields.

![verification pass: the annotated nadir view at 11 m](docs/results/verify.gif)

> Scope: a civilian aerial-robotics demo (search and rescue, conservation, agriculture, survey
> planning). The perception code is airframe-agnostic; the control action sits behind an interface
> so a fixed-wing variant can follow. Public data only.

## Problem

Small objects seen from survey altitude are a few pixels wide and easy to confuse, while low
passes are slow. The paper's answer is a two-stage profile: one fast high pass to find
candidates, one low pass to confirm them. This repo runs that profile end to end in simulation,
from takeoff to landing, with a detector trained on real drone imagery, and scores the result
against the world's ground truth.

## Results

| | Sparse field (12 bowls, 12.5 per ha) | Dense field (60 bowls, 62.5 per ha) |
|---|---|---|
| Survey recall / precision | 1.00 / 1.00 | 1.00 / 1.00 |
| Bowls found on the survey (TP / FP / FN) | 12 / 0 / 0 | 60 / 0 / 0 |
| Verified true / rejected | 12 / 0 | 60 / 0 |
| Verified recall / precision | 1.00 / 1.00 | 1.00 / 1.00 |
| Geolocation error, mean / max (m) | 0.45 / 0.56 | 0.48 / 0.65 |
| Mission time, takeoff to landing (s) | 290 | 667 |
| Frames processed / inference per frame (s) | 187 / 0.70 | 265 / 0.60 |

Field 120 x 80 m, survey at 40 m and 5 m/s with 30% side overlap, verify at 11 m with one descent and
hops between candidates, every second frame processed at survey altitude. Ground truth from the
world generator's manifest; a track counts as correct within 2 m of an unmatched bowl on the survey
and within 0.75 m at verification. Full tables: `docs/results/sparse.md`, `docs/results/dense.md`.


Detector: YOLOv9-C fine-tuned on real DJI Mini 4 Pro frames of bowls on grass at 11, 15 and
40 m (Loewenich et al. 2026), run on 640 px tiles at native resolution, FP16 on an RTX 2070.
Geolocation uses the autopilot's own position and attitude estimate through MAVROS, not the
simulator's ground truth, so the errors above include estimator error.

## How it works

```
Gazebo Garden ─ x500 quad + nadir 4032x3024 camera ─┐
PX4 SITL v1.15.4 (attached to that model) ──────────┤ MAVROS ── mission node (survey planner,
ros_gz bridge (camera, clock, ground-truth odometry) ┘            single-descent verify pass,
                                                                  PX4 offboard through a
perception node: tiled YOLO ─► pixel-to-ground geolocation ─►     FlightController interface)
                 ground-plane tracker (distance association,
                 duplicate merging) ─► tracks, verdicts, JSONL log ─► scripts/score.py
```

- `uav_dt/`: detector, geolocation, tracker, pipeline, mission state machine, controller adapters. Numpy only apart from the detector; 25 unit tests.
- `ros2_ws/src/uav_dt_ros/`: `sim.launch.py` (Gazebo, spawn, PX4, MAVROS, bridges, follow camera) and `mission.launch.py` (perception, mission, optional recording).
- `sim/`: world generator with a JSON ground-truth manifest, camera and bowl models, procedural grass.
- `scripts/`: `setup.sh`, `env.sh`, `smoke.sh`, `score.py`, `record_video.py`, `install_mavros.sh`.

## Run it

```bash
scripts/setup.sh                       # once, see docs/setup.md for the pinned versions
source scripts/env.sh
python -m pytest && python smoke_test.py
scripts/smoke.sh                       # headless, oracle detector, about 4 minutes, no weights needed
# real detector (weights in models/, see models/README.md), MAVROS pose, recording:
DETECTOR=yolo POSE=mavros RECORD=true MAX_VERIFY=0 FRAME_STRIDE=2 MIN_RECALL=0 RUN_NAME=sparse scripts/smoke.sh
python3 scripts/score.py runs/sparse/perception.jsonl sim/worlds/bowl_field_sparse.json
```

`ros2 launch uav_dt_ros sim.launch.py headless:=false` opens the Gazebo GUI.

## Scope and limitations

- Simulation only, so far. The real airframes run ArduPilot; the MAVROS-based controller keeps the swap to an ArduPilot adapter small, but that is unflown.
- The camera is body-fixed. Frames tilted beyond 12 degrees are skipped, which a gimbal would make unnecessary.
- Flat ground is assumed for geolocation.
- The grass is a procedural texture; the detector was trained on real turf and transferred without retraining, but this is not a claim about real-world recall.
- The workstation's thermal limits shaped some choices (FP16, every second frame at survey altitude). See docs/setup.md.

## Tech

Python, numpy, PyTorch, Ultralytics, ROS 2 Humble, Gazebo Garden, PX4 SITL 1.15.4, MAVROS 2.14, ffmpeg.

## License

MIT
