# UAV detect-and-track: survey high, verify low

A ROS 2 and Gazebo simulation of a small quadrotor that surveys a grass field at 40 m, detects
white 16 cm bowls with an onboard YOLOv9-C detector, tracks and geolocates them, then descends
once to 11 m and hops between the candidates to verify each one. It mirrors the flight profile
from my Remote Sensing paper on time-efficient UAV search strategies (Loewenich et al. 2026), in
which surveying high and verifying low cut mission cost by 46% in sparse fields, at a 22% penalty
in dense ones, with the crossover at a target density of 0.48 per grid cell.

![verification pass: the annotated nadir view at 11 m](docs/results/verify.gif)

> Scope: a civilian aerial-robotics demo (search and rescue, conservation, agriculture, survey
> planning). The perception code is airframe-agnostic. The control action sits behind an interface
> so a fixed-wing variant can follow. Public data only.

## Problem

Small objects seen from survey altitude are a few pixels wide and easy to confuse, while low
passes are slow. The paper's answer is a two-stage profile: one fast high pass to find
candidates, one low pass to confirm them. This repo runs that profile end to end in simulation,
from takeoff to landing, with a detector trained on real drone imagery, and scores the result
against the world's ground truth.

## Results

| | Sparse field (12 targets, 9.4 per ha) | Dense field (60 targets, 47 per ha) |
|---|---|---|
| Survey recall / precision | 1.00 / 1.00 | 0.98 / 0.98 |
| Targets found on the survey (TP / FP / FN) | 12 / 0 / 0 | 59 / 1 / 1 |
| Verified true / rejected | 12 / 0 | 58 / 1 |
| Verified recall / precision | 1.00 / 1.00 | 0.97 / 1.00 |
| Geolocation error, mean / max (m) | 0.21 / 0.35 | 0.20 / 0.41 |
| Mission time, takeoff to landing (s) | 248 | 617 |
| Frames processed / inference per frame (s) | 248 / 0.63 | 617 / 0.65 |

One mission per world, flown on 2026-09-17, so the recall figures are single-flight outcomes rather
than statistics.
Field 120 x 106.4 m, sized so that two survey passes at 40 m with 10% side overlap cover it exactly
(56 m swath, 50.4 m spacing), following the sweep model of the paper's section 3.2.4: a pass spans
only the centres of its first and last footprints, so the aircraft turns when the footprint reaches
the boundary. Survey at 5 m/s, one frame per second. Verify at 11 m with one descent and hops
between candidates ordered nearest-neighbour (the paper uses a TSP solver). Ground truth from the
world generator's manifest. A track counts as correct within 2 m of an unmatched bowl on the survey
and within 1 m at verification. Full tables: `docs/results/sparse.md`, `docs/results/dense.md`.

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

- `uav_dt/`: detector, geolocation, tracker, pipeline, mission state machine, controller adapters. Numpy only apart from the detector. 28 unit tests.
- `ros2_ws/src/uav_dt_ros/`: `sim.launch.py` (Gazebo, spawn, PX4, MAVROS, bridges) and `mission.launch.py` (perception, mission, optional recording).
- `sim/`: world generator with a JSON ground-truth manifest, camera and bowl models, procedural grass.
- `scripts/`: `setup.sh`, `env.sh`, `smoke.sh`, `score.py`, `record_video.py`, `replay_follow.py`, `install_mavros.sh`.
- `tools/`: world and texture generators, `flight_path_video.py` (top-down path animation), `annotate_follow.py`, `video_smoothness.py`.
- `detect_track.py`: standalone detect-and-track CLI for recorded video, independent of ROS and the simulator.

## Video

A mission records the annotated nadir view live. The third-person clips are rendered afterwards by
`scripts/replay_follow.py`, which replays the logged 50 Hz trajectory in a sim whose only sensor is
the follow camera, runs the world below real time so the renderer keeps up, and takes the frames
from the camera's own PNG output rather than a topic, so no frame is dropped or repeated. The survey
camera's field of view is drawn as a translucent cone ending at its footprint. `tools/flight_path_video.py` animates the flight path with
the survey and the verification pass in different colours. The clips are not in git. They are
published at https://loewenich.com.

```bash
python3 scripts/replay_follow.py --run runs/sparse --world bowl_field_sparse --distance 16 --height 11 --pitch-deg 47
python3 tools/flight_path_video.py --run runs/sparse --world bowl_field_sparse --out flight_path.mp4
```

## Run it

```bash
scripts/setup.sh                       # once, see docs/setup.md for the pinned versions
source scripts/env.sh
python3 -m pytest && python3 smoke_test.py  # inside the venv that setup.sh creates
scripts/smoke.sh                       # headless, oracle detector, about 4 minutes, no weights needed
# real detector (weights in models/, see models/README.md), MAVROS pose, recording:
DETECTOR=yolo POSE=mavros RECORD=true MAX_VERIFY=0 MIN_RECALL=0 RUN_NAME=sparse scripts/smoke.sh
# CAMERA_HZ=0.5 halves the render and inference load; FRAME_STRIDE=2 skips every second survey frame
python3 scripts/score.py runs/sparse/perception.jsonl sim/worlds/bowl_field_sparse.json
```

`ros2 launch uav_dt_ros sim.launch.py headless:=false` opens the Gazebo GUI.

## Scope and limitations

- Simulation only, so far. The real airframes run ArduPilot. The MAVROS-based controller keeps the swap to an ArduPilot adapter small, but that is unflown.
- The camera is body-fixed. Frames tilted beyond 20 degrees are skipped, which a gimbal would make unnecessary.
- Flat ground is assumed for geolocation.
- The grass is a procedural texture. The detector was trained on real turf and transferred without retraining, but this is not a claim about real-world recall.
- The workstation's thermal limits shaped some choices (FP16 inference, a 1 Hz survey camera, optional frame skipping). See docs/setup.md.

## Tech

Python, numpy, PyTorch, Ultralytics, ROS 2 Humble, Gazebo Garden, PX4 SITL 1.15.4, MAVROS 2.14, ffmpeg.

## Reference

Loewenich, F., Maire, F., Sandino, J., & Gonzalez, F. (2026). Fly High or Fly Low? Selecting
Time-Efficient UAV Search Strategies for High-Recall Aerial Detection. *Remote Sensing*, 18(18),
3129. https://doi.org/10.3390/rs18183129

## License

MIT
