# UAV Detect-and-Track

Real-time object **detection + multi-object tracking** for UAVs, designed to run on a companion computer and to drive a closed-loop "keep the target centred" behaviour later. Runs identically on a **quadrotor and a fixed-wing** — the perception is platform-agnostic; only the downstream control differs.

> **Scope / framing:** this is a civilian aerial-robotics demo — wildlife/livestock monitoring, traffic, infrastructure, search-and-rescue. It is not a targeting or surveillance system. Uses public datasets only. (See the portfolio policy in `../tier1_brand_starter.md` and the demo spec in `../demo_specs_autonomy.md`.)

## What it does
- Detects objects per frame (YOLO-class detector) and assigns **stable track IDs** across frames.
- Emits, per frame, the tracks and a **normalised target offset** (the selected target's distance from image centre, in [-1, 1]) — the signal the closed-loop orbit/loiter demo will consume.
- Writes an annotated video + a `tracks.jsonl` log; benchmarkable on-device.

## Design
The **tracking pipeline is numpy-only** so it's testable offline without GPU/weights. The detector is swappable behind an interface:
- `YoloDetector` — Ultralytics YOLO (real demo).
- `StubDetector` — scripted detections (used by the offline smoke test).

The tracker (`ByteTrackLite`) is a compact **ByteTrack-style** two-stage IoU association (high- then low-confidence detections) with track lifecycle — dependency-free. For production, swap in the reference ByteTrack / OC-SORT or Ultralytics' built-in tracker.

```
uav-detect-and-track/
├── detect_track.py        # CLI: video/camera -> annotated video + tracks.jsonl
├── smoke_test.py          # offline, numpy-only end-to-end check
├── uav_dt/
│   ├── geometry.py        # IoU, greedy matching, target offset
│   ├── tracker.py         # ByteTrackLite (two-stage association)
│   ├── detector.py        # YoloDetector + StubDetector
│   └── pipeline.py        # frames -> detections -> tracks -> target
├── ros2/detect_track_node.py   # ROS 2 node (subscribe Image, publish offset)
└── sitl/README.md         # PX4/ArduPilot SITL bring-up (quad + fixed-wing)
```

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # numpy is enough for the smoke test
```

## Usage
```bash
# Run on a video (downloads yolo11n.pt on first use). Track only people+vehicles:
python detect_track.py --source clip.mp4 --model yolo11n.pt --classes 0 2 \
    --output out.mp4 --tracks tracks.jsonl

# Webcam:
python detect_track.py --source 0 --show

# Offline smoke test (numpy only, no weights/network):
python smoke_test.py
```

## Data
Train/evaluate on public aerial sets — **VisDrone**, UAVDT. The repo ships no data.

## ROS 2 / SITL
`ros2/detect_track_node.py` subscribes to a `sensor_msgs/Image` topic, runs the pipeline, and publishes the normalised target offset (`geometry_msgs/PointStamped`) plus an annotated image — the bridge to closed-loop control. See `sitl/README.md` to run it against PX4/ArduPilot SITL on both a multicopter and a fixed-wing model.

## License
MIT
