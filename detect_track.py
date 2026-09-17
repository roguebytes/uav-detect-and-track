"""Run detect-and-track on a video file or camera; write annotated video + tracks.jsonl.

python detect_track.py --source clip.mp4 --model yolo11n.pt --classes 0 2 \
        --output out.mp4 --tracks tracks.jsonl
"""

from __future__ import annotations

__author__ = "Frank Loewenich"

import argparse
import json

from uav_dt.detector import YoloDetector
from uav_dt.pipeline import Pipeline


def parse_args():
    """Parse the command line."""
    p = argparse.ArgumentParser(description="UAV detect-and-track")
    p.add_argument("--source", required=True, help="Video path, or webcam index (e.g. 0)")
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--classes", type=int, nargs="*", default=None, help="COCO class ids to keep")
    p.add_argument("--target-policy", default="best", help="'best', 'largest', or a track id")
    p.add_argument("--output", default=None, help="Annotated video output path")
    p.add_argument("--tracks", default=None, help="Per-frame tracks JSONL output path")
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def draw(frame, result):
    """Draw track boxes, the image centre and the selected target's offset on a frame."""
    import cv2
    h, w = frame.shape[:2]
    for t in result["tracks"]:
        x1, y1, x2, y2 = (int(v) for v in t["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID{t['id']}", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.drawMarker(frame, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
    tg = result["target"]
    if tg:
        cx = int((tg["box"][0] + tg["box"][2]) / 2)
        cy = int((tg["box"][1] + tg["box"][3]) / 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.line(frame, (w // 2, h // 2), (cx, cy), (0, 0, 255), 2)
        cv2.putText(frame, f"offset {tg['offset'][0]:+.2f},{tg['offset'][1]:+.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame


def main():
    """Run detection and tracking over a video or camera and write the outputs."""
    import cv2
    args = parse_args()
    policy = int(args.target_policy) if args.target_policy.lstrip("-").isdigit() else args.target_policy

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    pipeline = Pipeline(YoloDetector(args.model, args.conf, args.classes), target_policy=policy)

    writer, tracks_f = None, None
    if args.output:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if args.tracks:
        tracks_f = open(args.tracks, "w")

    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = pipeline.process(frame, idx)
            if tracks_f:
                tracks_f.write(json.dumps(result) + "\n")
            if writer or args.show:
                annotated = draw(frame, result)
                if writer:
                    writer.write(annotated)
                if args.show:
                    cv2.imshow("detect-and-track", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            idx += 1
    finally:
        cap.release()
        if writer:
            writer.release()
        if tracks_f:
            tracks_f.close()
        if args.show:
            cv2.destroyAllWindows()
    print(f"Processed {idx} frames.")


if __name__ == "__main__":
    main()
