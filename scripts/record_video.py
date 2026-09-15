#!/usr/bin/env python3
"""Record one or more ROS 2 image topics to MP4 files, paced by the message timestamps.

    python3 scripts/record_video.py --out-dir runs/x --fps 15 /uav/chase /perception/annotated

Frames are written at a fixed fps in simulation time: if the renderer delivers frames slower than
real time, earlier frames are repeated so the video still plays at sim speed. Frames are piped to
ffmpeg as fragmented MP4 (H.264), so the file stays playable even if the process is killed before
it can close cleanly. Needs the system ROS 2 Python (rclpy, cv_bridge) and ffmpeg.
"""
import argparse
import os
import shutil
import signal
import subprocess

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class Recorder(Node):
    def __init__(self, topics, out_dir, fps, max_width):
        super().__init__("record_video")
        self.bridge, self.fps, self.max_width, self.out_dir = CvBridge(), fps, max_width, out_dir
        self.writers, self.last_t, self.counts = {}, {}, {}
        if not shutil.which("ffmpeg"):
            raise SystemExit("ffmpeg not found; install it with apt")
        encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        self.have_nvenc = "h264_nvenc" in encoders and shutil.which("nvidia-smi") is not None
        self.have_x264 = "libx264" in encoders
        self.get_logger().info("encoder: " + ("h264_nvenc (GPU)" if self.have_nvenc else "libx264 (CPU)" if self.have_x264 else "mpeg4"))
        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        for t in topics:
            self.create_subscription(Image, t, lambda msg, t=t: self.on_image(t, msg), qos)
        self.get_logger().info(f"recording {topics} to {out_dir} at {fps} fps")

    def on_image(self, topic, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if self.max_width and frame.shape[1] > self.max_width:
            s = self.max_width / frame.shape[1]
            frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        h, w = frame.shape[:2]
        h, w = h - h % 2, w - w % 2                      # yuv420p needs even dimensions
        frame = frame[:h, :w]
        if topic not in self.writers:
            path = os.path.join(self.out_dir, topic.strip("/").replace("/", "_") + ".mp4")
            self.writers[topic] = self._open_ffmpeg(path, w, h)
            self.last_t[topic], self.counts[topic] = t, 0
            self.get_logger().info(f"{topic}: {w}x{h} -> {path}")
        repeats = max(1, int(round((t - self.last_t[topic]) * self.fps))) if self.counts[topic] else 1
        data = frame.tobytes()
        try:
            for _ in range(min(repeats, self.fps * 5)):
                self.writers[topic].stdin.write(data)
                self.counts[topic] += 1
        except BrokenPipeError:
            self.get_logger().error(f"{topic}: ffmpeg exited")
        self.last_t[topic] = t

    def _open_ffmpeg(self, path, w, h):
        # NVENC keeps the encode off the CPU: two software encoders plus the sim overheated the laptop.
        if self.have_nvenc:
            codec = ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "23", "-b:v", "0"]
        elif self.have_x264:
            codec = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "22"]
        else:
            codec = ["-c:v", "mpeg4", "-q:v", "3"]
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{w}x{h}", "-r", str(self.fps), "-i", "-", *codec, "-pix_fmt", "yuv420p",
               "-movflags", "+frag_keyframe+empty_moov+default_base_moof", path]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def close(self):
        for topic, p in self.writers.items():
            try:
                p.stdin.close()
                p.wait(timeout=20)
            except Exception:
                p.kill()
            self.get_logger().info(f"{topic}: {self.counts[topic]} frames written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("topics", nargs="+")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--max-width", type=int, default=1920)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rclpy.init()
    node = Recorder(a.topics, a.out_dir, a.fps, a.max_width)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
