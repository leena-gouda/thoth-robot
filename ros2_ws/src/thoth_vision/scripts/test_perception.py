#!/usr/bin/env python3
"""
Standalone test for thoth_vision/perception.py (no ROS needed).

Place this file in:  ros2_ws/src/thoth_vision/scripts/test_perception.py

Usage (inside the thoth venv; stop the usb_cam launch first, only one program can open the camera):
    python3 test_perception.py --image some_face.jpg
    python3 test_perception.py --camera /dev/thoth_camera --seconds 30
    python3 test_perception.py --camera /dev/thoth_camera --seconds 30 --save-dir snaps

Prints one line per second for the primary (biggest) face, then a timing summary.
--save-dir writes an annotated frame every 2 s (no GUI window needed).
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

# Import perception.py from the package folder without needing colcon build.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "thoth_vision"))
from perception import VisitorPerception  # noqa: E402

WARMUP_CALLS = 5  # first GPU calls are slow; excluded from the timing summary


def annotate(frame, results):
    vis = frame.copy()
    for r in results:
        x1, y1, x2, y2 = r.box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 3)
        lock = "" if r.age_locked else " (estimating)"
        txt = f"#{r.track_id} {r.age_group} {r.age_confidence:.0%}{lock} | {r.emotion} {r.emotion_confidence:.0%}"
        cv2.putText(vis, txt, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2, cv2.LINE_AA)
    return vis


def describe(r):
    if r is None:
        return "no face"
    lock = "locked" if r.age_locked else "estimating"
    return (f"#{r.track_id} age={r.age_group} ({r.age_confidence:.2f}, ~{r.age_estimate:.0f}y, {lock}) "
            f"emotion={r.emotion} ({r.emotion_confidence:.2f})")


def summary(p, timings, frames_read, seconds):
    print("\n=== TIMING SUMMARY ===")
    print(f"Device: {p.device_name} | detector providers: {p.detector_providers}")
    t = timings[WARMUP_CALLS:] or timings
    if not t:
        print("No analyses recorded.")
        return
    for key in ("det_ms", "age_ms", "emo_ms", "total_ms"):
        vals = [x[key] for x in t]
        print(f"{key:9s} mean {np.mean(vals):7.1f} ms   p95 {np.percentile(vals, 95):7.1f} ms")
    mean_total = np.mean([x["total_ms"] for x in t])
    print(f"Analysis rate: {1000 / mean_total:.1f} per second (target >= 5)")
    print(f"Average faces per analysis: {np.mean([x['faces'] for x in t]):.1f}")
    if seconds:
        print(f"Camera frames read: {frames_read} in {seconds:.0f} s")


def run_image(p, path):
    frame = cv2.imread(path)
    if frame is None:
        sys.exit(f"Could not read image: {path}")
    timings = []
    for i in range(10):  # 10 calls: age locks after 5, the rest measure steady-state speed
        results = p.process(frame)
        timings.append(p.last_timing)
        print(f"call {i + 1:2d}: {len(results)} face(s) | primary: {describe(p.primary(results))} "
              f"| {p.last_timing['total_ms']:.0f} ms")
    out = os.path.splitext(path)[0] + "_annotated.jpg"
    cv2.imwrite(out, annotate(frame, results))
    print("Annotated image saved to", out)
    summary(p, timings, 0, 0)


def run_camera(p, device, width, height, seconds, save_dir):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # must be set before the size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always analyse a fresh frame, not a queued old one
    if not cap.isOpened():
        sys.exit(f"Could not open {device}. Is the usb_cam launch still running?")
    print(f"Camera {device}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    timings, frames_read = [], 0
    t_start = t_print = t_save = time.time()
    while time.time() - t_start < seconds:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed")
            break
        frames_read += 1
        results = p.process(frame)
        timings.append(p.last_timing)
        now = time.time()
        if now - t_print >= 1.0:
            print(f"[{now - t_start:5.1f}s] {len(results)} face(s) | primary: {describe(p.primary(results))} "
                  f"| {p.last_timing['total_ms']:.0f} ms")
            t_print = now
        if save_dir and now - t_save >= 2.0:
            cv2.imwrite(os.path.join(save_dir, f"frame_{now - t_start:05.1f}s.jpg"), annotate(frame, results))
            t_save = now
    cap.release()
    summary(p, timings, frames_read, time.time() - t_start)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="test on one image file")
    ap.add_argument("--camera", default="/dev/thoth_camera")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    print("Loading models (first run downloads them) ...")
    p = VisitorPerception(device=args.device)
    print(f"Loaded. Device: {p.device_name} | detector providers: {p.detector_providers}")

    if args.image:
        run_image(p, args.image)
    else:
        run_camera(p, args.camera, args.width, args.height, args.seconds, args.save_dir)


if __name__ == "__main__":
    main()
