"""
THOTH – Read the Rapoo C260 in MJPG mode through ffmpeg (Windows).

Why: OpenCV on Windows always opens this camera in uncompressed YUY2 mode (1080p = 5 FPS, 720p = 10 FPS).
The camera itself supports MJPG at 1920x1080 / 30 FPS, so we let ffmpeg request MJPG and hand us normal
OpenCV (BGR numpy) frames. The rest of the pipeline uses this class exactly like cv2.VideoCapture.

Setup (once):
    pip install opencv-python imageio-ffmpeg

Run:
    python ffmpeg_camera.py            -> FPS test at 1080p / 720p / 480p, then live preview
    python ffmpeg_camera.py preview    -> skip the test, go straight to the live preview

Keys in the preview window:  s = snapshot, r = next resolution, q = quit
"""
import os
import subprocess
import sys
import threading
import time
from datetime import datetime

import cv2
import imageio_ffmpeg
import numpy as np

CAMERA_NAME = "Rapoo Camera"          # name exactly as shown by list_camera_formats.py
RESOLUTIONS = [(1920, 1080), (1280, 720), (640, 480)]
SNAP_DIR = "snapshots"


class FFmpegCamera:
    """Drop-in replacement for cv2.VideoCapture: cam.read() -> (ok, frame_bgr)."""

    def __init__(self, name=CAMERA_NAME, width=1920, height=1080, fps=30):
        self.w, self.h = width, height
        self.frame_bytes = width * height * 3
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
               "-f", "dshow", "-rtbufsize", "100M",
               "-vcodec", "mjpeg",                         # ask the camera for compressed MJPG
               "-video_size", f"{width}x{height}", "-framerate", str(fps),
               "-i", f"video={name}",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]  # give us raw BGR frames for OpenCV
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     bufsize=self.frame_bytes * 2)
        self.latest, self.frame_id, self.running = None, 0, True
        self.lock = threading.Lock()
        # Background thread always keeps only the NEWEST frame, so a slow model never sees old (laggy) frames
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            raw = self.proc.stdout.read(self.frame_bytes)
            if len(raw) < self.frame_bytes:
                self.running = False
                break
            frame = np.frombuffer(raw, np.uint8).reshape(self.h, self.w, 3)
            with self.lock:
                self.latest, self.frame_id = frame, self.frame_id + 1

    def read(self, timeout=5.0):
        t0 = time.time()
        while self.latest is None and self.running and time.time() - t0 < timeout:
            time.sleep(0.005)
        with self.lock:
            if self.latest is None:
                return False, None
            return True, self.latest.copy()

    def camera_fps(self, seconds=3.0):
        """How many NEW frames the camera delivers per second."""
        self.read()
        start_id, t0 = self.frame_id, time.time()
        time.sleep(seconds)
        return (self.frame_id - start_id) / (time.time() - t0)

    def error_text(self):
        try:
            return self.proc.stderr.read().decode(errors="replace") if not self.running else ""
        except Exception:
            return ""

    def release(self):
        self.running = False
        try:
            self.proc.kill()
        except Exception:
            pass


def fps_test():
    print(f"Camera: {CAMERA_NAME}  (format requested: MJPG)\n")
    print(f"{'resolution':>12} | {'real FPS':>8}")
    results = []
    for w, h in RESOLUTIONS:
        cam = FFmpegCamera(width=w, height=h)
        ok, _ = cam.read()
        if not ok:
            print(f"{w}x{h:<7} |  FAILED  {cam.error_text().strip()[:200]}")
            cam.release()
            continue
        fps = cam.camera_fps()
        print(f"{w}x{h:<7} | {fps:8.1f}")
        results.append((w, h, fps))
        cam.release()
        time.sleep(0.5)
    return results


def preview(res_i=0):
    os.makedirs(SNAP_DIR, exist_ok=True)
    w, h = RESOLUTIONS[res_i]
    cam = FFmpegCamera(width=w, height=h)
    print("\nLive preview: s = snapshot, r = next resolution, q = quit")
    last_id, t_last, fps_smooth, raised = -1, time.time(), 0.0, False
    while True:
        ok, frame = cam.read()
        if not ok:
            print("Camera stopped:", cam.error_text()[:300])
            break
        if cam.frame_id != last_id:  # update FPS only on new frames
            now = time.time()
            fps_smooth = 0.9 * fps_smooth + 0.1 / max(now - t_last, 1e-6)
            t_last, last_id = now, cam.frame_id
        view = frame if w <= 1280 else cv2.resize(frame, (1280, 720))
        info = f"{w}x{h} MJPG  {fps_smooth:4.1f} FPS"
        cv2.rectangle(view, (0, 0), (len(info) * 13 + 20, 38), (0, 0, 0), -1)
        cv2.putText(view, info, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("THOTH camera (MJPG)", view)
        if not raised:
            cv2.setWindowProperty("THOTH camera (MJPG)", cv2.WND_PROP_TOPMOST, 1)
            raised = True
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            path = os.path.join(SNAP_DIR, datetime.now().strftime("snap_%Y%m%d_%H%M%S.jpg"))
            cv2.imwrite(path, frame)
            print("Saved", path)
        if key == ord("r"):
            cam.release()
            res_i = (res_i + 1) % len(RESOLUTIONS)
            w, h = RESOLUTIONS[res_i]
            cam = FFmpegCamera(width=w, height=h)
            print(f"Switched to {w}x{h}")
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if "preview" not in sys.argv[1:]:
        fps_test()
    preview()