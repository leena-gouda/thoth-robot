#!/usr/bin/env python3
"""
THOTH visitor perception node.

    /image_raw/compressed  (sensor_msgs/CompressedImage, from usb_cam)
        -> VisitorPerception (SCRFD + ViT age + ViT-FER, see perception.py)
        -> /visitor/attributes            (thoth_interfaces/VisitorAttributes)  primary visitor only
        -> /visitor/annotated/compressed  (sensor_msgs/CompressedImage)        boxes + labels, for rqt

Place in: ros2_ws/src/thoth_vision/thoth_vision/visitor_perception_node.py
Run:      ros2 run thoth_vision visitor_perception_node   (inside ~/thoth_venv, see build notes)

Design notes
  * Subscribes to the COMPRESSED topic: raw 1080p frames (~6 MB) stall over DDS; JPEG (~300 KB) runs at 30 Hz.
  * Best-effort, depth-1 subscription: if analysis is busy, old frames are dropped, never queued (no lag).
  * max_rate_hz caps analysis so the GPU is shared with other nodes later (LLM, STT).
  * Primary visitor is "sticky": a new face takes over only if it is switch_ratio x bigger than the
    current one, or the current one leaves. Stops the primary flipping between similar-size faces.
  * Always publishes, also when nobody is in view (face_detected=false, labels "unknown"), so
    subscribers can tell "no visitor" from "node is dead".
"""
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image

from thoth_interfaces.msg import VisitorAttributes
from thoth_vision.perception import UNKNOWN, VisitorPerception


class VisitorPerceptionNode(Node):
    def __init__(self):
        super().__init__('visitor_perception')

        # ---------------- parameters ----------------
        p = self.declare_parameter
        self.image_topic = p('image_topic', '/image_raw/compressed').value
        self.max_rate_hz = float(p('max_rate_hz', 10.0).value)
        self.publish_annotated = bool(p('publish_annotated', True).value)
        self.annotated_width = int(p('annotated_width', 960).value)
        self.raw_view_width = int(p('raw_view_width', 640).value)
        self.switch_ratio = float(p('switch_ratio', 1.2).value)
        device = p('device', 'auto').value
        det_size = int(p('det_size', 640).value)
        min_face_px = int(p('min_face_px', 40).value)
        age_min_conf = float(p('age_min_conf', 0.4).value)
        emo_min_conf = float(p('emo_min_conf', 0.4).value)

        # ---------------- models ----------------
        self.get_logger().info('Loading perception models ...')
        t0 = time.time()
        self.perception = VisitorPerception(device=device, det_size=det_size, min_face_px=min_face_px,
                                            age_min_conf=age_min_conf, emo_min_conf=emo_min_conf)
        self.get_logger().info(
            f'Models loaded in {time.time() - t0:.1f} s on {self.perception.device_name}; '
            f'detector providers: {self.perception.detector_providers}')

        # ---------------- ROS interfaces ----------------
        self.pub_attr = self.create_publisher(VisitorAttributes, '/visitor/attributes', 10)
        # Reliable, depth 1: compatible with both reliable (rqt) and best-effort subscribers.
        # /visitor/annotated/compressed : JPEG, for the web UI later (small, fast).
        # /visitor/annotated/image      : small raw image, for rqt_image_view (always displays).
        self.pub_img = (self.create_publisher(CompressedImage, '/visitor/annotated/compressed', 1)
                        if self.publish_annotated else None)
        self.pub_raw = (self.create_publisher(Image, '/visitor/annotated/image', 1)
                        if self.publish_annotated else None)
        self.sub = self.create_subscription(CompressedImage, self.image_topic, self.on_image,
                                            qos_profile_sensor_data)

        self.primary_id = None
        self.last_proc = 0.0
        self.stats_t0, self.stats_n, self.stats_ms = time.monotonic(), 0, 0.0
        self.get_logger().info(f'Listening on {self.image_topic} (max {self.max_rate_hz:.0f} analyses/s)')

    # ------------------------------------------------------------ callback
    def on_image(self, msg):
        now = time.monotonic()
        if self.max_rate_hz > 0 and now - self.last_proc < 1.0 / self.max_rate_hz:
            return
        self.last_proc = now

        frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warn('Could not decode image', throttle_duration_sec=5.0)
            return

        results = self.perception.process(frame)
        primary = self.select_primary(results)

        out = VisitorAttributes()
        out.header = msg.header
        out.num_faces = min(len(results), 255)
        if primary is not None:
            out.face_detected = True
            out.visitor_id = primary.track_id
            out.age_group = primary.age_group
            out.age_confidence = float(primary.age_confidence)
            out.emotion = primary.emotion
            out.emotion_confidence = float(primary.emotion_confidence)
        else:
            out.face_detected = False
            out.visitor_id = 0
            out.age_group = UNKNOWN
            out.emotion = UNKNOWN
        self.pub_attr.publish(out)

        if self.pub_img is not None and (self.pub_img.get_subscription_count() > 0
                                         or self.pub_raw.get_subscription_count() > 0):
            self.publish_annotated_image(msg.header, frame, results, primary)

        self.log_stats()

    # ------------------------------------------------------------ helpers
    def select_primary(self, results):
        """Biggest face, but keep the current visitor unless someone is clearly bigger."""
        if not results:
            self.primary_id = None
            return None
        biggest = max(results, key=lambda r: r.area)
        current = next((r for r in results if r.track_id == self.primary_id), None)
        if current is None or biggest.area > self.switch_ratio * current.area:
            self.primary_id = biggest.track_id
            return biggest
        return current

    def publish_annotated_image(self, header, frame, results, primary):
        vis = frame.copy()
        scale = frame.shape[1] / 1280
        for r in results:
            is_primary = primary is not None and r.track_id == primary.track_id
            color = (0, 200, 0) if is_primary else (160, 160, 160)
            x1, y1, x2, y2 = r.box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, max(2, int((4 if is_primary else 2) * scale)))
            lock = '' if r.age_locked else ' ...'
            txt = (f'#{r.track_id} {r.age_group} {r.age_confidence:.0%}{lock} | '
                   f'{r.emotion} {r.emotion_confidence:.0%}')
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, max(1, int(2 * scale)))
            ty = max(th + 10, y1 - 10)
            cv2.rectangle(vis, (x1, ty - th - 8), (x1 + tw + 10, ty + 6), color, -1)
            cv2.putText(vis, txt, (x1 + 5, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (0, 0, 0),
                        max(1, int(2 * scale)), cv2.LINE_AA)
        if self.pub_img.get_subscription_count() > 0:
            jpg = self._resize(vis, self.annotated_width)
            ok, buf = cv2.imencode('.jpg', jpg, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                img = CompressedImage()
                img.header = header
                img.format = 'jpeg'
                img.data = buf.tobytes()
                self.pub_img.publish(img)

        if self.pub_raw.get_subscription_count() > 0:
            small = np.ascontiguousarray(self._resize(vis, self.raw_view_width))
            raw = Image()
            raw.header = header
            raw.height, raw.width = small.shape[:2]
            raw.encoding = 'bgr8'
            raw.is_bigendian = 0
            raw.step = raw.width * 3
            raw.data = small.tobytes()
            self.pub_raw.publish(raw)

    @staticmethod
    def _resize(img, width):
        if width and img.shape[1] > width:
            return cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])))
        return img

    def log_stats(self):
        self.stats_n += 1
        self.stats_ms += self.perception.last_timing.get('total_ms', 0.0)
        elapsed = time.monotonic() - self.stats_t0
        if elapsed >= 5.0:
            self.get_logger().info(
                f'{self.stats_n / elapsed:.1f} analyses/s, {self.stats_ms / self.stats_n:.0f} ms each, '
                f'{self.perception.last_timing.get("faces", 0)} face(s) in view, primary #{self.primary_id}')
            self.stats_t0, self.stats_n, self.stats_ms = time.monotonic(), 0, 0.0


def main(args=None):
    rclpy.init(args=args)
    node = VisitorPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
