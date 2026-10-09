"""
THOTH – Live demo v2: Rapoo camera -> SCRFD face detection -> ViT age group + ViT-FER emotion

Pipeline (the models chosen in the week-1 comparison):
    camera (ffmpeg, MJPG)  ->  SCRFD-10G (InsightFace)  finds every face
                           ->  ViT age classifier (nateraw/vit-age-classifier)  -> Child / Teen / Adult / Senior
                           ->  ViT-FER (trpakov/vit-face-expression)            -> Happy / Sad / Neutral / Excited
    (Bored needs the arousal score from another model – not in this demo yet.)

v2 changes (after the first live test on CPU: age group flipping, emotion reacting late, 1.8 s per analysis)
  * AGE: estimated only for the first few analyses of each person, then LOCKED (a person's age doesn't change).
         Uses the expected age from all bucket probabilities (e.g. 40% "10-19" + 60% "20-29" -> ~20.5)
         averaged over those analyses, instead of the single top bucket -> no more Teen/Adult flipping.
         Also saves the age model's time on every later frame -> faster emotion updates.
  * EMOTION: smoothed over the last 3 analyses only (was 7) -> reacts much faster.
  * CPU mode: camera 720p and detector input 320 px (GPU keeps 1080p / 640 px).

Setup (once, inside the thoth-cv venv):
    pip install insightface onnxruntime transformers torch pillow imageio-ffmpeg

Run (ffmpeg_camera.py must be in the same folder):
    python live_demo.py              -> automatic settings (CPU: 720p + fast detector, GPU: 1080p)
    python live_demo.py 1080         -> force 1080p camera
    python live_demo.py det640       -> force full-size detector (better for far faces, slower on CPU)

Keys:  q = quit   s = save annotated snapshot   v = start/stop recording   r = reset (re-estimate ages)
At exit it prints average timings — copy them into the doc.
"""
import os
import re
import sys
import threading
import time
from collections import Counter, deque
from datetime import datetime

import cv2
import numpy as np

from ffmpeg_camera import FFmpegCamera, CAMERA_NAME

print("Loading models (first run downloads them, this can take a while) ...")
import onnxruntime as ort
import torch
from insightface.app import FaceAnalysis
from PIL import Image
from transformers import pipeline

USE_GPU = torch.cuda.is_available()
DEVICE_NAME = torch.cuda.get_device_name(0) if USE_GPU else "CPU"

# ---------------- settings ----------------
args = [a.lower() for a in sys.argv[1:]]
if "1080" in args or (USE_GPU and "720" not in args):
    RES = (1920, 1080)
else:
    RES = (1280, 720)
DET_SIZE = (640, 640) if (USE_GPU or "det640" in args) else (320, 320)
DET_THRESHOLD = 0.5
MIN_FACE_PX = 40            # ignore faces smaller than this (too far to judge age/emotion)
MAX_FACES = 6               # analyse at most the N biggest (closest) faces
CROP_MARGIN = 0.15          # same margin as in the Colab comparison
AGE_SAMPLES = 5             # age estimated on the first N analyses of a person, then locked
EMO_HISTORY = 3             # emotion smoothed over the last N analyses
OUT_DIR = "demo_output"

AGE_GROUPS = [("Child", 0, 12), ("Teen", 13, 19), ("Adult", 20, 59), ("Senior", 60, 200)]
COLORS = {"Child": (255, 170, 0), "Teen": (0, 200, 255), "Adult": (0, 200, 0), "Senior": (200, 0, 200),
          "...": (180, 180, 180)}
EMO_COLORS = {"Happy": (0, 200, 0), "Excited": (0, 200, 255), "Neutral": (200, 200, 200),
              "Bored": (255, 170, 0), "Sad": (0, 0, 255)}


# ---------------- helpers ----------------
def bucket_mid(label):
    """'20-29' -> 24.5, 'more than 70' -> 75, '3-9' -> 6"""
    nums = [int(n) for n in re.findall(r"\d+", label)]
    if not nums:
        return 30.0
    return nums[0] + 5.0 if len(nums) == 1 else (nums[0] + nums[1]) / 2


def expected_age(prob_list):
    """prob_list: [{'label': '20-29', 'score': 0.6}, ...] (all buckets) -> probability-weighted age"""
    total = sum(d["score"] for d in prob_list)
    return sum(bucket_mid(d["label"]) * d["score"] for d in prob_list) / max(total, 1e-6)


def age_to_group(age):
    for name, lo, hi in AGE_GROUPS:
        if age < hi + 1:
            return name
    return "Senior"


EMO_ALIASES = {"angry": "anger", "happy": "happy", "sad": "sad", "surprise": "surprise",
               "neutral": "neutral", "fear": "fear", "disgust": "disgust", "contempt": "contempt"}


def emotion_to_5(raw):
    raw = EMO_ALIASES.get(raw.lower(), raw.lower())
    if raw == "happy":
        return "Happy"
    if raw == "surprise":
        return "Excited"
    if raw in ("neutral", "contempt"):
        return "Neutral"
    return "Sad"  # anger, fear, disgust, sad


def crop_with_margin(img, box, m=CROP_MARGIN):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    return img[int(max(0, y1 - m * bh)):int(min(h, y2 + m * bh)),
               int(max(0, x1 - m * bw)):int(min(w, x2 + m * bw))]


def to_pil(img_bgr):
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0


# ---------------- load models ----------------
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
    if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
print("Running on:", DEVICE_NAME, "| camera", RES, "| detector input", DET_SIZE)

detector = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"], providers=providers)
detector.prepare(ctx_id=0 if USE_GPU else -1, det_size=DET_SIZE, det_thresh=DET_THRESHOLD)
age_pipe = pipeline("image-classification", model="nateraw/vit-age-classifier", device=0 if USE_GPU else -1)
emo_pipe = pipeline("image-classification", model="trpakov/vit-face-expression", device=0 if USE_GPU else -1)
print("Models loaded.")


# ---------------- background analysis thread ----------------
class Analyzer:
    def __init__(self, cam):
        self.cam = cam
        self.tracks = {}          # id -> dict(box, ages[list of expected ages], emos deque, last_seen)
        self.next_id = 1
        self.lock = threading.Lock()
        self.running = True
        self.stats = {"det": [], "age": [], "emo": [], "total": [], "faces": [], "age_runs": 0}
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def match(self, boxes):
        """Assign each box to an existing track (same person) or a new one. Returns track ids."""
        now, ids, used = time.time(), [], set()
        with self.lock:
            for b in boxes:
                best, best_iou = None, 0.3
                for tid, t in self.tracks.items():
                    if tid in used:
                        continue
                    v = iou(b, t["box"])
                    if v > best_iou:
                        best, best_iou = tid, v
                if best is None:
                    best = self.next_id
                    self.next_id += 1
                    self.tracks[best] = {"ages": [], "emos": deque(maxlen=EMO_HISTORY)}
                self.tracks[best]["box"], self.tracks[best]["last_seen"] = b, now
                used.add(best)
                ids.append(best)
            for tid in [k for k, t in self.tracks.items() if now - t["last_seen"] > 1.5]:
                del self.tracks[tid]   # person left the view
        return ids

    def loop(self):
        while self.running:
            ok, frame = self.cam.read()
            if not ok:
                time.sleep(0.01)
                continue
            t0 = time.perf_counter()
            faces = detector.get(frame)
            t1 = time.perf_counter()

            boxes = [f.bbox.astype(int).tolist() for f in faces]
            boxes = [b for b in boxes if min(b[2] - b[0], b[3] - b[1]) >= MIN_FACE_PX]
            boxes = sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))[:MAX_FACES]
            ids = self.match(boxes)
            crops = [to_pil(crop_with_margin(frame, b)) for b in boxes]

            # AGE: only for people whose age is not locked yet
            t2 = t1
            need = [i for i, tid in enumerate(ids) if len(self.tracks.get(tid, {}).get("ages", [])) < AGE_SAMPLES]
            if need:
                res = age_pipe([crops[i] for i in need], top_k=None)
                if len(need) == 1 and res and isinstance(res[0], dict):
                    res = [res]
                with self.lock:
                    for i, r in zip(need, res):
                        if ids[i] in self.tracks:
                            self.tracks[ids[i]]["ages"].append(expected_age(r))
                t2 = time.perf_counter()
                self.stats["age"].append((t2 - t1) * 1000)
                self.stats["age_runs"] += 1

            # EMOTION: every analysis, for everyone
            t3 = t2
            if crops:
                res = emo_pipe(crops, top_k=1)
                res = [r[0] if isinstance(r, list) else r for r in res]
                with self.lock:
                    for tid, e in zip(ids, res):
                        if tid in self.tracks:
                            self.tracks[tid]["emos"].append((emotion_to_5(e["label"]), float(e["score"])))
                t3 = time.perf_counter()
                self.stats["emo"].append((t3 - t2) * 1000)

            s = self.stats
            s["det"].append((t1 - t0) * 1000)
            s["faces"].append(len(boxes))
            s["total"].append((time.perf_counter() - t0) * 1000)

    def reset(self):
        with self.lock:
            self.tracks.clear()

    def snapshot(self):
        """For drawing: list of (id, box, age_text, age_group, emotion, emo_conf)."""
        out, now = [], time.time()
        with self.lock:
            for tid, t in self.tracks.items():
                if now - t["last_seen"] > 0.7 or "box" not in t:
                    continue
                if t["ages"]:
                    a = float(np.mean(t["ages"]))
                    grp = age_to_group(a)
                    locked = len(t["ages"]) >= AGE_SAMPLES
                    age_text = f"{grp} (~{a:.0f})" + ("" if locked else f" {len(t['ages'])}/{AGE_SAMPLES}")
                else:
                    grp, age_text = "...", "age ..."
                if t["emos"]:
                    emo = Counter(x[0] for x in t["emos"]).most_common(1)[0][0]
                    conf = float(np.mean([x[1] for x in t["emos"] if x[0] == emo]))
                else:
                    emo, conf = "...", 0.0
                out.append((tid, t["box"], age_text, grp, emo, conf))
        return out


def draw(frame, items, info_lines):
    vis = frame.copy()
    scale = frame.shape[1] / 1280
    fs, th_ = 0.8 * scale, max(2, int(2 * scale))
    for tid, (x1, y1, x2, y2), age_text, grp, emo, conf in items:
        c = COLORS.get(grp, (0, 255, 0))
        cv2.rectangle(vis, (x1, y1), (x2, y2), c, max(2, int(3 * scale)))
        lines = [(f"#{tid} {age_text}", c), (f"{emo} {conf:.0%}" if emo != "..." else "emotion ...",
                                              EMO_COLORS.get(emo, (180, 180, 180)))]
        y = y1 - int(10 * scale)
        for txt, col in reversed(lines):
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th_)
            yy = max(th + 8, y)
            cv2.rectangle(vis, (x1, yy - th - 8), (x1 + tw + 10, yy + 6), col, -1)
            cv2.putText(vis, txt, (x1 + 5, yy), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th_, cv2.LINE_AA)
            y = yy - th - 14
    for i, txt in enumerate(info_lines):
        yy = int((35 + i * 32) * scale)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, th_)
        cv2.rectangle(vis, (0, yy - th - 8), (tw + 20, yy + 8), (0, 0, 0), -1)
        cv2.putText(vis, txt, (10, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (0, 255, 0), th_, cv2.LINE_AA)
    return vis


def avg(x, n=30):
    return float(np.mean(x[-n:])) if x else 0.0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cam = FFmpegCamera(width=RES[0], height=RES[1])
    ok, _ = cam.read()
    if not ok:
        print("Camera did not start:", cam.error_text()[:300])
        return
    an = Analyzer(cam)
    an.thread.start()

    win = "THOTH live demo"
    writer, raised = None, False
    last_id, t_last, cam_fps = -1, time.time(), 0.0
    print("\nRunning. q = quit, s = snapshot, v = start/stop recording, r = reset ages")
    while True:
        ok, frame = cam.read()
        if not ok:
            print("Camera stopped:", cam.error_text()[:300])
            break
        if cam.frame_id != last_id:
            now = time.time()
            cam_fps = 0.9 * cam_fps + 0.1 / max(now - t_last, 1e-6)
            t_last, last_id = now, cam.frame_id

        s = an.stats
        analysis_fps = 1000 / avg(s["total"]) if s["total"] else 0
        info = [f"{RES[0]}x{RES[1]} cam {cam_fps:4.1f} FPS | emotion updates {analysis_fps:4.1f}/s | {DEVICE_NAME}",
                f"detect {avg(s['det']):4.0f} ms  emotion {avg(s['emo']):4.0f} ms  age {avg(s['age']):4.0f} ms "
                f"(only until locked)",
                f"faces: {s['faces'][-1] if s['faces'] else 0}" + ("   REC" if writer else "")]
        vis = draw(frame, an.snapshot(), info)
        disp = cv2.resize(vis, (1280, int(1280 * vis.shape[0] / vis.shape[1]))) if vis.shape[1] > 1280 else vis
        cv2.imshow(win, disp)
        if not raised:
            cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
            raised = True
        if writer is not None:
            writer.write(disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            an.reset()
            print("Reset: ages will be re-estimated")
        if key == ord("s"):
            p = os.path.join(OUT_DIR, datetime.now().strftime("snap_%Y%m%d_%H%M%S.jpg"))
            cv2.imwrite(p, vis)
            print("Saved", p)
        if key == ord("v"):
            if writer is None:
                p = os.path.join(OUT_DIR, datetime.now().strftime("demo_%Y%m%d_%H%M%S.mp4"))
                writer = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 15, (disp.shape[1], disp.shape[0]))
                print("Recording ->", p)
            else:
                writer.release()
                writer = None
                print("Recording stopped")

    an.running = False
    time.sleep(0.3)
    if writer is not None:
        writer.release()
    cam.release()
    cv2.destroyAllWindows()

    s = an.stats
    m = lambda x: float(np.mean(x)) if x else 0.0
    lines = [
        f"Device: {DEVICE_NAME}   Camera: {CAMERA_NAME} {RES[0]}x{RES[1]} MJPG   Detector input: {DET_SIZE[0]}px",
        f"Frames analysed: {len(s['total'])}   average faces per frame: {m(s['faces']):.1f}",
        f"Face detection (SCRFD-10G): {m(s['det']):.0f} ms per frame",
        f"Emotion (ViT-FER):          {m(s['emo']):.0f} ms per frame (all faces together)",
        f"Age (ViT age classifier):   {m(s['age']):.0f} ms per run, ran on {s['age_runs']} frames only "
        f"(locked after {AGE_SAMPLES} estimates per person)",
        f"Total analysis:             {m(s['total']):.0f} ms per frame "
        f"= {1000 / m(s['total']) if s['total'] else 0:.1f} emotion updates per second",
    ]
    print("\n=== TIMING SUMMARY ===")
    print("\n".join(lines))
    with open(os.path.join(OUT_DIR, "timing_summary_v2.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"(also saved to {OUT_DIR}/timing_summary_v2.txt)")


if __name__ == "__main__":
    main()