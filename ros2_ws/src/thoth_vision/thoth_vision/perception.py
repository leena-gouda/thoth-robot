"""
THOTH visitor perception (ROS-independent).

Pipeline (models chosen in the CV week-1 comparison, provisional):
    frame (BGR) -> SCRFD-10G face detection (InsightFace buffalo_l, detection only)
                -> per-face tracking (IoU between frames)
                -> age group: nateraw/vit-age-classifier (FairFace), estimated on the first
                   AGE_SAMPLES analyses of a person, then LOCKED
                -> emotion:   trpakov/vit-face-expression (FER2013), smoothed over the last
                   EMO_HISTORY analyses

Ported from Leena's live_demo.py v2. There are NO ROS imports in this file on purpose:
it is tested on its own first (scripts/test_perception.py), then wrapped by the ROS 2 node.

Output labels match thoth_interfaces/msg/VisitorAttributes:
    age_group: child | teen | adult | senior | unknown
    emotion:   happy | sad | neutral | excited | unknown   ("bored" needs an arousal model; not yet)
"""
import re
import time
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

UNKNOWN = "unknown"

AGE_GROUPS = [("child", 0, 12), ("teen", 13, 19), ("adult", 20, 59), ("senior", 60, 200)]

# Raw FER labels -> THOTH classes (same mapping as live_demo.py v2).
# angry / fear / disgust are folded into "sad"; "bored" is not produced yet.
EMOTION_MAP = {
    "happy": "happy",
    "surprise": "excited",
    "neutral": "neutral",
    "contempt": "neutral",
    "sad": "sad",
    "angry": "sad",
    "anger": "sad",
    "fear": "sad",
    "disgust": "sad",
}


# ---------------------------------------------------------------- helpers
def bucket_mid(label):
    """'20-29' -> 24.5, 'more than 70' -> 75, '3-9' -> 6"""
    nums = [int(n) for n in re.findall(r"\d+", label)]
    if not nums:
        return 30.0
    return nums[0] + 5.0 if len(nums) == 1 else (nums[0] + nums[1]) / 2


def age_to_group(age):
    for name, _lo, hi in AGE_GROUPS:
        if age < hi + 1:
            return name
    return "senior"


def crop_with_margin(img, box, margin):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    return img[int(max(0, y1 - margin * bh)):int(min(h, y2 + margin * bh)),
               int(max(0, x1 - margin * bw)):int(min(w, x2 + margin * bw))]


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _as_batch(res, n):
    """HF pipelines return [dicts] for a single image and [[dicts], ...] for a batch; normalise."""
    if n == 1 and res and isinstance(res[0], dict):
        return [res]
    return res


# ---------------------------------------------------------------- result type
@dataclass
class FaceResult:
    track_id: int
    box: tuple                 # (x1, y1, x2, y2) in frame pixels
    age_group: str             # child | teen | adult | senior | unknown
    age_confidence: float      # 0..1, mean probability of the chosen group
    age_estimate: float        # probability-weighted age in years (for logs/debug only)
    age_locked: bool           # True once AGE_SAMPLES estimates were collected
    emotion: str               # happy | sad | neutral | excited | unknown
    emotion_confidence: float  # 0..1, mean score of the winning label over the history

    @property
    def area(self):
        x1, y1, x2, y2 = self.box
        return (x2 - x1) * (y2 - y1)


# ---------------------------------------------------------------- main class
class VisitorPerception:
    def __init__(self,
                 device="auto",          # "auto" | "cuda" | "cpu"
                 det_size=640,           # detector input; 640 finds far faces, 320 is faster on CPU
                 det_threshold=0.5,
                 min_face_px=40,         # ignore faces smaller than this (too far to judge)
                 max_faces=6,            # analyse at most the N biggest faces
                 crop_margin=0.15,
                 age_samples=5,          # age estimates per person before locking
                 emo_history=3,          # emotion smoothing window
                 age_min_conf=0.4,       # below this -> age_group "unknown"
                 emo_min_conf=0.4,       # below this -> emotion "unknown"
                 track_timeout_s=1.5):   # forget a person not seen for this long
        import torch
        import onnxruntime as ort
        from insightface.app import FaceAnalysis
        from transformers import pipeline

        if device == "auto":
            use_gpu = torch.cuda.is_available()
        else:
            use_gpu = device == "cuda"
        self.device_name = torch.cuda.get_device_name(0) if use_gpu else "CPU"

        # Let onnxruntime-gpu find the CUDA/cuDNN libraries shipped with the torch wheel.
        if use_gpu and hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:
                pass
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if use_gpu and "CUDAExecutionProvider" in ort.get_available_providers()
                     else ["CPUExecutionProvider"])

        self.detector = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"], providers=providers)
        self.detector.prepare(ctx_id=0 if providers[0].startswith("CUDA") else -1,
                              det_size=(det_size, det_size), det_thresh=det_threshold)
        try:
            self.detector_providers = self.detector.models["detection"].session.get_providers()
        except Exception:
            self.detector_providers = providers

        dev = 0 if use_gpu else -1
        self.age_pipe = pipeline("image-classification", model="nateraw/vit-age-classifier", device=dev)
        self.emo_pipe = pipeline("image-classification", model="trpakov/vit-face-expression", device=dev)

        self.min_face_px = min_face_px
        self.max_faces = max_faces
        self.crop_margin = crop_margin
        self.age_samples = age_samples
        self.emo_history = emo_history
        self.age_min_conf = age_min_conf
        self.emo_min_conf = emo_min_conf
        self.track_timeout_s = track_timeout_s

        self.tracks = {}   # id -> {"box", "last_seen", "ages": [(age, {group: prob})], "emos": deque}
        self.next_id = 1
        self.last_timing = {}  # ms of the last process() call: det, age, emo, total

    # ------------------------------------------------------------ public API
    def reset(self):
        """Forget everyone (ages will be re-estimated)."""
        self.tracks.clear()

    def process(self, frame_bgr, now=None):
        """Analyse one BGR frame. Returns a list of FaceResult, biggest face first."""
        from PIL import Image
        import cv2

        now = time.time() if now is None else now
        t0 = time.perf_counter()

        faces = self.detector.get(frame_bgr)
        t_det = time.perf_counter()

        boxes = [tuple(int(v) for v in f.bbox) for f in faces]
        boxes = [b for b in boxes if min(b[2] - b[0], b[3] - b[1]) >= self.min_face_px]
        boxes = sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))[:self.max_faces]
        ids = self._match(boxes, now)
        crops = [Image.fromarray(cv2.cvtColor(crop_with_margin(frame_bgr, b, self.crop_margin),
                                              cv2.COLOR_BGR2RGB)) for b in boxes]

        # AGE: only for people whose age is not locked yet
        need = [i for i, tid in enumerate(ids) if len(self.tracks[tid]["ages"]) < self.age_samples]
        if need:
            res = _as_batch(self.age_pipe([crops[i] for i in need], top_k=None), len(need))
            for i, r in zip(need, res):
                self.tracks[ids[i]]["ages"].append(self._age_sample(r))
        t_age = time.perf_counter()

        # EMOTION: every call, for everyone
        if crops:
            res = _as_batch(self.emo_pipe(crops, top_k=None), len(crops))
            for tid, r in zip(ids, res):
                top = max(r, key=lambda d: d["score"])
                label = EMOTION_MAP.get(top["label"].strip().lower(), UNKNOWN)
                self.tracks[tid]["emos"].append((label, float(top["score"])))
        t_emo = time.perf_counter()

        self.last_timing = {
            "det_ms": (t_det - t0) * 1000,
            "age_ms": (t_age - t_det) * 1000,
            "emo_ms": (t_emo - t_age) * 1000,
            "total_ms": (t_emo - t0) * 1000,
            "faces": len(boxes),
        }
        return [self._result(tid) for tid in ids]

    @staticmethod
    def primary(results):
        """The visitor THOTH talks to: the biggest (closest) face, or None."""
        return max(results, key=lambda r: r.area) if results else None

    # ------------------------------------------------------------ internals
    def _match(self, boxes, now):
        """Assign each box to an existing track (same person) or a new one."""
        ids, used = [], set()
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
                self.tracks[best] = {"ages": [], "emos": deque(maxlen=self.emo_history)}
            self.tracks[best]["box"], self.tracks[best]["last_seen"] = b, now
            used.add(best)
            ids.append(best)
        for tid in [k for k, t in self.tracks.items() if now - t["last_seen"] > self.track_timeout_s]:
            del self.tracks[tid]
        return ids

    @staticmethod
    def _age_sample(prob_list):
        """All age buckets -> (expected age, probability per THOTH age group)."""
        total = max(sum(d["score"] for d in prob_list), 1e-6)
        exp_age = sum(bucket_mid(d["label"]) * d["score"] for d in prob_list) / total
        group_probs = {}
        for d in prob_list:
            g = age_to_group(bucket_mid(d["label"]))
            group_probs[g] = group_probs.get(g, 0.0) + d["score"] / total
        return exp_age, group_probs

    def _result(self, tid):
        t = self.tracks[tid]
        if t["ages"]:
            mean_age = float(np.mean([a for a, _ in t["ages"]]))
            group = age_to_group(mean_age)
            age_conf = float(np.mean([gp.get(group, 0.0) for _, gp in t["ages"]]))
            if age_conf < self.age_min_conf:
                group = UNKNOWN
        else:
            mean_age, group, age_conf = 0.0, UNKNOWN, 0.0

        if t["emos"]:
            emo = Counter(e for e, _ in t["emos"]).most_common(1)[0][0]
            emo_conf = float(np.mean([c for e, c in t["emos"] if e == emo]))
            if emo == UNKNOWN or emo_conf < self.emo_min_conf:
                emo = UNKNOWN
        else:
            emo, emo_conf = UNKNOWN, 0.0

        return FaceResult(track_id=tid, box=t["box"], age_group=group, age_confidence=age_conf,
                          age_estimate=mean_age, age_locked=len(t["ages"]) >= self.age_samples,
                          emotion=emo, emotion_confidence=emo_conf)
