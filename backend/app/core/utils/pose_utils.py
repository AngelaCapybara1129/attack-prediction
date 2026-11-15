import os
import threading
from pathlib import Path
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[3]

POSE_TASK_CANDIDATES = [
    BASE_DIR / "weights" / "pose_landmarker_lite.task",
    BASE_DIR / "pose_landmarker_lite.task",
]


def _find_pose_task_path() -> Path:
    for p in POSE_TASK_CANDIDATES:
        if p.is_file():
            return p
    return POSE_TASK_CANDIDATES[0]


POSE_TASK_PATH = _find_pose_task_path()

print(f"[POSE] POSE_TASK_PATH = {POSE_TASK_PATH} (exists={POSE_TASK_PATH.is_file()})")

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

_pose_estimator = None
_pose_lock = threading.Lock()

# 低解析度監視器畫面（RWF-2000 實測，327 個人物裁切）：
#   原設定 conf 0.5、不放大 → 偵測率 24%；conf 0.3 + 裁切放大到 256 px 高 → 57%（full/heavy 模型沒有更好）
# MediaPipe 輸出的是裁切內的正規化座標，放大不影響座標換算。
POSE_MIN_CONFIDENCE = float(os.getenv("POSE_MIN_CONFIDENCE", "0.3"))
POSE_UPSCALE_HEIGHT = int(os.getenv("POSE_UPSCALE_HEIGHT", "256"))

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
]


def pose_model_exists() -> bool:
    return POSE_TASK_PATH.is_file()


def get_pose_task_path() -> Path:
    return POSE_TASK_PATH


def get_pose_estimator():
    global _pose_estimator

    if _pose_estimator is None:
        with _pose_lock:
            if _pose_estimator is None:
                if not POSE_TASK_PATH.is_file():
                    raise FileNotFoundError(
                        "找不到 MediaPipe task 檔案，候選路徑如下：\n"
                        + "\n".join(str(p) for p in POSE_TASK_CANDIDATES)
                    )

                model_bytes = POSE_TASK_PATH.read_bytes()

                options = PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_buffer=model_bytes),
                    running_mode=VisionRunningMode.IMAGE,
                    num_poses=1,
                    min_pose_detection_confidence=POSE_MIN_CONFIDENCE,
                    min_pose_presence_confidence=POSE_MIN_CONFIDENCE,
                    min_tracking_confidence=POSE_MIN_CONFIDENCE,
                    output_segmentation_masks=False,
                )
                _pose_estimator = PoseLandmarker.create_from_options(options)
                print(f"[POSE] PoseLandmarker initialized from bytes: {POSE_TASK_PATH}")

    return _pose_estimator


def extract_mediapipe_pose_full(person_crop: Optional[np.ndarray]):
    """回傳 (landmarks 33x3 [x,y,z] 裁切正規化座標, visibility 33)；偵測失敗回傳 (None, None)。"""
    if person_crop is None or person_crop.size == 0:
        return None, None

    try:
        estimator = get_pose_estimator()
    except Exception as e:
        print(f"[WARN] PoseLandmarker 初始化失敗: {e}")
        return None, None

    try:
        h = person_crop.shape[0]
        if POSE_UPSCALE_HEIGHT and 0 < h < POSE_UPSCALE_HEIGHT:
            s = POSE_UPSCALE_HEIGHT / h
            person_crop = cv2.resize(person_crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        image_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image_rgb))
        with _pose_lock:
            detection_result = estimator.detect(mp_image)
    except Exception as e:
        print(f"[WARN] PoseLandmarker detect 失敗: {e}")
        return None, None

    if not detection_result.pose_landmarks:
        return None, None

    landmarks = []
    visibility = []
    for lm in detection_result.pose_landmarks[0]:
        landmarks.append([float(lm.x), float(lm.y), float(lm.z)])
        vis = getattr(lm, "visibility", None)
        visibility.append(float(vis) if vis is not None else 1.0)

    return landmarks, visibility


def extract_mediapipe_pose(person_crop: Optional[np.ndarray]):
    landmarks, _ = extract_mediapipe_pose_full(person_crop)
    return landmarks


def draw_pose_on_frame(
    frame: np.ndarray,
    box: List[int],
    landmarks: Optional[List[List[float]]],
    color=(0, 255, 255),
    point_color=(255, 255, 0),
    line_thickness=2,
):
    if frame is None or landmarks is None or len(landmarks) == 0:
        return

    x1, y1, x2, y2 = box
    h_img, w_img = frame.shape[:2]
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    radius = max(2, min(4, int(min(w, h) * 0.015)))

    pts = []
    for lm in landmarks:
        px = int(x1 + lm[0] * w)
        py = int(y1 + lm[1] * h)
        px = max(0, min(w_img - 1, px))
        py = max(0, min(h_img - 1, py))
        pts.append((px, py))

    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], color, line_thickness)

    for px, py in pts:
        cv2.circle(frame, (px, py), radius, point_color, -1)