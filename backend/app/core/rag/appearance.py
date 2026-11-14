"""人物外觀描述（上身/下身主色），讓值勤人員能以「穿黑衣」等文字檢索歷史事件。"""
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

COLOR_NAMES_ZH = {
    "black": "黑", "white": "白", "gray": "灰", "red": "紅", "orange": "橘", "yellow": "黃",
    "green": "綠", "cyan": "青", "blue": "藍", "purple": "紫", "pink": "粉", "brown": "棕",
}


def _classify_hsv(h: float, s: float, v: float) -> str:
    # OpenCV HSV：H 0~180，S/V 0~255
    if v < 50:
        return "black"
    if s < 40:
        if v > 200:
            return "white"
        return "gray" if v > 80 else "black"
    if (h < 10 or h >= 170):
        return "pink" if v > 200 and s < 120 else "red"
    if h < 22:
        return "brown" if v < 150 else "orange"
    if h < 35:
        return "yellow"
    if h < 85:
        return "green"
    if h < 100:
        return "cyan"
    if h < 130:
        return "blue"
    if h < 155:
        return "purple"
    return "pink"


def _dominant_color(region: Optional[np.ndarray]) -> Optional[str]:
    if region is None or region.size == 0 or region.shape[0] < 4 or region.shape[1] < 4:
        return None
    small = cv2.resize(region, (24, 24), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    votes: Dict[str, int] = {}
    for h, s, v in hsv:
        name = _classify_hsv(float(h), float(s), float(v))
        votes[name] = votes.get(name, 0) + 1
    return max(votes.items(), key=lambda kv: kv[1])[0]


def describe_appearance(frame: Optional[np.ndarray], bbox: Sequence[int],
                        landmarks: Optional[List[List[float]]] = None) -> Dict[str, Optional[str]]:
    if frame is None:
        return {"upper": None, "lower": None}
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, x2 = max(0, x1), min(w_img, x2)
    y1, y2 = max(0, y1), min(h_img, y2)
    if x2 - x1 < 8 or y2 - y1 < 16:
        return {"upper": None, "lower": None}

    bh = y2 - y1
    if landmarks is not None and len(landmarks) >= 27:
        lm = np.asarray(landmarks, dtype=np.float32)
        sh_y = int((lm[11, 1] + lm[12, 1]) / 2)
        hip_y = int((lm[23, 1] + lm[24, 1]) / 2)
        knee_y = int((lm[25, 1] + lm[26, 1]) / 2)
    else:
        sh_y, hip_y, knee_y = y1 + int(bh * 0.2), y1 + int(bh * 0.5), y1 + int(bh * 0.75)

    # 只取中間 60% 寬度，降低背景干擾
    cx1 = x1 + int((x2 - x1) * 0.2)
    cx2 = x2 - int((x2 - x1) * 0.2)
    upper = frame[max(y1, sh_y):max(y1, hip_y), cx1:cx2] if hip_y > sh_y else None
    lower = frame[max(y1, hip_y):min(y2, knee_y), cx1:cx2] if knee_y > hip_y else None
    up, lo = _dominant_color(upper), _dominant_color(lower)
    return {
        "upper": up,
        "lower": lo,
        "upper_zh": COLOR_NAMES_ZH.get(up or "", None),
        "lower_zh": COLOR_NAMES_ZH.get(lo or "", None),
    }
