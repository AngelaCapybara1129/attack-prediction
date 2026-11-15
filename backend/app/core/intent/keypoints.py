"""YOLO26-pose（COCO 17 點）→ MediaPipe 33 點格式。

為什麼改用 YOLO26-pose：RWF-2000（低解析度、多人遮擋的監視器畫面）實測，
偵測到的人之中「肩膀＋髖部 4 點皆可信（≥ 0.5）」的比例：
    YOLO26l 偵測 + MediaPipe 裁切估計  40%（92 ms/幀）
    YOLO26n-pose 整張畫面多人關鍵點     85%（13 ms/幀）
MediaPipe 一次只估計裁切中的一個人，兩人重疊時常把前面那人的手腳算到後面那人身上。

保留 33 點格式，讓 X1–X9 特徵、ST-GCN 圖結構、資料格式與既有程式不用改：
  - COCO 有的 17 個點直接對應（鼻、眼、耳、肩、肘、腕、髖、膝、踝；左右皆為「當事人的左右」，兩者一致）
  - 其餘 16 個點由鄰近點推得：眼內外側 = 眼睛、嘴角 = 鼻子、手指 = 手腕沿前臂方向延伸、腳跟／腳尖 = 腳踝
    推得點的可信度取來源點可信度 × 0.8
  - z（深度）一律為 0；X1–X9 只使用 2D 座標
"""
from typing import Optional, Tuple

import numpy as np

NUM_COCO = 17

# MediaPipe 索引 ← COCO 索引
_DIRECT = {
    0: 0,             # nose
    2: 1, 5: 2,       # left / right eye
    7: 3, 8: 4,       # left / right ear
    11: 5, 12: 6,     # shoulders
    13: 7, 14: 8,     # elbows
    15: 9, 16: 10,    # wrists
    23: 11, 24: 12,   # hips
    25: 13, 26: 14,   # knees
    27: 15, 28: 16,   # ankles
}
# MediaPipe 推得點：(目標, 來源 MediaPipe 索引)
_COPY = [(1, 2), (3, 2), (4, 5), (6, 5), (9, 0), (10, 0), (29, 27), (30, 28), (31, 27), (32, 28)]
# 手指：手腕沿「手肘→手腕」方向再延伸 ratio 倍前臂長
_HAND = [(17, 13, 15, 0.25), (19, 13, 15, 0.30), (21, 13, 15, 0.15),
         (18, 14, 16, 0.25), (20, 14, 16, 0.30), (22, 14, 16, 0.15)]
DERIVED_CONF_FACTOR = 0.8


def coco17_to_mp33(xy: np.ndarray, conf: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """xy: (17,2) 像素座標；conf: (17,) 0~1。回傳 (33,3) 像素座標（z=0）與 (33,) 可信度。"""
    xy = np.asarray(xy, dtype=np.float32).reshape(NUM_COCO, 2)
    c = np.ones(NUM_COCO, np.float32) if conf is None else np.asarray(conf, dtype=np.float32).reshape(NUM_COCO)
    out = np.zeros((33, 3), np.float32)
    vis = np.zeros(33, np.float32)
    for mp_i, coco_i in _DIRECT.items():
        out[mp_i, :2] = xy[coco_i]
        vis[mp_i] = c[coco_i]
    for dst, src in _COPY:
        out[dst] = out[src]
        vis[dst] = vis[src] * DERIVED_CONF_FACTOR
    for dst, elbow, wrist, ratio in _HAND:
        out[dst, :2] = out[wrist, :2] + ratio * (out[wrist, :2] - out[elbow, :2])
        vis[dst] = min(vis[wrist], vis[elbow]) * DERIVED_CONF_FACTOR
    return out, vis


def torso_confidence(vis33: np.ndarray) -> float:
    """肩膀與髖部 4 點的最低可信度，用來判斷這副骨架是否可用。"""
    v = np.asarray(vis33)
    return float(min(v[11], v[12], v[23], v[24]))


def pose_from_result(result, index: int, min_torso_conf: float = 0.3):
    """從 ultralytics pose 結果取第 index 個人的 33 點骨架；無關鍵點或軀幹不可信時回傳 (None, None)。"""
    kp = getattr(result, "keypoints", None)
    if kp is None or kp.xy is None or index >= len(kp.xy):
        return None, None
    xy = kp.xy[index].cpu().numpy()
    conf = kp.conf[index].cpu().numpy() if kp.conf is not None else None
    lm, vis = coco17_to_mp33(xy, conf)
    if torso_confidence(vis) < min_torso_conf:
        return None, None
    return lm, vis
