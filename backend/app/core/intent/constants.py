"""研究計畫表 1（九種攻擊意圖指數）與表 2（風險等級對應規則）的常數定義。"""
from typing import Dict, List, Tuple

INDEX_KEYS: List[str] = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9"]
NUM_INDICES = len(INDEX_KEYS)

INDEX_NAMES: Dict[str, str] = {
    "X1": "武器準備",
    "X2": "拳擊姿態",
    "X3": "追逐衝刺",
    "X4": "徘徊",
    "X5": "尾隨",
    "X6": "距離侵略",
    "X7": "群聚對峙",
    "X8": "歷史風險",
    "X9": "空間脈絡",
}

INDEX_CRITERIA: Dict[str, str] = {
    "X1": "武器偵測可信度 > 0.7",
    "X2": "手肘伸展角度 > 150° 且手腕高速移動，或雙手護頭架勢",
    "X3": "身體移動速度 > 2 m/s（含前傾、朝向他人）",
    "X4": "路徑方向熵 > 1.5 bits",
    "X5": "與目標距離 < 2 m 且持續跟隨 > 10 s",
    "X6": "距離瞬間縮短至 < 0.5 m",
    "X7": "三人以上群聚密度 > 0.8 或面對面對峙",
    "X8": "累積風險分數 > 2.5",
    "X9": "區域權重 × 時段權重",
}

LEVELS: List[str] = ["L0", "L1", "L2", "L3", "L4"]
LEVEL_INDEX: Dict[str, int] = {lv: i for i, lv in enumerate(LEVELS)}

# AHP 加權和 Σ w_i X_i 介於 0~1；表 2 的分數門檻為 0~5+，
# 因此總風險分數 R = RISK_SCALE × Σ w_i X_i（0~10）。
RISK_SCALE = 10.0

# (下限, 等級)，由高到低比對
SCORE_THRESHOLDS: List[Tuple[float, str]] = [
    (5.0, "L4"),
    (3.5, "L3"),
    (2.0, "L2"),
    (1.0, "L1"),
]

LEVEL_COLOR_NAME: Dict[str, str] = {
    "L0": "藍色",
    "L1": "綠色",
    "L2": "黃色",
    "L3": "橘色",
    "L4": "紅色",
}

LEVEL_REACTION: Dict[str, str] = {
    "L0": "持續監控",
    "L1": "記錄事件",
    "L2": "標示",
    "L3": "推播",
    "L4": "立即警告",
}

# OpenCV BGR
LEVEL_BGR: Dict[str, Tuple[int, int, int]] = {
    "L0": (230, 140, 30),   # 藍
    "L1": (60, 190, 60),    # 綠
    "L2": (0, 215, 255),    # 黃
    "L3": (0, 140, 255),    # 橘
    "L4": (0, 0, 255),      # 紅
}

# 行為類型（ST-GCN 動作分類頭）
ACTION_CLASSES: List[str] = ["normal", "suspicious", "dangerous"]
ACTION_NAMES: Dict[str, str] = {
    "normal": "正常",
    "suspicious": "可疑",
    "dangerous": "危險/暴力",
}


def score_to_level(score: float) -> str:
    for lower, level in SCORE_THRESHOLDS:
        if score >= lower:
            return level
    return "L0"


def max_level(*levels: str) -> str:
    return LEVELS[max(LEVEL_INDEX.get(lv, 0) for lv in levels)] if levels else "L0"
