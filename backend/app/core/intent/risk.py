"""風險聚合與 L0–L4 分級（研究計畫公式 (4) 與表 2）。

最終等級 = max(分數等級, 規則觸發等級)，並加上降級遲滯避免畫面閃爍。
表 2 觸發條件：
  L4  X1 ≥ 0.7 持續 ≥ 3 s
  L3  X2 與 X6 同時 > 0.6，或 X5 + X8（尾隨 + 歷史風險）
  L2  X3 或 X5 連續 > 5 s，或 X7 > 0.8
  L1  X4 或 X9 單獨觸發 > 10 s
  L0  無異常或僅 X9 短暫觸發
"""
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from app.core.intent.ahp import weighted_sum
from app.core.intent.constants import LEVEL_INDEX, LEVELS, RISK_SCALE, max_level, score_to_level

GRACE_SEC = 0.6          # 「連續」條件容許的短暫中斷（偵測閃爍）
HOLD_DOWN_SEC = 4.0      # 降級前需持續低於目前等級的秒數
HISTORY_HALF_LIFE = 600.0
HISTORY_RATE = {"L0": 0.0, "L1": 0.0, "L2": 0.03, "L3": 0.10, "L4": 0.25}   # 每秒累積
ALERT_HISTORY_BONUS = {"L1": 0.2, "L2": 0.5, "L3": 1.0, "L4": 1.5}


@dataclass(frozen=True)
class Rule:
    key: str
    level: str
    min_duration: float
    description: str
    condition: Callable[[np.ndarray], bool]


RULES: List[Rule] = [
    Rule("L4_weapon", "L4", 3.0, "X₁ 武器準備 ≥ 0.7 持續 ≥ 3 秒", lambda x: x[0] >= 0.7),
    Rule("L3_punch_intrusion", "L3", 0.3, "X₂ 拳擊姿態與 X₆ 距離侵略同時 > 0.6", lambda x: x[1] > 0.6 and x[5] > 0.6),
    Rule("L3_follow_history", "L3", 1.0, "X₅ 尾隨與 X₈ 歷史風險同時 > 0.6", lambda x: x[4] > 0.6 and x[7] > 0.6),
    Rule("L2_chase", "L2", 5.0, "X₃ 追逐衝刺連續 > 5 秒", lambda x: x[2] > 0.5),
    Rule("L2_follow", "L2", 5.0, "X₅ 尾隨連續 > 5 秒", lambda x: x[4] > 0.5),
    Rule("L2_group", "L2", 2.0, "X₇ 群聚對峙 > 0.8", lambda x: x[6] > 0.8),
    Rule("L1_loiter", "L1", 10.0, "X₄ 徘徊持續 > 10 秒", lambda x: x[3] > 0.5),
    Rule("L1_context", "L1", 10.0, "X₉ 高風險時段/區域停留 > 10 秒", lambda x: x[8] > 0.5),
]


@dataclass
class RiskResult:
    level: str
    score: float
    rule_score: float
    model_score: Optional[float]
    score_level: str
    rule_level: str
    triggers: List[Dict]
    escalated: bool


@dataclass
class _RiskState:
    last_ts: Optional[float] = None
    durations: Dict[str, float] = field(default_factory=dict)
    false_for: Dict[str, float] = field(default_factory=dict)
    level: str = "L0"
    lower_for: float = 0.0
    history: float = 0.0


class RiskEngine:
    """跨鏡頭、以全域 person_id 為鍵的風險狀態機。"""

    def __init__(self, model_weight: float = 0.5):
        self.model_weight = float(model_weight)
        self.states: Dict[int, _RiskState] = {}

    def history_score(self, pid: int) -> float:
        st = self.states.get(pid)
        return st.history if st else 0.0

    def history_scores(self) -> Dict[int, float]:
        return {pid: st.history for pid, st in self.states.items()}

    def add_history(self, pid: int, level: str) -> None:
        st = self.states.setdefault(pid, _RiskState())
        st.history += ALERT_HISTORY_BONUS.get(level, 0.0)

    def evaluate(self, pid: int, ts: float, x: np.ndarray, model_score: Optional[float] = None) -> RiskResult:
        st = self.states.setdefault(pid, _RiskState())
        dt = 0.0 if st.last_ts is None else float(min(max(ts - st.last_ts, 0.0), 0.5))
        st.last_ts = ts
        x = np.asarray(x, dtype=np.float32)

        rule_level = "L0"
        triggers: List[Dict] = []
        for rule in RULES:
            if rule.condition(x):
                st.durations[rule.key] = st.durations.get(rule.key, 0.0) + dt
                st.false_for[rule.key] = 0.0
            else:
                st.false_for[rule.key] = st.false_for.get(rule.key, 0.0) + dt
                if st.false_for[rule.key] > GRACE_SEC:
                    st.durations[rule.key] = 0.0
            dur = st.durations.get(rule.key, 0.0)
            if dur >= rule.min_duration and st.false_for.get(rule.key, 0.0) <= GRACE_SEC:
                triggers.append({"rule": rule.key, "level": rule.level, "description": rule.description, "duration": round(dur, 1)})
                rule_level = max_level(rule_level, rule.level)

        rule_score = RISK_SCALE * weighted_sum(x)
        if model_score is not None and math.isfinite(model_score):
            score = (1.0 - self.model_weight) * rule_score + self.model_weight * float(np.clip(model_score, 0.0, RISK_SCALE))
        else:
            score = rule_score
        score_level = score_to_level(score)
        target = max_level(score_level, rule_level)

        prev = st.level
        if LEVEL_INDEX[target] > LEVEL_INDEX[st.level]:
            st.level = target
            st.lower_for = 0.0
        elif LEVEL_INDEX[target] < LEVEL_INDEX[st.level]:
            st.lower_for += dt
            if st.lower_for >= HOLD_DOWN_SEC:
                st.level = LEVELS[LEVEL_INDEX[st.level] - 1] if LEVEL_INDEX[st.level] - 1 > LEVEL_INDEX[target] else target
                st.lower_for = 0.0
        else:
            st.lower_for = 0.0

        # X8 歷史風險：依目前等級累積、半衰期衰減
        st.history = st.history * (0.5 ** (dt / HISTORY_HALF_LIFE)) + HISTORY_RATE.get(st.level, 0.0) * dt

        return RiskResult(
            level=st.level,
            score=float(score),
            rule_score=float(rule_score),
            model_score=None if model_score is None else float(model_score),
            score_level=score_level,
            rule_level=rule_level,
            triggers=triggers,
            escalated=LEVEL_INDEX[st.level] > LEVEL_INDEX[prev],
        )

    def forget(self, max_idle_sec: float, now: float) -> None:
        """移除太久沒出現且無歷史風險的人，避免記憶體無限成長。"""
        drop: List[int] = []
        for pid, st in self.states.items():
            if st.last_ts is not None and now - st.last_ts > max_idle_sec and st.history < 0.05:
                drop.append(pid)
        for pid in drop:
            self.states.pop(pid, None)


def level_and_triggers_summary(result: RiskResult) -> Tuple[str, str]:
    reasons = "；".join(t["description"] for t in result.triggers) or "分數門檻"
    return result.level, reasons
