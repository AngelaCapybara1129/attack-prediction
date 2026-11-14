"""即時攻擊意圖分析引擎：每幀接收追蹤 + 骨架 + 武器偵測，輸出每人 X1–X9、風險分數、L0–L4 與告警。

流程（研究計畫圖 2）：
  目標鎖定與骨架提取 → 行為特徵量化 (X1–X9) → 雙通道時序預測 (ST-GCN + LSTM, 可選)
  → AHP 風險聚合 + 表 2 規則 → L0–L4 → 告警
"""
import threading
import time
import uuid
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.core.intent.constants import (
    ACTION_NAMES, INDEX_KEYS, INDEX_NAMES, LEVEL_COLOR_NAME, LEVEL_INDEX, LEVEL_REACTION,
)
from app.core.intent.context import SpatialContext
from app.core.intent.features import FeatureEngine, PersonObservation, WeaponObservation
from app.core.intent.predictor import DEFAULT_SEQ_LEN, DEFAULT_WINDOW_SEC, IntentPredictor
from app.core.intent.risk import RiskEngine, RiskResult
from app.core.intent.skeleton import NUM_JOINTS, normalize_skeleton_sequence, resample_by_time

PREDICT_INTERVAL_SEC = 1.0
MODEL_OUTPUT_TTL_SEC = 2.5
BUFFER_SEC = 12.0
LIVE_VISIBLE_SEC = 3.0
ALERT_COOLDOWN_SEC = {"L1": 120.0, "L2": 60.0, "L3": 30.0, "L4": 15.0}


def _xdict(x: Sequence[float]) -> Dict[str, float]:
    return {k: round(float(v), 4) for k, v in zip(INDEX_KEYS, x)}


class IntentEngine:
    def __init__(
        self,
        task_id: str,
        predictor: Optional[IntentPredictor] = None,
        context: Optional[SpatialContext] = None,
        alert_min_level: str = "L1",
        model_weight: float = 0.5,
    ):
        self.task_id = task_id
        self.predictor = predictor
        self.context = context or SpatialContext.load()
        self.alert_min_level = alert_min_level
        self.seq_len = predictor.seq_len if predictor else DEFAULT_SEQ_LEN
        self.window_sec = predictor.window_sec if predictor else DEFAULT_WINDOW_SEC

        self.features: Dict[str, FeatureEngine] = {}
        self.risk = RiskEngine(model_weight=model_weight)
        self.skel_buf: Dict[int, Deque[Tuple[float, Optional[np.ndarray]]]] = {}
        self.x_buf: Dict[int, Deque[Tuple[float, np.ndarray]]] = {}
        self.model_out: Dict[int, Dict[str, Any]] = {}
        self.last_pred_ts: Dict[int, float] = {}
        self.alert_state: Dict[int, Dict[str, float]] = {}
        self.camera_path: Dict[int, List[Dict[str, Any]]] = {}
        self.score_hist: Dict[int, Deque[Tuple[float, float, str]]] = {}
        self.latest: Dict[int, Dict[str, Any]] = {}
        self.last_seen: Dict[int, float] = {}

        self._lock = threading.Lock()
        self._last_cleanup = 0.0
        self.started_at = time.time()
        self.frames_processed = 0
        self.alerts_emitted = 0
        self.running = True

    # ------------------------------------------------------------------
    def process(
        self,
        camera_id: str,
        ts: float,
        frame_w: int,
        frame_h: int,
        persons: Sequence[PersonObservation],
        weapons: Sequence[WeaponObservation] = (),
    ) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]]]:
        fe = self.features.get(camera_id)
        if fe is None:
            fe = FeatureEngine(camera_id, self.context)
            self.features[camera_id] = fe

        feats = fe.update(ts, frame_w, frame_h, persons, weapons, self.risk.history_scores())
        obs_by_pid = {o.person_id: o for o in persons}
        states: Dict[int, Dict[str, Any]] = {}
        alerts: List[Dict[str, Any]] = []

        for pid, fr in feats.items():
            obs = obs_by_pid[pid]
            self._push_buffers(pid, ts, obs, fr.x)
            self._update_camera_path(pid, camera_id, ts)
            model_out = self._maybe_predict(pid, ts)
            res = self.risk.evaluate(pid, ts, fr.x, model_out["risk"] if model_out else None)
            self._push_score(pid, ts, res)

            state = self._build_state(pid, camera_id, ts, obs, fr, res, model_out)
            states[pid] = state
            alert = self._maybe_alert(pid, ts, state, res, obs)
            if alert is not None:
                alerts.append(alert)

        with self._lock:
            for pid, st in states.items():
                self.latest[pid] = st
                self.last_seen[pid] = ts
            self.frames_processed += 1
            self.alerts_emitted += len(alerts)

        if ts - self._last_cleanup > 10.0:
            self._cleanup(ts)
        return states, alerts

    # ------------------------------------------------------------------
    def _push_buffers(self, pid: int, ts: float, obs: PersonObservation, x: np.ndarray) -> None:
        sb = self.skel_buf.setdefault(pid, deque())
        lm = None
        if obs.landmarks is not None and np.asarray(obs.landmarks).shape == (NUM_JOINTS, 3):
            lm = np.asarray(obs.landmarks, dtype=np.float32)
        sb.append((ts, lm))
        while sb and ts - sb[0][0] > BUFFER_SEC:
            sb.popleft()
        xb = self.x_buf.setdefault(pid, deque())
        xb.append((ts, np.asarray(x, dtype=np.float32)))
        while xb and ts - xb[0][0] > BUFFER_SEC:
            xb.popleft()

    def _push_score(self, pid: int, ts: float, res: RiskResult) -> None:
        hist = self.score_hist.setdefault(pid, deque(maxlen=120))
        if not hist or ts - hist[-1][0] >= 1.0:
            hist.append((ts, round(res.score, 3), res.level))

    def _update_camera_path(self, pid: int, camera_id: str, ts: float) -> None:
        path = self.camera_path.setdefault(pid, [])
        if path and path[-1]["camera_id"] == camera_id:
            path[-1]["last_ts"] = ts
        else:
            path.append({"camera_id": camera_id, "first_ts": ts, "last_ts": ts})
            if len(path) > 20:
                del path[0]

    def _peak_x(self, pid: int, ts: float, window: float) -> np.ndarray:
        xb = self.x_buf.get(pid)
        if not xb:
            return np.zeros(len(INDEX_KEYS), dtype=np.float32)
        arr = np.stack([x for t, x in xb if ts - t <= window])
        return arr.max(axis=0)

    def _maybe_predict(self, pid: int, ts: float) -> Optional[Dict[str, Any]]:
        if self.predictor is None:
            return None
        cached = self.model_out.get(pid)
        if ts - self.last_pred_ts.get(pid, -1e9) < PREDICT_INTERVAL_SEC:
            return cached if cached and ts - cached["ts"] <= MODEL_OUTPUT_TTL_SEC else None
        self.last_pred_ts[pid] = ts

        sb = [item for item in self.skel_buf.get(pid, ()) if ts - item[0] <= self.window_sec]
        valid_items = [item for item in sb if item[1] is not None]
        if len(valid_items) < 8 or valid_items[-1][0] - valid_items[0][0] < 0.6 * self.window_sec:
            return cached if cached and ts - cached["ts"] <= MODEL_OUTPUT_TTL_SEC else None

        times = [t for t, _ in sb]
        frames = np.stack([lm if lm is not None else np.zeros((NUM_JOINTS, 3), np.float32) for _, lm in sb])
        valid = np.array([lm is not None for _, lm in sb])
        skel_seq, mask = resample_by_time(times, frames, valid, ts, self.window_sec, self.seq_len)
        if mask.mean() < 0.5:
            return None
        skel = normalize_skeleton_sequence(skel_seq, mask > 0)

        xb = [item for item in self.x_buf.get(pid, ()) if ts - item[0] <= self.window_sec + 0.5]
        x_times = [t for t, _ in xb]
        x_frames = np.stack([x for _, x in xb])
        xseq, _ = resample_by_time(x_times, x_frames, np.ones(len(xb), bool), ts, self.window_sec, self.seq_len, max_gap_sec=1e9)

        try:
            out = self.predictor.predict(skel, xseq.astype(np.float32))
        except Exception as e:
            print(f"[INTENT] predict 失敗 pid={pid}: {e}")
            return None
        out["ts"] = ts
        self.model_out[pid] = out
        return out

    def _build_state(self, pid: int, camera_id: str, ts: float, obs: PersonObservation, fr, res: RiskResult,
                     model_out: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        top = sorted(
            ({"key": k, "name": INDEX_NAMES[k], "value": round(float(v), 3)} for k, v in zip(INDEX_KEYS, fr.x)),
            key=lambda d: d["value"], reverse=True,
        )
        return {
            "task_id": self.task_id,
            "person_id": int(pid),
            "track_id": int(obs.track_id),
            "camera_id": camera_id,
            "ts": ts,
            "bbox": [int(v) for v in obs.bbox],
            "level": res.level,
            "level_color": LEVEL_COLOR_NAME[res.level],
            "reaction": LEVEL_REACTION[res.level],
            "score": round(res.score, 3),
            "rule_score": round(res.rule_score, 3),
            "model_score": None if res.model_score is None else round(res.model_score, 3),
            "score_level": res.score_level,
            "rule_level": res.rule_level,
            "indices": _xdict(fr.x),
            "indices_peak": _xdict(self._peak_x(pid, ts, 5.0)),
            "top_indices": [t for t in top if t["value"] >= 0.2][:3],
            "triggers": res.triggers,
            "aux": fr.aux,
            "action": model_out.get("action") if model_out else None,
            "action_name": ACTION_NAMES.get(model_out.get("action", ""), None) if model_out else None,
            "action_conf": round(model_out["action_conf"], 3) if model_out else None,
            "predicted_next": _xdict(model_out["next_x"]) if model_out else None,
            "predicted_level": model_out.get("level") if model_out else None,
            "trail": fr.trail,
            "foot_px": [round(fr.foot_px[0], 1), round(fr.foot_px[1], 1)],
            "position_m": [round(fr.position_m[0], 2), round(fr.position_m[1], 2)],
            "velocity_mps": [round(fr.velocity_mps[0], 2), round(fr.velocity_mps[1], 2)],
            "camera_path": [dict(p) for p in self.camera_path.get(pid, [])],
            "has_pose": obs.landmarks is not None,
        }

    def _maybe_alert(self, pid: int, ts: float, state: Dict[str, Any], res: RiskResult,
                     obs: PersonObservation) -> Optional[Dict[str, Any]]:
        lvl = LEVEL_INDEX[res.level]
        if lvl == 0:
            self.alert_state.pop(pid, None)
        if lvl < LEVEL_INDEX[self.alert_min_level]:
            return None
        prev = self.alert_state.get(pid)
        cooldown = ALERT_COOLDOWN_SEC.get(res.level, 60.0)
        if prev is not None and lvl <= prev["level_idx"] and ts - prev["ts"] < cooldown:
            return None
        self.alert_state[pid] = {"level_idx": lvl, "ts": ts}
        self.risk.add_history(pid, res.level)
        alert = dict(state)
        alert["alert_id"] = uuid.uuid4().hex[:12]
        alert["escalated"] = prev is None or lvl > prev["level_idx"]
        alert["score_history"] = [
            {"ts": t, "score": s, "level": lv} for t, s, lv in self.score_hist.get(pid, ())
        ][-30:]
        alert["landmarks"] = None if obs.landmarks is None else np.asarray(obs.landmarks).round(1).tolist()
        return alert

    def _cleanup(self, ts: float) -> None:
        self._last_cleanup = ts
        with self._lock:
            gone = [pid for pid, t in self.last_seen.items() if ts - t > 60.0]
            for pid in gone:
                for store in (self.skel_buf, self.x_buf, self.model_out, self.last_pred_ts, self.latest,
                              self.last_seen, self.score_hist):
                    store.pop(pid, None)
        self.risk.forget(600.0, ts)

    # ------------------------------------------------------------------ 查詢 API 用
    def snapshot(self) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            persons = []
            for pid, st in self.latest.items():
                if now - self.last_seen.get(pid, 0) > LIVE_VISIBLE_SEC:
                    continue
                item = {k: v for k, v in st.items() if k not in ("trail",)}
                item["history"] = [{"ts": t, "score": s, "level": lv} for t, s, lv in self.score_hist.get(pid, ())][-60:]
                persons.append(item)
        persons.sort(key=lambda s: (LEVEL_INDEX[s["level"]], s["score"]), reverse=True)
        return {
            "task_id": self.task_id,
            "running": self.running,
            "started_at": self.started_at,
            "frames_processed": self.frames_processed,
            "alerts_emitted": self.alerts_emitted,
            "model_loaded": self.predictor is not None,
            "persons": persons,
        }

    def person_detail(self, pid: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            st = self.latest.get(pid)
            if st is None:
                return None
            out = dict(st)
            out["history"] = [{"ts": t, "score": s, "level": lv} for t, s, lv in self.score_hist.get(pid, ())]
            out["camera_path"] = [dict(p) for p in self.camera_path.get(pid, [])]
            return out
