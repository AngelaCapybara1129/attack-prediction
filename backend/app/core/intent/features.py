"""九種攻擊意圖指數 X1–X9 的可解釋特徵計算（研究計畫表 1）。

輸入為「同一鏡頭同一時間點」的所有人物觀測（bbox + 33 點骨架像素座標）與武器偵測，
輸出每位人物 0~1 的 X1–X9 與用於解釋的輔助量測值（aux）。

公尺換算：未做相機標定時，以人體身高 ≈ 1.7 m 的針孔模型估計
    Z = f · H / h_px，X = (u − W/2) · H / h_px，f ≈ 0.9 · 畫面寬
可在 FeatureEngine(focal_ratio=...) 依實際鏡頭視角調整。
"""
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.core.intent.constants import NUM_INDICES
from app.core.intent.context import SpatialContext
from app.core.intent.skeleton import (
    L_ELBOW, L_HIP, L_SHOULDER, L_WRIST, NOSE, R_ELBOW, R_HIP, R_SHOULDER, R_WRIST,
    joint_angle, torso_length,
)

PERSON_HEIGHT_M = 1.70
DEFAULT_FOCAL_RATIO = 0.9
STATE_TTL_SEC = 8.0
POS_HISTORY_SEC = 35.0
TRAIL_SEC = 4.0
VIS_MIN = 0.5


def _clip01(v: float) -> float:
    return float(min(1.0, max(0.0, v)))


@dataclass
class PersonObservation:
    person_id: int
    track_id: int
    bbox: Sequence[int]
    landmarks: Optional[np.ndarray] = None    # (33,3) 畫面像素座標
    visibility: Optional[np.ndarray] = None   # (33,)
    det_conf: float = 1.0


@dataclass
class WeaponObservation:
    label: str
    conf: float
    bbox: Sequence[int]


@dataclass
class FeatureResult:
    x: np.ndarray
    aux: Dict[str, Any]
    foot_px: Tuple[float, float]
    trail: List[Tuple[int, int]]
    position_m: Tuple[float, float]
    velocity_mps: Tuple[float, float]


@dataclass
class _PersonState:
    pid: int
    first_ts: float
    last_ts: float
    h_px: Optional[float] = None
    pos_hist: Deque[Tuple[float, float, float]] = field(default_factory=deque)   # (ts, X, Z) m
    img_hist: Deque[Tuple[float, float, float]] = field(default_factory=deque)   # (ts, u, v) px
    wrist_hist: Deque[Tuple[float, np.ndarray, np.ndarray]] = field(default_factory=deque)
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    speed: float = 0.0
    x: np.ndarray = field(default_factory=lambda: np.zeros(NUM_INDICES, dtype=np.float32))
    weapon_conf: float = 0.0
    weapon_ts: float = -1e9
    weapon_label: str = ""
    follow: Dict[int, float] = field(default_factory=dict)
    confront: Dict[int, float] = field(default_factory=dict)
    pair_dist: Dict[int, Deque[Tuple[float, float]]] = field(default_factory=dict)
    x4_ts: float = -1e9
    x4_cache: Tuple[float, float, float] = (0.0, 0.0, 0.0)   # (x4, entropy, path_len)
    landmarks: Optional[np.ndarray] = None
    visibility: Optional[np.ndarray] = None
    bbox: Sequence[int] = (0, 0, 0, 0)
    foot: Tuple[float, float] = (0.0, 0.0)


def _visible(vis: Optional[np.ndarray], *joints: int) -> bool:
    if vis is None:
        return True
    return all(float(vis[j]) >= VIS_MIN for j in joints)


def _facing_direction(lm: Optional[np.ndarray], vis: Optional[np.ndarray]) -> int:
    """側身時以鼻子相對肩膀中點的位置推估臉朝畫面左(-1)或右(+1)；正面/背面回傳 0。"""
    if lm is None or not _visible(vis, NOSE, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
        return 0
    torso = torso_length(lm)
    if torso < 1e-3:
        return 0
    shoulder_w = abs(lm[L_SHOULDER, 0] - lm[R_SHOULDER, 0]) / torso
    if shoulder_w > 0.55:
        return 0
    offset = lm[NOSE, 0] - (lm[L_SHOULDER, 0] + lm[R_SHOULDER, 0]) / 2.0
    if abs(offset) < 0.12 * torso:
        return 0
    return 1 if offset > 0 else -1


def _velocity(hist: Deque[Tuple[float, float, float]], now: float, window: float = 1.0) -> np.ndarray:
    pts = [p for p in hist if now - p[0] <= window]
    if len(pts) < 3 or pts[-1][0] - pts[0][0] < 0.3:
        return np.zeros(2)
    t = np.array([p[0] for p in pts])
    t = t - t.mean()
    denom = float((t ** 2).sum())
    if denom < 1e-9:
        return np.zeros(2)
    xs = np.array([p[1] for p in pts])
    zs = np.array([p[2] for p in pts])
    return np.array([float((t * (xs - xs.mean())).sum() / denom), float((t * (zs - zs.mean())).sum() / denom)])


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class FeatureEngine:
    """單一鏡頭的 X1–X9 計算器。X8 由外部（跨鏡頭的風險歷史）傳入。"""

    def __init__(self, camera_id: str, context: Optional[SpatialContext] = None,
                 focal_ratio: float = DEFAULT_FOCAL_RATIO):
        self.camera_id = camera_id
        self.context = context or SpatialContext({})
        self.focal_ratio = focal_ratio
        self.states: Dict[int, _PersonState] = {}

    # ------------------------------------------------------------------
    def update(
        self,
        ts: float,
        frame_w: int,
        frame_h: int,
        persons: Sequence[PersonObservation],
        weapons: Sequence[WeaponObservation] = (),
        history_scores: Optional[Dict[int, float]] = None,
    ) -> Dict[int, FeatureResult]:
        history_scores = history_scores or {}
        dts: Dict[int, float] = {}
        present: Dict[int, _PersonState] = {}

        for obs in persons:
            if obs.person_id in present:
                continue
            st = self.states.get(obs.person_id)
            if st is None:
                st = _PersonState(pid=obs.person_id, first_ts=ts, last_ts=ts)
                self.states[obs.person_id] = st
            dts[obs.person_id] = float(min(max(ts - st.last_ts, 0.0), 0.5))
            st.last_ts = ts
            self._update_kinematics(st, obs, ts, frame_w, frame_h)
            present[obs.person_id] = st

        self._assign_weapons(ts, present, weapons)

        results: Dict[int, FeatureResult] = {}
        for pid, st in present.items():
            dt = dts[pid]
            aux: Dict[str, Any] = {}
            others = [o for o in present.values() if o.pid != pid]
            x = np.zeros(NUM_INDICES, dtype=np.float32)
            x[0] = self._x1_weapon(st, ts, aux)
            x[1] = self._x2_punch(st, ts, dt, aux)
            x[2] = self._x3_chase(st, others, aux)
            x[3] = self._x4_loiter(st, ts, aux)
            x[4] = self._x5_follow(st, others, dt, ts, aux)
            x[5] = self._x6_intrusion(st, others, ts, dt, aux)
            x[6] = self._x7_group(st, others, dt, aux)
            hist = float(history_scores.get(pid, 0.0))
            x[7] = _clip01(hist / 3.0)
            aux["history_score"] = round(hist, 3)
            x9, ctx = self.context.score(self.camera_id, ts, (st.foot[0] / max(frame_w, 1), st.foot[1] / max(frame_h, 1)))
            x[8] = x9
            aux.update(ctx)
            st.x = x
            trail = [(int(u), int(v)) for (t, u, v) in st.img_hist if ts - t <= TRAIL_SEC]
            results[pid] = FeatureResult(
                x=x.copy(), aux=aux, foot_px=st.foot, trail=trail,
                position_m=(st.pos_hist[-1][1], st.pos_hist[-1][2]) if st.pos_hist else (0.0, 0.0),
                velocity_mps=(float(st.vel[0]), float(st.vel[1])),
            )

        stale = [pid for pid, s in self.states.items() if ts - s.last_ts > STATE_TTL_SEC]
        for pid in stale:
            self.states.pop(pid, None)
        return results

    # ------------------------------------------------------------------
    def _update_kinematics(self, st: _PersonState, obs: PersonObservation, ts: float, fw: int, fh: int) -> None:
        x1, y1, x2, y2 = [float(v) for v in obs.bbox]
        bh = max(1.0, y2 - y1)
        lm = obs.landmarks
        vis = obs.visibility
        st.landmarks = lm
        st.visibility = vis
        st.bbox = obs.bbox

        truncated = y2 >= fh - 3 or y1 <= 2
        h_obs: Optional[float] = None
        if not truncated:
            h_obs = bh
        elif lm is not None and _visible(vis, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
            tl = torso_length(lm)
            if tl > 5:
                h_obs = tl * 3.3
        if h_obs is not None:
            st.h_px = h_obs if st.h_px is None else 0.7 * st.h_px + 0.3 * h_obs
        h = st.h_px or bh

        u = (x1 + x2) / 2.0
        v = y2 if not truncated else min(y1 + h, float(fh))
        st.foot = (u, v)
        focal = self.focal_ratio * fw
        z_m = focal * PERSON_HEIGHT_M / h
        x_m = (u - fw / 2.0) * PERSON_HEIGHT_M / h
        if st.pos_hist:
            _, px, pz = st.pos_hist[-1]
            x_m = 0.5 * x_m + 0.5 * px
            z_m = 0.5 * z_m + 0.5 * pz
        st.pos_hist.append((ts, x_m, z_m))
        while st.pos_hist and ts - st.pos_hist[0][0] > POS_HISTORY_SEC:
            st.pos_hist.popleft()
        st.img_hist.append((ts, u, v))
        while st.img_hist and ts - st.img_hist[0][0] > TRAIL_SEC + 1:
            st.img_hist.popleft()

        st.vel = _velocity(st.pos_hist, ts)
        st.speed = float(np.linalg.norm(st.vel))

        if lm is not None and _visible(vis, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
            torso = torso_length(lm)
            if torso > 5:
                hip = (lm[L_HIP, :2] + lm[R_HIP, :2]) / 2.0
                st.wrist_hist.append((ts, (lm[L_WRIST, :2] - hip) / torso, (lm[R_WRIST, :2] - hip) / torso))
        while st.wrist_hist and ts - st.wrist_hist[0][0] > 1.5:
            st.wrist_hist.popleft()

    def _assign_weapons(self, ts: float, present: Dict[int, _PersonState], weapons: Sequence[WeaponObservation]) -> None:
        for w in weapons:
            wx1, wy1, wx2, wy2 = [float(v) for v in w.bbox]
            cx, cy = (wx1 + wx2) / 2.0, (wy1 + wy2) / 2.0
            best_pid, best_d = None, 1e9
            for pid, st in present.items():
                x1, y1, x2, y2 = [float(v) for v in st.bbox]
                h = max(1.0, y2 - y1)
                inside = x1 - 0.15 * (x2 - x1) <= cx <= x2 + 0.15 * (x2 - x1) and y1 <= cy <= y2
                lm = st.landmarks
                if lm is not None:
                    d = min(math.hypot(cx - lm[L_WRIST, 0], cy - lm[L_WRIST, 1]),
                            math.hypot(cx - lm[R_WRIST, 0], cy - lm[R_WRIST, 1])) / h
                else:
                    dx = max(x1 - cx, 0.0, cx - x2)
                    dy = max(y1 - cy, 0.0, cy - y2)
                    d = math.hypot(dx, dy) / h
                if inside:
                    d = min(d, 0.3)
                if d < best_d:
                    best_d, best_pid = d, pid
            if best_pid is not None and best_d < 0.35:
                st = present[best_pid]
                if ts - st.weapon_ts < 1.0:
                    st.weapon_conf = 0.5 * st.weapon_conf + 0.5 * float(w.conf)
                else:
                    st.weapon_conf = float(w.conf)
                st.weapon_ts = ts
                st.weapon_label = w.label

    # ------------------------------------------------------------------ X1
    @staticmethod
    def _x1_weapon(st: _PersonState, ts: float, aux: Dict[str, Any]) -> float:
        age = ts - st.weapon_ts
        if age <= 0.7:
            val = st.weapon_conf
        else:
            val = st.weapon_conf * math.exp(-(age - 0.7) / 1.0)
        if val < 0.05:
            return 0.0
        aux["weapon"] = st.weapon_label
        aux["weapon_conf"] = round(st.weapon_conf, 3)
        return _clip01(val)

    # ------------------------------------------------------------------ X2
    @staticmethod
    def _x2_punch(st: _PersonState, ts: float, dt: float, aux: Dict[str, Any]) -> float:
        lm, vis = st.landmarks, st.visibility
        raw = 0.0
        if lm is not None:
            torso = torso_length(lm)
            # 手腕速度（相對骨盆、以軀幹長為單位）
            wrist_speed = [0.0, 0.0]
            if len(st.wrist_hist) >= 2:
                now_t, l_now, r_now = st.wrist_hist[-1]
                ref = None
                for item in st.wrist_hist:
                    if now_t - item[0] >= 0.1:
                        ref = item
                    else:
                        break
                if ref is not None:
                    span = max(now_t - ref[0], 1e-3)
                    wrist_speed = [float(np.linalg.norm(l_now - ref[1]) / span), float(np.linalg.norm(r_now - ref[2]) / span)]

            arms = [(L_SHOULDER, L_ELBOW, L_WRIST), (R_SHOULDER, R_ELBOW, R_WRIST)]
            angles = []
            for k, (s, e, w) in enumerate(arms):
                if not _visible(vis, s, e, w):
                    angles.append(None)
                    continue
                ang = joint_angle(lm[s], lm[e], lm[w])
                angles.append(ang)
                ext = _clip01((ang - 120.0) / 40.0)
                # 出拳約 5–10 m/s ≈ 10–20 軀幹長/秒；在 10–15 fps 下平均後約 4–8，一般手勢 < 2
                speed_f = _clip01((wrist_speed[k] - 2.0) / 4.0)
                raw = max(raw, speed_f * (0.4 + 0.6 * ext))

            # 護頭/備戰架勢：雙手肘彎曲且手腕抬到肩膀附近
            guard = False
            if torso > 5 and all(a is not None for a in angles) and _visible(vis, NOSE):
                sh_y = (lm[L_SHOULDER, 1] + lm[R_SHOULDER, 1]) / 2.0
                wy = [lm[L_WRIST, 1], lm[R_WRIST, 1]]
                guard = (
                    angles[0] < 100 and angles[1] < 100
                    and all(y < sh_y + 0.1 * torso for y in wy)
                    and all(y > lm[NOSE, 1] - 0.6 * torso for y in wy)
                )
            if guard:
                raw = max(raw, 0.55)
            aux["elbow_angle"] = [round(a, 1) if a is not None else None for a in angles]
            aux["wrist_speed"] = [round(v, 2) for v in wrist_speed]
            aux["guard_pose"] = guard
        decayed = float(st.x[1]) * math.exp(-dt / 1.0) if dt > 0 else float(st.x[1])
        return _clip01(max(raw, decayed))

    # ------------------------------------------------------------------ X3
    @staticmethod
    def _x3_chase(st: _PersonState, others: List[_PersonState], aux: Dict[str, Any]) -> float:
        v = st.speed
        raw = _clip01((v - 1.2) / 1.4)
        lean = 0.0
        lm = st.landmarks
        if lm is not None and _visible(st.visibility, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
            sh = (lm[L_SHOULDER, :2] + lm[R_SHOULDER, :2]) / 2.0
            hip = (lm[L_HIP, :2] + lm[R_HIP, :2]) / 2.0
            d = sh - hip
            if d[1] < 0:
                lean = float(np.degrees(np.arctan2(abs(d[0]), -d[1])))
        if v > 1.2:
            raw += 0.15 * _clip01((lean - 10.0) / 20.0)
            if st.pos_hist:
                me = np.array(st.pos_hist[-1][1:])
                for o in others:
                    if not o.pos_hist:
                        continue
                    rel = np.array(o.pos_hist[-1][1:]) - me
                    if np.linalg.norm(rel) < 8.0 and _cos(st.vel, rel) > 0.7 and v > 1.5:
                        raw += 0.15
                        aux["chasing"] = o.pid
                        break
        aux["speed_mps"] = round(v, 2)
        aux["lean_deg"] = round(lean, 1)
        return _clip01(0.6 * float(st.x[2]) + 0.4 * _clip01(raw))

    # ------------------------------------------------------------------ X4
    @staticmethod
    def _x4_loiter(st: _PersonState, ts: float, aux: Dict[str, Any]) -> float:
        if ts - st.x4_ts >= 0.5:
            st.x4_ts = ts
            # 先以 1 秒平均降低深度估計抖動，再計算移動方向熵
            buckets: Dict[int, List[Tuple[float, float]]] = {}
            for t, x, z in st.pos_hist:
                if ts - t <= 30.0:
                    buckets.setdefault(int(t), []).append((x, z))
            pts = [np.mean(np.array(b), axis=0) for _, b in sorted(buckets.items())]
            angles: List[float] = []
            path_len = 0.0
            if pts:
                anchor = pts[0]
                for p in pts[1:]:
                    step = p - anchor
                    dist = float(np.linalg.norm(step))
                    if dist >= 0.5:
                        angles.append(math.atan2(step[1], step[0]))
                        path_len += dist
                        anchor = p
            entropy = 0.0
            if len(angles) >= 6:
                hist, _ = np.histogram(angles, bins=8, range=(-math.pi, math.pi))
                p = hist[hist > 0] / float(len(angles))
                entropy = float(-(p * np.log2(p)).sum())
            dwell = ts - st.first_ts
            x4 = _clip01((entropy - 0.5) / 2.0) * _clip01(dwell / 15.0) if path_len >= 3.0 else 0.0
            st.x4_cache = (x4, entropy, path_len)
        x4, entropy, path_len = st.x4_cache
        aux["path_entropy"] = round(entropy, 3)
        aux["path_len_m"] = round(path_len, 1)
        aux["dwell_sec"] = round(ts - st.first_ts, 1)
        return x4

    # ------------------------------------------------------------------ X5
    @staticmethod
    def _x5_follow(st: _PersonState, others: List[_PersonState], dt: float, ts: float, aux: Dict[str, Any]) -> float:
        if not st.pos_hist:
            return 0.0
        me = np.array(st.pos_hist[-1][1:])
        seen = set()
        for o in others:
            if not o.pos_hist:
                continue
            seen.add(o.pid)
            rel = np.array(o.pos_hist[-1][1:]) - me
            d = float(np.linalg.norm(rel))
            cond = (
                d < 2.0 and st.speed > 0.3 and o.speed > 0.3
                and _cos(st.vel, o.vel) > 0.6 and _cos(st.vel, rel) > 0.5
            )
            prev = st.follow.get(o.pid, 0.0)
            st.follow[o.pid] = prev + dt if cond else max(0.0, prev - 2.0 * dt)
        for pid in list(st.follow):
            if pid not in seen:
                st.follow[pid] = max(0.0, st.follow[pid] - 2.0 * dt)
                if st.follow[pid] <= 0:
                    st.follow.pop(pid, None)
        if not st.follow:
            return 0.0
        target, sec = max(st.follow.items(), key=lambda kv: kv[1])
        if sec > 0:
            aux["follow_target"] = target
            aux["follow_sec"] = round(sec, 1)
        return _clip01(sec / 10.0)

    # ------------------------------------------------------------------ X6
    @staticmethod
    def _x6_intrusion(st: _PersonState, others: List[_PersonState], ts: float, dt: float, aux: Dict[str, Any]) -> float:
        raw = 0.0
        if st.pos_hist:
            me = np.array(st.pos_hist[-1][1:])
            nearest = None
            for o in others:
                if not o.pos_hist:
                    continue
                rel = np.array(o.pos_hist[-1][1:]) - me
                d = float(np.linalg.norm(rel))
                hist = st.pair_dist.setdefault(o.pid, deque())
                hist.append((ts, d))
                while hist and ts - hist[0][0] > 1.2:
                    hist.popleft()
                closing = 0.0
                ref = None
                for item in hist:
                    if ts - item[0] >= 0.3:
                        ref = item
                    else:
                        break
                if ref is not None:
                    closing = (ref[1] - d) / max(ts - ref[0], 1e-3)
                prox = _clip01((1.0 - d) / 0.6)
                sudden = _clip01(closing / 1.0)
                pair = prox * (0.4 + 0.6 * sudden)
                u = rel / d if d > 1e-6 else np.zeros(2)
                a_me = max(0.0, float(np.dot(st.vel, u)))
                a_other = max(0.0, float(np.dot(o.vel, -u)))
                share = a_me / (a_me + a_other) if a_me + a_other > 0.2 else 0.5
                val = _clip01(pair * (0.5 + share))
                if nearest is None or d < nearest[0]:
                    nearest = (d, o.pid, closing)
                raw = max(raw, val)
            alive = {o.pid for o in others}
            for pid in list(st.pair_dist):
                if pid not in alive:
                    st.pair_dist.pop(pid, None)
            if nearest is not None:
                aux["nearest_dist_m"] = round(nearest[0], 2)
                aux["nearest_pid"] = nearest[1]
                aux["closing_mps"] = round(nearest[2], 2)
        decayed = float(st.x[5]) * math.exp(-dt / 1.5) if dt > 0 else float(st.x[5])
        return _clip01(max(raw, decayed))

    # ------------------------------------------------------------------ X7
    @staticmethod
    def _x7_group(st: _PersonState, others: List[_PersonState], dt: float, aux: Dict[str, Any]) -> float:
        if not st.pos_hist:
            return 0.0
        me = np.array(st.pos_hist[-1][1:])
        dists = []
        for o in others:
            if o.pos_hist:
                dists.append((float(np.linalg.norm(np.array(o.pos_hist[-1][1:]) - me)), o))
        close = [d for d, _ in dists if d < 1.5]
        group = len(close) + 1
        if group >= 3:
            dens = 0.7 + 0.1 * (group - 3) + 0.2 * _clip01((1.5 - float(np.mean(close))) / 1.0)
        elif group == 2:
            dens = 0.3 * _clip01((1.5 - close[0]) / 1.0)
        else:
            dens = 0.0

        confront_val = 0.0
        my_dir = _facing_direction(st.landmarks, st.visibility)
        seen = set()
        for d, o in dists:
            seen.add(o.pid)
            cond = False
            if d < 1.8 and my_dir != 0 and st.speed < 0.6 and o.speed < 0.6:
                o_dir = _facing_direction(o.landmarks, o.visibility)
                side = 1 if o.foot[0] > st.foot[0] else -1
                cond = my_dir == side and o_dir == -side
            prev = st.confront.get(o.pid, 0.0)
            st.confront[o.pid] = prev + dt if cond else max(0.0, prev - 2.0 * dt)
        for pid in list(st.confront):
            if pid not in seen:
                st.confront.pop(pid, None)
        if st.confront:
            sec = max(st.confront.values())
            confront_val = 0.85 * _clip01(sec / 3.0)
            if sec > 0:
                aux["confront_sec"] = round(sec, 1)
        aux["group_size"] = group
        raw = _clip01(max(dens, confront_val))
        return _clip01(0.7 * float(st.x[6]) + 0.3 * raw)
