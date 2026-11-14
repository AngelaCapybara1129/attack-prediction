"""模擬鏡頭（測試／展示用）：依劇本產生人物骨架與畫面，不需要攝影機或影片。

來源字串：sim://<情境>
    demo     綜合：路人經過 → 尾隨 → 衝上前揮拳 → 持刀逼近 → 群聚對峙（約 82 秒循環）
    calm     兩名路人經過（預期 L0）
    follow   繞圈尾隨（預期 X5 → L2）
    fight    衝刺貼近後反覆揮拳（預期 X2+X6 → L3）
    weapon   持刀逼近他人（預期 X1 ≥ 0.7 持續 3 秒 → L4）
    group    四人聚集對峙（預期 X7 → L2）
    loiter   單人長時間無目的徘徊（預期 X4 → L1）

骨架直接由劇本產生並交給 IntentEngine（略過 YOLO / Re-ID / MediaPipe），
用來驗證 X1–X9 → AHP → L0–L4 → 告警 → RAG → 前端 這段流程。
要測試偵測端請改用真實影片（影片檔來源會自動循環播放）。
"""
import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.intent.features import PersonObservation, WeaponObservation

W, H = 1280, 720
HORIZON = 170
FPS = 15.0
TAU = 2 * math.pi


def height_at(v: float) -> float:
    """透視：腳越靠近畫面下方，人越高。v=560 → 約 242 px。"""
    return max(60.0, 0.62 * (v - HORIZON))


def px_per_m(v: float) -> float:
    return height_at(v) / 1.7


@dataclass
class ActorSpec:
    actor: int
    u: float                 # 腳底 x（px）
    v: float                 # 腳底 y（px）
    facing: int = 0          # -1 朝左、+1 朝右、0 正面
    arm: str = "rest"        # rest / guard / punch / knife
    punch: float = 0.0       # 揮拳伸展程度 0~1
    moving: bool = False
    color: Tuple[int, int, int] = (90, 90, 90)
    target_dir: int = 1      # 揮拳／持刀朝向


# ---------------------------------------------------------------------------
# 骨架產生
# ---------------------------------------------------------------------------

def _arm(shoulder: np.ndarray, h: float, pose: str, direction: int, e: float, swing: float):
    if pose == "guard":
        elbow = shoulder + np.array([direction * 0.05 * h, 0.12 * h])
        wrist = shoulder + np.array([direction * 0.08 * h, -0.05 * h])
    elif pose == "punch":
        reach = 0.12 * h + 0.24 * h * e
        wrist = shoulder + np.array([direction * reach, -0.02 * h])
        elbow = (shoulder + wrist) / 2 + np.array([0.0, 0.07 * h * (1 - e)])
    elif pose == "knife":
        wrist = shoulder + np.array([direction * 0.26 * h, 0.06 * h])
        elbow = (shoulder + wrist) / 2 + np.array([0.0, 0.03 * h])
    else:  # rest，走路時前後擺動
        elbow = shoulder + np.array([swing * 0.03 * h, 0.17 * h])
        wrist = elbow + np.array([swing * 0.05 * h, 0.16 * h])
    return elbow, wrist


def make_skeleton(a: ActorSpec, t: float) -> np.ndarray:
    """回傳 (33,3) 畫面像素座標（MediaPipe 索引）。"""
    h = height_at(a.v)
    u, v = a.u, a.v
    side = a.facing
    sw = 0.35 if side != 0 else 1.0
    pts = np.zeros((33, 3), dtype=np.float32)
    phase = t * TAU * 1.8 if a.moving else 0.0
    stride = 0.10 * h if a.moving else 0.0

    nose = np.array([u + side * 0.05 * h, v - 0.91 * h])
    pts[0, :2] = nose
    face = [(-0.015, -0.02), (-0.025, -0.02), (-0.035, -0.02), (0.015, -0.02), (0.025, -0.02),
            (0.035, -0.02), (-0.05, -0.01), (0.05, -0.01), (-0.015, 0.025), (0.015, 0.025)]
    for i, (dx, dy) in enumerate(face, start=1):
        pts[i, :2] = nose + np.array([dx * h * (sw if dx else 1), dy * h])

    sh_y, hip_y, knee_y, ank_y = v - 0.82 * h, v - 0.52 * h, v - 0.27 * h, v - 0.03 * h
    pts[11, :2] = (u - 0.12 * h * sw, sh_y)
    pts[12, :2] = (u + 0.12 * h * sw, sh_y)
    pts[23, :2] = (u - 0.07 * h * sw, hip_y)
    pts[24, :2] = (u + 0.07 * h * sw, hip_y)

    for hip_i, knee_i, ank_i, heel_i, toe_i, sgn in ((23, 25, 27, 29, 31, 1), (24, 26, 28, 30, 32, -1)):
        off = sgn * stride * math.sin(phase)
        hx = pts[hip_i, 0]
        pts[knee_i, :2] = (hx + off * 0.5, knee_y)
        pts[ank_i, :2] = (hx + off, ank_y)
        pts[heel_i, :2] = (hx + off - (side or 1) * 0.02 * h, v)
        pts[toe_i, :2] = (hx + off + (side or 1) * 0.05 * h, v)

    d = a.target_dir or 1
    if a.arm == "punch":
        poses = {12: ("punch", a.punch), 11: ("guard", 0.0)} if d > 0 else {11: ("punch", a.punch), 12: ("guard", 0.0)}
    elif a.arm == "knife":
        poses = {12: ("knife", 0.0), 11: ("rest", 0.0)} if d > 0 else {11: ("knife", 0.0), 12: ("rest", 0.0)}
    else:
        poses = {11: (a.arm, 0.0), 12: (a.arm, 0.0)}
    for s_i, e_i, w_i, hand, sgn in ((11, 13, 15, (17, 19, 21), 1), (12, 14, 16, (18, 20, 22), -1)):
        pose, e = poses[s_i]
        direction = d   # 揮拳／持刀／護頭都朝向對手
        elbow, wrist = _arm(pts[s_i, :2].copy(), h, pose, direction, e, sgn * math.sin(phase))
        pts[e_i, :2] = elbow
        pts[w_i, :2] = wrist
        for k, hi in enumerate(hand):
            pts[hi, :2] = wrist + np.array([direction * 0.02 * h, (k - 1) * 0.012 * h])
    return pts


def bbox_of(pts: np.ndarray, v: float) -> List[int]:
    h = height_at(v)
    x1, x2 = float(pts[:, 0].min()) - 0.04 * h, float(pts[:, 0].max()) + 0.04 * h
    return [int(x1), int(v - h * 1.02), int(x2), int(v)]


# ---------------------------------------------------------------------------
# 劇本
# ---------------------------------------------------------------------------

def _lerp(a: float, b: float, x: float) -> float:
    return a + (b - a) * min(1.0, max(0.0, x))


def _punch_ext(t: float, period: float = 0.9) -> float:
    """快速出拳：靜止 → 0.1 秒內伸直 → 停留 → 收回。"""
    x = t % period
    if x < 0.35:
        return 0.0
    if x < 0.45:
        return (x - 0.35) / 0.10
    if x < 0.65:
        return 1.0
    return max(0.0, 1.0 - (x - 0.65) / 0.25)


def scene_calm(t: float) -> List[ActorSpec]:
    va, vb = 560.0, 470.0
    return [
        ActorSpec(0, -40 + 0.9 * px_per_m(va) * t, va, facing=1, moving=True, color=(120, 110, 90)),
        ActorSpec(1, W + 40 - 1.0 * px_per_m(vb) * t, vb, facing=-1, moving=True, color=(90, 120, 110)),
    ]


def scene_follow(t: float) -> List[ActorSpec]:
    cx, cy, ra, rb = 640.0, 520.0, 450.0, 90.0
    omega = TAU / 16.0
    lag = 0.57   # 約落後 1.5 m

    def on_path(theta: float) -> Tuple[float, float, int]:
        u = cx + ra * math.cos(theta)
        v = cy + rb * math.sin(theta)
        du = -ra * math.sin(theta)
        return u, v, (1 if du > 0 else -1)

    ua, va, fa = on_path(omega * t)
    ub, vb, fb = on_path(omega * t - lag)
    return [
        ActorSpec(0, ua, va, facing=fa, moving=True, color=(120, 110, 90)),
        ActorSpec(1, ub, vb, facing=fb, moving=True, color=(60, 60, 150)),
    ]


def scene_fight(t: float) -> List[ActorSpec]:
    v = 560.0
    m = px_per_m(v)
    victim_u = 760.0
    actors = [ActorSpec(0, victim_u, v, facing=-1, arm="guard" if t > 3.5 else "rest",
                        target_dir=-1, color=(120, 110, 90))]
    if t < 3.0:
        u, moving, arm, e = 220.0, False, "rest", 0.0
    elif t < 4.2:                                   # 2.5 m/s 衝刺貼近到 0.4 m
        u, moving, arm, e = _lerp(220.0, victim_u - 0.4 * m, (t - 3.0) / 1.2), True, "rest", 0.0
    elif t < 12.0:                                  # 反覆後退 → 突進 → 揮拳
        c = (t - 4.2) % 1.6
        if c < 0.6:
            u = _lerp(victim_u - 0.4 * m, victim_u - 0.9 * m, c / 0.6)
        elif c < 0.8:
            u = _lerp(victim_u - 0.9 * m, victim_u - 0.35 * m, (c - 0.6) / 0.2)
        else:
            u = victim_u - 0.35 * m
        moving, arm, e = c < 0.8, "punch", _punch_ext(max(0.0, c - 0.5), 0.55)
    else:                                           # 逃離
        u, moving, arm, e = _lerp(victim_u - 0.35 * m, -80.0, (t - 12.0) / 2.5), True, "rest", 0.0
    actors.append(ActorSpec(1, u, v, facing=1 if t < 12 else -1, arm=arm, punch=e, moving=moving,
                            target_dir=1, color=(50, 50, 170)))
    return actors


def scene_weapon(t: float) -> List[ActorSpec]:
    v = 560.0
    m = px_per_m(v)
    victim_u = 420.0
    if t < 3.0:
        u, moving = _lerp(W + 40, 950.0, t / 3.0), True
    elif t < 6.0:
        u, moving = 950.0, False
    else:
        u, moving = _lerp(950.0, victim_u + 0.8 * m, (t - 6.0) / 4.0), t < 10.0
    attacker = ActorSpec(1, u, v, facing=-1, arm="knife" if t >= 3.0 else "rest", moving=moving,
                         target_dir=-1, color=(40, 40, 160))
    victim = ActorSpec(0, victim_u, v, facing=1 if t < 8 else -1, arm="guard" if t > 9 else "rest",
                       target_dir=1, color=(120, 110, 90))
    return [victim, attacker]


def scene_group(t: float) -> List[ActorSpec]:
    v = 560.0
    finals = [470.0, 580.0, 700.0, 810.0]
    starts = [-60.0, 250.0, 1050.0, W + 60]
    actors = []
    for i, (s, f) in enumerate(zip(starts, finals)):
        x = min(1.0, t / 4.0)
        u = _lerp(s, f, x)
        facing = 1 if u < 640 else -1
        arm = "guard" if t > 6 and i in (1, 2) else "rest"
        actors.append(ActorSpec(i, u, v + (i % 2) * 8, facing=facing, arm=arm, moving=x < 1.0,
                                target_dir=facing, color=(80 + 25 * i, 100, 140 - 20 * i)))
    return actors


def scene_loiter(t: float) -> List[ActorSpec]:
    u = 640 + 300 * math.sin(0.37 * t) + 120 * math.sin(1.1 * t + 1.0)
    v = 540 + 50 * math.sin(0.53 * t + 2.0)
    du = 300 * 0.37 * math.cos(0.37 * t) + 120 * 1.1 * math.cos(1.1 * t + 1.0)
    return [ActorSpec(0, u, v, facing=1 if du > 0 else -1, moving=True, color=(100, 100, 100))]


SCENES: Dict[str, Tuple[float, Callable[[float], List[ActorSpec]], str]] = {
    "calm": (12.0, scene_calm, "路人經過"),
    "follow": (24.0, scene_follow, "尾隨"),
    "fight": (16.0, scene_fight, "衝刺貼近後揮拳"),
    "weapon": (16.0, scene_weapon, "持刀逼近"),
    "group": (14.0, scene_group, "群聚對峙"),
    "loiter": (40.0, scene_loiter, "徘徊"),
}
PLAYLISTS: Dict[str, List[str]] = {
    "demo": ["calm", "follow", "fight", "weapon", "group"],
    **{k: [k] for k in SCENES},
}
SCENE_EN = {"calm": "CALM", "follow": "FOLLOWING", "fight": "ASSAULT", "weapon": "WEAPON",
            "group": "GROUP", "loiter": "LOITERING"}


# ---------------------------------------------------------------------------
# 繪圖
# ---------------------------------------------------------------------------

_LIMBS = [(11, 12), (11, 23), (12, 24), (23, 24), (11, 13), (13, 15), (12, 14), (14, 16),
          (23, 25), (25, 27), (24, 26), (26, 28), (27, 31), (28, 32)]


def _background() -> np.ndarray:
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        if y < HORIZON:
            img[y, :] = (205, 195, 185)
        else:
            g = int(150 + 60 * (y - HORIZON) / (H - HORIZON))
            img[y, :] = (g, g, g - 8)
    vp = (W // 2, HORIZON)
    for x in range(-W, 2 * W, 160):
        cv2.line(img, vp, (x, H), (135, 135, 128), 1, cv2.LINE_AA)
    for k in range(1, 9):
        y = int(HORIZON + (H - HORIZON) * (k / 8) ** 1.6)
        cv2.line(img, (0, y), (W, y), (135, 135, 128), 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (W, HORIZON), (215, 205, 195), -1)
    return img


_BG: Optional[np.ndarray] = None


def render(specs: List[ActorSpec], skeletons: Dict[int, np.ndarray], caption: str) -> np.ndarray:
    global _BG
    if _BG is None:
        _BG = _background()
    img = _BG.copy()
    for a in sorted(specs, key=lambda s: s.v):
        pts = skeletons[a.actor]
        h = height_at(a.v)
        th = max(3, int(h * 0.045))
        cv2.ellipse(img, (int(a.u), int(a.v)), (int(0.12 * h), int(0.025 * h)), 0, 0, 360, (110, 110, 105), -1)
        for i, j in _LIMBS:
            cv2.line(img, tuple(int(c) for c in pts[i, :2]), tuple(int(c) for c in pts[j, :2]), a.color, th, cv2.LINE_AA)
        cv2.circle(img, (int(pts[0, 0]), int(pts[0, 1])), int(0.065 * h), a.color, -1, cv2.LINE_AA)
        if a.arm == "knife":
            w = 16 if a.target_dir > 0 else 15
            tip = pts[w, :2] + np.array([a.target_dir * 0.13 * h, -0.02 * h])
            cv2.line(img, tuple(int(c) for c in pts[w, :2]), tuple(int(c) for c in tip), (210, 210, 220), max(2, th // 2), cv2.LINE_AA)
    cv2.rectangle(img, (W - 430, 12), (W - 12, 44), (40, 40, 40), -1)
    cv2.putText(img, caption, (W - 420, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# 來源物件
# ---------------------------------------------------------------------------

class SimulatedSource:
    """與 cv2.VideoCapture 類似的介面，另外回傳劇本產生的人物與武器觀測。"""

    is_simulated = True

    def __init__(self, scenario: str, cam_index: int = 0, fps: float = FPS):
        name = scenario.strip().lower() or "demo"
        if name not in PLAYLISTS:
            raise ValueError(f"未知的模擬情境 {name}，可用：{', '.join(PLAYLISTS)}")
        self.name = name
        self.playlist = PLAYLISTS[name]
        self.cycle_len = sum(SCENES[s][0] for s in self.playlist)
        self.fps = fps
        self.pid_base = 1000 * (cam_index + 1)
        self.t0 = time.time()
        self.last = 0.0
        self.rng = np.random.default_rng(cam_index)

    def isOpened(self) -> bool:  # noqa: N802（與 cv2 介面一致）
        return True

    def release(self) -> None:
        pass

    def _locate(self, t: float) -> Tuple[str, float, int]:
        cycle = int(t // self.cycle_len)
        x = t - cycle * self.cycle_len
        for k, scene in enumerate(self.playlist):
            dur = SCENES[scene][0]
            if x < dur:
                return scene, x, cycle * len(self.playlist) + k
            x -= dur
        return self.playlist[-1], 0.0, cycle * len(self.playlist)

    def read_sim(self) -> Tuple[bool, np.ndarray, List[PersonObservation], List[WeaponObservation]]:
        wait = 1.0 / self.fps - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        self.last = time.time()
        t = self.last - self.t0
        scene, local_t, seg_no = self._locate(t)
        specs = SCENES[scene][1](local_t)

        skeletons: Dict[int, np.ndarray] = {}
        persons: List[PersonObservation] = []
        weapons: List[WeaponObservation] = []
        for a in specs:
            pts = make_skeleton(a, local_t)
            skeletons[a.actor] = pts
            box = bbox_of(pts, a.v)
            if box[2] < 0 or box[0] > W:
                continue
            noisy = pts.copy()
            noisy[:, :2] += self.rng.normal(0.0, 0.8, size=(33, 2))   # 模擬骨架估計誤差
            pid = self.pid_base + seg_no * 10 + a.actor
            persons.append(PersonObservation(
                person_id=pid, track_id=pid, bbox=box, landmarks=noisy,
                visibility=np.ones(33, dtype=np.float32), det_conf=0.9,
            ))
            if a.arm == "knife":
                h = height_at(a.v)
                w = 16 if a.target_dir > 0 else 15
                cx = pts[w, 0] + a.target_dir * 0.08 * h
                cy = pts[w, 1]
                weapons.append(WeaponObservation(
                    label="knife", conf=float(np.clip(0.86 + self.rng.normal(0, 0.03), 0, 1)),
                    bbox=[int(cx - 0.08 * h), int(cy - 0.04 * h), int(cx + 0.08 * h), int(cy + 0.04 * h)],
                ))
        caption = f"SIMULATED CAM | {SCENE_EN[scene]} {local_t:4.1f}s / {SCENES[scene][0]:.0f}s"
        return True, render(specs, skeletons, caption), persons, weapons


def is_sim_source(source: str) -> bool:
    return str(source).lower().startswith("sim://")


def scenario_of(source: str) -> str:
    return str(source)[len("sim://"):] or "demo"
