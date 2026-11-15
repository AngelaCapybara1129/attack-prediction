"""MediaPipe Pose 33 點骨架：圖結構（ST-GCN 鄰接矩陣）與序列正規化工具。

runtime（後端即時推論）與 STGCN/scripts（資料前處理、訓練）共用此檔，
確保兩邊送進模型的張量格式完全一致。
"""
from typing import List, Optional, Sequence, Tuple

import numpy as np

NUM_JOINTS = 33

# MediaPipe landmark index
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

# MediaPipe 官方連線 + 補上臉部與軀幹的連結，讓整張圖連通
MEDIAPIPE_EDGES: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (0, 9), (0, 10),
    (0, 11), (0, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21), (17, 19),
    (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]

# 空間構型分群（spatial configuration partitioning）的重心：軀幹四點
CENTER_JOINTS: Tuple[int, ...] = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)


def hop_distance(num_node: int, edges: Sequence[Tuple[int, int]], max_hop: int = 1) -> np.ndarray:
    a = np.zeros((num_node, num_node))
    for i, j in edges:
        a[i, j] = 1
        a[j, i] = 1
    hop = np.full((num_node, num_node), np.inf)
    transfer = [np.linalg.matrix_power(a + np.eye(num_node), d) for d in range(max_hop + 1)]
    arrive = np.stack(transfer) > 0
    for d in range(max_hop, -1, -1):
        hop[arrive[d]] = d
    return hop


def _graph_distance_to_center(num_node: int, edges: Sequence[Tuple[int, int]]) -> np.ndarray:
    """BFS：每個節點到最近重心節點的跳數。"""
    adj: List[List[int]] = [[] for _ in range(num_node)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    dist = np.full(num_node, np.inf)
    queue = list(CENTER_JOINTS)
    for c in CENTER_JOINTS:
        dist[c] = 0
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if dist[v] == np.inf:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def normalize_digraph(a: np.ndarray) -> np.ndarray:
    dl = np.sum(a, 0)
    dn = np.zeros_like(a)
    for i in range(a.shape[0]):
        if dl[i] > 0:
            dn[i, i] = dl[i] ** (-1)
    return a @ dn


def build_spatial_adjacency(num_node: int = NUM_JOINTS, edges: Sequence[Tuple[int, int]] = MEDIAPIPE_EDGES) -> np.ndarray:
    """回傳 (3, V, V)：[自身, 向心(含同距), 離心]，與 Yan et al. (2018) 的 spatial strategy 相同。"""
    hop = hop_distance(num_node, edges, max_hop=1)
    adjacency = (hop <= 1).astype(np.float64)
    norm_adj = normalize_digraph(adjacency)
    center_dist = _graph_distance_to_center(num_node, edges)

    a_self = np.zeros((num_node, num_node))
    a_close = np.zeros((num_node, num_node))
    a_far = np.zeros((num_node, num_node))
    for i in range(num_node):
        for j in range(num_node):
            if hop[j, i] == 0:
                a_self[j, i] = norm_adj[j, i]
            elif hop[j, i] == 1:
                if center_dist[j] <= center_dist[i]:
                    a_close[j, i] = norm_adj[j, i]
                else:
                    a_far[j, i] = norm_adj[j, i]
    return np.stack([a_self, a_close, a_far]).astype(np.float32)


# ---------------------------------------------------------------------------
# 座標轉換與序列正規化
# ---------------------------------------------------------------------------

def crop_landmarks_to_frame(landmarks: Sequence[Sequence[float]], pose_box: Sequence[int]) -> np.ndarray:
    """MediaPipe 輸出為裁切圖的正規化座標，轉成整張畫面的像素座標 (33,3)。z 以裁切寬度換算成像素尺度。"""
    lm = np.asarray(landmarks, dtype=np.float32)
    x1, y1, x2, y2 = [float(v) for v in pose_box]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    out = np.empty((lm.shape[0], 3), dtype=np.float32)
    out[:, 0] = x1 + lm[:, 0] * w
    out[:, 1] = y1 + lm[:, 1] * h
    out[:, 2] = lm[:, 2] * w
    return out


def expand_box(box: Sequence[int], frame_w: Optional[int] = None, frame_h: Optional[int] = None,
               scale_x: float = 0.18, scale_y: float = 0.22) -> List[int]:
    x1, y1, x2, y2 = [int(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = int(bw * scale_x), int(bh * scale_y)
    nx1, ny1, nx2, ny2 = x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y
    nx1, ny1 = max(0, nx1), max(0, ny1)
    if frame_w is not None:
        nx2 = min(frame_w - 1, nx2)
    if frame_h is not None:
        ny2 = min(frame_h - 1, ny2)
    return [nx1, ny1, nx2, ny2]


def fill_missing(seq: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """沿時間軸線性內插遺失幀；兩端以最近的有效幀補齊。seq: (T,V,C)。"""
    t = seq.shape[0]
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return np.zeros_like(seq)
    if idx.size == t:
        return seq
    out = seq.copy()
    grid = np.arange(t)
    flat = out.reshape(t, -1)
    src = seq.reshape(t, -1)
    for k in range(flat.shape[1]):
        flat[:, k] = np.interp(grid, idx, src[idx, k])
    return flat.reshape(seq.shape)


def resample_by_time(
    timestamps: Sequence[float],
    frames: np.ndarray,
    valid: np.ndarray,
    t_end: float,
    window_sec: float,
    num_samples: int,
    max_gap_sec: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray]:
    """把不等間隔的觀測依時間重取樣成固定 T 幀，讓不同 FPS 的來源輸入格式一致。

    frames: (N, ...)；回傳 (num_samples, ...) 與 mask (num_samples,)。
    """
    ts = np.asarray(timestamps, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    target = np.linspace(t_end - window_sec, t_end, num_samples)
    out_shape = (num_samples,) + frames.shape[1:]
    if ts.size == 0 or not valid.any():
        return np.zeros(out_shape, dtype=np.float32), np.zeros(num_samples, dtype=np.float32)

    vt = ts[valid]
    vf = frames[valid].reshape(int(valid.sum()), -1).astype(np.float64)
    out = np.empty((num_samples, vf.shape[1]), dtype=np.float32)
    for k in range(vf.shape[1]):
        out[:, k] = np.interp(target, vt, vf[:, k])

    # mask：目標時間點附近 max_gap_sec 內必須有真實觀測
    pos = np.searchsorted(vt, target)
    left = np.abs(target - vt[np.clip(pos - 1, 0, vt.size - 1)])
    right = np.abs(vt[np.clip(pos, 0, vt.size - 1)] - target)
    mask = (np.minimum(left, right) <= max_gap_sec).astype(np.float32)
    return out.reshape(out_shape), mask


def normalize_skeleton_sequence(seq_px: np.ndarray, valid: Optional[np.ndarray] = None) -> np.ndarray:
    """(T,33,3) 像素座標 → ST-GCN 輸入 (3,T,33,1)。

    以每幀骨盆中心為原點、以視窗內軀幹長度中位數為尺度，消除位置與遠近差異。
    """
    seq = np.asarray(seq_px, dtype=np.float32)
    t = seq.shape[0]
    if valid is None:
        valid = np.abs(seq).sum(axis=(1, 2)) > 0
    valid = np.asarray(valid, dtype=bool)
    if not valid.any():
        return np.zeros((3, t, NUM_JOINTS, 1), dtype=np.float32)

    seq = fill_missing(seq, valid)
    hip = (seq[:, L_HIP, :2] + seq[:, R_HIP, :2]) / 2.0
    shoulder = (seq[:, L_SHOULDER, :2] + seq[:, R_SHOULDER, :2]) / 2.0
    torso = np.linalg.norm(shoulder - hip, axis=1)
    torso_valid = torso[valid]
    scale = float(np.median(torso_valid)) if torso_valid.size else 1.0
    if not np.isfinite(scale) or scale < 1e-3:
        scale = 1.0

    out = np.empty_like(seq)
    out[:, :, :2] = (seq[:, :, :2] - hip[:, None, :]) / scale
    out[:, :, 2] = seq[:, :, 2] / scale
    out = np.clip(out, -10.0, 10.0)
    return out.transpose(2, 0, 1)[..., None].astype(np.float32)


def normalize_pair_sequence(seqs_px: Sequence[np.ndarray], valids: Sequence[np.ndarray]) -> np.ndarray:
    """多人（通常 2 人）像素骨架 (T,33,3) → ST-GCN 輸入 (3,T,33,M)，保留人與人之間的相對位置。

    與單人版不同：所有人共用同一個座標原點（視窗內所有人骨盆中心的平均位置，固定不隨幀移動）
    與尺度（所有人軀幹長度中位數），因此「兩人距離、誰衝向誰、整體移動」都留在座標裡。
    缺人或整段無骨架的位置填 0（ST-GCN 慣例）。
    """
    m = len(seqs_px)
    t = np.asarray(seqs_px[0]).shape[0]
    filled, oks = [], []
    for seq, valid in zip(seqs_px, valids):
        seq = np.asarray(seq, dtype=np.float32)
        valid = np.asarray(valid, dtype=bool)
        if valid.any():
            filled.append(fill_missing(seq, valid))
            oks.append(valid)
        else:
            filled.append(None)
            oks.append(valid)
    present = [i for i in range(m) if filled[i] is not None]
    out = np.zeros((3, t, NUM_JOINTS, m), dtype=np.float32)
    if not present:
        return out

    hips, torsos = [], []
    for i in present:
        s = filled[i]
        hip = (s[:, L_HIP, :2] + s[:, R_HIP, :2]) / 2.0
        sh = (s[:, L_SHOULDER, :2] + s[:, R_SHOULDER, :2]) / 2.0
        hips.append(hip[oks[i]])
        torsos.append(np.linalg.norm(sh - hip, axis=1)[oks[i]])
    origin = np.concatenate(hips).mean(axis=0)
    tv = np.concatenate(torsos)
    scale = float(np.median(tv)) if tv.size else 1.0
    if not np.isfinite(scale) or scale < 1e-3:
        scale = 1.0

    for i in present:
        s = filled[i]
        o = np.empty_like(s)
        o[:, :, :2] = (s[:, :, :2] - origin[None, None, :]) / scale
        o[:, :, 2] = s[:, :, 2] / scale
        out[..., i] = np.clip(o, -10.0, 10.0).transpose(2, 0, 1)
    return out


def torso_length(lm_px: np.ndarray) -> float:
    hip = (lm_px[L_HIP, :2] + lm_px[R_HIP, :2]) / 2.0
    sh = (lm_px[L_SHOULDER, :2] + lm_px[R_SHOULDER, :2]) / 2.0
    return float(np.linalg.norm(sh - hip))


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """∠abc（度）。"""
    v1 = a[:2] - b[:2]
    v2 = c[:2] - b[:2]
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))
