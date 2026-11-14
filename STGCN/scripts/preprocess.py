"""骨架 session → 雙通道訓練樣本。

輸入：
  - STGCN/data/interim/sessions/*（extract_skeletons.py 從影片資料集產生）
  - backend/data/sessions/*（即時系統 DataRecorder 錄下的資料，可在 windows.jsonl 手動標註 action_label / risk_label）
流程（研究計畫：攻擊意圖預測模型）：
  1. 依時間重播每個 session，以與即時系統相同的 FeatureEngine / RiskEngine 計算每人每幀 X1–X9 與等級
  2. 以 5 秒滑動視窗（步長 1 秒）切出骨架序列 → 重取樣成 150 幀 → 正規化 (3,150,33,1)
  3. 同一時間軸上的 X1–X9 序列 (150,9) 作為第二通道
  4. 標籤：下一秒的 X1–X9（next_x）、風險分數（risk）、等級（level）、行為類型（action）
輸出：data/processed/{train,val,test}/samples/*.npz + labels.csv + dataset_meta.json
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config  # noqa: E402

import numpy as np  # noqa: E402

from app.core.intent.constants import ACTION_CLASSES, LEVEL_INDEX, LEVELS  # noqa: E402
from app.core.intent.context import SpatialContext  # noqa: E402
from app.core.intent.features import FeatureEngine, PersonObservation, WeaponObservation  # noqa: E402
from app.core.intent.risk import RiskEngine  # noqa: E402
from app.core.intent.skeleton import (  # noqa: E402
    NUM_JOINTS, crop_landmarks_to_frame, expand_box, normalize_skeleton_sequence, resample_by_time,
)

NEUTRAL_CONTEXT = {"default_region_weight": 0.4, "time_weights": [{"start": 0, "end": 24, "weight": 0.5}]}


# ---------------------------------------------------------------------------
# 讀取
# ---------------------------------------------------------------------------

def find_sessions(dirs: List[Path]) -> List[Path]:
    out = []
    for d in dirs:
        if not d.is_dir():
            continue
        for s in sorted(d.iterdir()):
            if (s / "frames.jsonl").is_file() and (s / "frames.jsonl").stat().st_size > 0:
                out.append(s)
    return out


def load_records(session: Path) -> List[Dict]:
    recs = []
    with open(session / "frames.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return recs


def load_meta(session: Path) -> Dict:
    p = session / "meta.json"
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def load_window_labels(session: Path) -> Dict[int, List[Dict]]:
    """windows.jsonl 中人工標註的 action_label / risk_label。"""
    out: Dict[int, List[Dict]] = defaultdict(list)
    p = session / "windows.jsonl"
    if not p.is_file():
        return out
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            try:
                w = json.loads(line)
            except json.JSONDecodeError:
                continue
            if w.get("action_label") or w.get("risk_label"):
                out[int(w["person_id"])].append(w)
    return out


def record_landmarks_px(r: Dict, fw: int, fh: int) -> Optional[np.ndarray]:
    if r.get("landmarks_px"):
        arr = np.asarray(r["landmarks_px"], dtype=np.float32)
        return arr if arr.shape == (NUM_JOINTS, 3) else None
    if r.get("landmarks"):
        pose_box = r.get("pose_box") or expand_box(r["bbox"], fw, fh)
        arr = crop_landmarks_to_frame(r["landmarks"], pose_box)
        return arr if arr.shape == (NUM_JOINTS, 3) else None
    return None


# ---------------------------------------------------------------------------
# 重播：計算每人時間軸
# ---------------------------------------------------------------------------

def replay(records: List[Dict], meta: Dict) -> Dict[int, List[Tuple[float, Optional[np.ndarray], np.ndarray, str, float]]]:
    ctx = SpatialContext(NEUTRAL_CONTEXT) if meta.get("source") == "video" else SpatialContext.load()
    default_size = meta.get("frame_size") or list(config.DEFAULT_FRAME_SIZE)
    frames: Dict[Tuple[str, int], List[Dict]] = defaultdict(list)
    for r in records:
        frames[(str(r.get("camera_id", "cam0")), int(r.get("frame_idx", 0)))].append(r)
    ordered = sorted(frames.items(), key=lambda kv: float(kv[1][0]["timestamp"]))

    engines: Dict[str, FeatureEngine] = {}
    risk = RiskEngine()
    timeline: Dict[int, List] = defaultdict(list)
    for (cam, _), recs in ordered:
        ts = float(recs[0]["timestamp"])
        fw, fh = recs[0].get("frame_size") or default_size
        persons, seen = [], set()
        for r in recs:
            pid = int(r["person_id"])
            if pid in seen:
                continue
            seen.add(pid)
            vis = r.get("visibility")
            persons.append(PersonObservation(
                person_id=pid, track_id=int(r.get("track_id", pid)), bbox=r["bbox"],
                landmarks=record_landmarks_px(r, fw, fh),
                visibility=None if vis is None else np.asarray(vis, dtype=np.float32),
                det_conf=float(r.get("det_conf") or 1.0),
            ))
        weapons = [WeaponObservation(label=w.get("label", "weapon"), conf=float(w.get("conf", 0)), bbox=w["bbox"])
                   for w in (recs[0].get("weapons") or [])]
        fe = engines.setdefault(cam, FeatureEngine(cam, ctx))
        feats = fe.update(ts, int(fw), int(fh), persons, weapons, risk.history_scores())
        lm_by_pid = {p.person_id: p.landmarks for p in persons}
        for pid, fr in feats.items():
            res = risk.evaluate(pid, ts, fr.x)
            timeline[pid].append((ts, lm_by_pid.get(pid), fr.x.copy(), res.level, res.score))
    return timeline


# ---------------------------------------------------------------------------
# 切樣本
# ---------------------------------------------------------------------------

def derive_action(level: str) -> str:
    idx = LEVEL_INDEX[level]
    if idx >= 3:
        return "dangerous"
    if idx >= 1:
        return "suspicious"
    return "normal"


def window_annotation(labels: List[Dict], t: float) -> Tuple[Optional[str], Optional[str]]:
    for w in labels:
        if float(w.get("start_ts", -1e18)) <= t <= float(w.get("end_ts", 1e18)) + 1.0:
            return w.get("action_label"), w.get("risk_label")
    return None, None


def build_samples(session: Path, args) -> List[Dict]:
    meta = load_meta(session)
    records = load_records(session)
    if not records:
        return []
    timeline = replay(records, meta)
    ann = load_window_labels(session)
    video_label = config.LABEL_ALIASES.get(str(meta.get("label") or "").lower())
    samples = []
    for pid, tl in timeline.items():
        if len(tl) < 8:
            continue
        ts = np.array([t[0] for t in tl])
        lms = [t[1] for t in tl]
        xs = np.stack([t[2] for t in tl])
        levels = [t[3] for t in tl]
        scores = np.array([t[4] for t in tl])
        valid = np.array([lm is not None for lm in lms])
        frames = np.stack([lm if lm is not None else np.zeros((NUM_JOINTS, 3), np.float32) for lm in lms])

        t = ts[0] + config.WINDOW_SEC
        while t <= ts[-1] - config.HORIZON_SEC + 1e-6:
            win = (ts >= t - config.WINDOW_SEC) & (ts <= t)
            fut = (ts > t) & (ts <= t + config.HORIZON_SEC)
            if win.sum() >= 8 and fut.any() and valid[win].mean() >= config.MIN_VALID_RATIO \
                    and ts[win][-1] - ts[win][0] >= 0.6 * config.WINDOW_SEC:
                skel_seq, mask = resample_by_time(ts[win], frames[win], valid[win], t, config.WINDOW_SEC, config.SEQ_LEN)
                if mask.mean() >= config.MIN_VALID_RATIO:
                    xseq, _ = resample_by_time(ts[win], xs[win], np.ones(int(win.sum()), bool), t,
                                               config.WINDOW_SEC, config.SEQ_LEN, max_gap_sec=1e9)
                    cur_i = int(np.flatnonzero(win)[-1])
                    fut_idx = np.flatnonzero(fut)
                    level = levels[int(fut_idx[-1])]
                    risk = float(scores[fut_idx].mean())
                    a_lab, r_lab = window_annotation(ann.get(pid, []), t)
                    if r_lab in LEVEL_INDEX:
                        level = r_lab
                    if a_lab in ACTION_CLASSES:
                        action, src = a_lab, "manual"
                    elif video_label:
                        action, src = video_label, "video"
                    else:
                        action, src = derive_action(level), "derived"
                    samples.append({
                        "skel": normalize_skeleton_sequence(skel_seq, mask > 0),
                        "xseq": xseq.astype(np.float32),
                        "cur_x": xs[cur_i].astype(np.float32),
                        "next_x": xs[fut_idx].mean(axis=0).astype(np.float32),
                        "risk": np.float32(risk),
                        "cur_risk": np.float32(scores[cur_i]),
                        "level": np.int64(LEVEL_INDEX[level]),
                        "cur_level": np.int64(LEVEL_INDEX[levels[cur_i]]),
                        "action": np.int64(ACTION_CLASSES.index(action)),
                        "pid": pid, "t": float(t), "label_source": src,
                    })
            t += config.STRIDE_SEC
    return samples


def assign_split(session: Path, meta: Dict, ratios=(0.7, 0.15, 0.15)) -> str:
    """以 session（影片）為單位切分，避免同一段影片同時出現在訓練與測試集。"""
    h = int(hashlib.md5(session.name.encode("utf-8")).hexdigest(), 16) % 1000 / 1000.0
    split = meta.get("split")
    if split == "train":
        return "train"
    if split in ("val", "test"):
        # RWF-2000 只有 train/val：val 再對半分成 val / test
        return "val" if h < 0.5 else "test"
    if h < ratios[0]:
        return "train"
    return "val" if h < ratios[0] + ratios[1] else "test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*", default=None,
                    help="session 根目錄（預設：STGCN/data/interim/sessions 與 backend/data/sessions）")
    ap.add_argument("--output", type=str, default=str(config.PROCESSED_DIR))
    ap.add_argument("--clean", action="store_true", help="先清空輸出資料夾的 train/val/test")
    ap.add_argument("--window-sec", type=float, default=config.WINDOW_SEC,
                    help="視窗長度（RWF-2000 每段僅 5 秒，建議 --window-sec 3 --horizon-sec 1 --seq-len 90）")
    ap.add_argument("--horizon-sec", type=float, default=config.HORIZON_SEC)
    ap.add_argument("--seq-len", type=int, default=config.SEQ_LEN)
    ap.add_argument("--stride-sec", type=float, default=config.STRIDE_SEC)
    args = ap.parse_args()

    config.WINDOW_SEC = args.window_sec
    config.HORIZON_SEC = args.horizon_sec
    config.SEQ_LEN = args.seq_len
    config.STRIDE_SEC = args.stride_sec
    config.MODEL_CONFIG.update({"window_sec": args.window_sec, "horizon_sec": args.horizon_sec, "seq_len": args.seq_len})

    roots = [Path(p) for p in args.sessions] if args.sessions else [config.INTERIM_SESSIONS_DIR, *config.BACKEND_SESSION_DIRS]
    sessions = find_sessions(roots)
    print(f"找到 {len(sessions)} 個 session：{[str(r) for r in roots]}")
    out_root = Path(args.output)
    if args.clean:
        for split in ("train", "val", "test"):
            shutil.rmtree(out_root / split, ignore_errors=True)

    writers, files = {}, {}
    for split in ("train", "val", "test"):
        (out_root / split / "samples").mkdir(parents=True, exist_ok=True)
        files[split] = open(out_root / split / "labels.csv", "w", newline="", encoding="utf-8")
        writers[split] = csv.writer(files[split])
        writers[split].writerow(["sample_id", "label", "level", "risk", "session", "pid", "t", "label_source"])

    counts = {s: Counter() for s in ("train", "val", "test")}
    level_counts = {s: Counter() for s in ("train", "val", "test")}
    try:
        for i, session in enumerate(sessions, 1):
            meta = load_meta(session)
            split = assign_split(session, meta)
            samples = build_samples(session, args)
            for j, s in enumerate(samples):
                sid = f"{session.name}_p{s['pid']}_{j:05d}"
                np.savez_compressed(
                    out_root / split / "samples" / f"{sid}.npz",
                    skel=s["skel"], xseq=s["xseq"], cur_x=s["cur_x"], next_x=s["next_x"],
                    risk=s["risk"], cur_risk=s["cur_risk"], level=s["level"], cur_level=s["cur_level"],
                    action=s["action"],
                )
                writers[split].writerow([sid, int(s["action"]), int(s["level"]), round(float(s["risk"]), 4),
                                         session.name, s["pid"], round(s["t"], 2), s["label_source"]])
                counts[split][ACTION_CLASSES[int(s["action"])]] += 1
                level_counts[split][LEVELS[int(s["level"])]] += 1
            print(f"[{i}/{len(sessions)}] {session.name} → {split}: {len(samples)} 樣本")
    finally:
        for f in files.values():
            f.close()

    summary = {
        "config": {k: v for k, v in config.MODEL_CONFIG.items()},
        "stride_sec": config.STRIDE_SEC,
        "min_valid_ratio": config.MIN_VALID_RATIO,
        "action_counts": {s: dict(c) for s, c in counts.items()},
        "level_counts": {s: dict(c) for s, c in level_counts.items()},
    }
    (out_root / "dataset_meta.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if sum(sum(c.values()) for c in counts.values()) == 0:
        print(f"⚠️ 沒有產生任何樣本：請確認 session 內有骨架資料（has_pose=true），"
              f"且同一人物連續出現超過 {config.WINDOW_SEC + config.HORIZON_SEC:.0f} 秒（可調小 --window-sec）")


if __name__ == "__main__":
    main()
