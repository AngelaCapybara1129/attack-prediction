"""骨架 session → 「整段影片 × 雙人」攻擊類型分類樣本（RWF-2000 等影片層級標籤資料集）。

與 preprocess.py（逐人 3 秒視窗 + 預測下一秒）的差異：
  1. 時間窗用滿整段影片（RWF-2000 每段 5 秒）；超過 --window-sec 的長影片以 --stride-sec 切成多段，
     短於 --window-sec 的片段（Hockey 約 1.6 秒、Movies 約 2 秒）以片段本身長度為視窗，同樣重取樣成 --seq-len 幀。
  2. 每個視窗挑「互動最強的兩個人」（兩人同時有骨架的時間 × 距離越近分數越高），
     兩人放在同一個座標系（normalize_pair_sequence），保留相對位置與接近／衝撞動作 → skel (3,T,33,2)。
     只有一人時第二人填 0。
  3. 不預測下一秒：只訓練攻擊類型（train.py --task classify）。
  4. 另外存下規則系統（X1–X9 + AHP）在視窗內的最高等級 rule_level，評估時作為基線比較。

    python scripts/preprocess_clips.py --clean
    python scripts/train.py --task classify --data data/processed_clips
    python scripts/evaluate.py --data data/processed_clips --model checkpoints/attack_classifier_rwf2000.pt
"""
import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import config  # noqa: E402

import numpy as np  # noqa: E402

from app.core.intent.constants import ATTACK_TAXONOMY, LEVEL_INDEX, LEVELS  # noqa: E402
from app.core.intent.skeleton import NUM_JOINTS, normalize_pair_sequence, resample_by_time  # noqa: E402
from preprocess import (  # noqa: E402
    assign_split, build_class_list, find_sessions, load_meta, load_records, map_label, replay, session_raw_label,
)


def person_boxes(records: List[Dict]) -> Dict[int, Dict[float, List[float]]]:
    out: Dict[int, Dict[float, List[float]]] = defaultdict(dict)
    for r in records:
        out[int(r["person_id"])][round(float(r["timestamp"]), 4)] = r["bbox"]
    return out


def pair_score(ba: Dict[float, List[float]], bb: Dict[float, List[float]], t0: float, t1: float) -> float:
    """兩人在視窗內同時出現的幀數 ÷ (1 + 距離中位數)；距離 = 腳底中心距離 ÷ 平均身高。"""
    common = [t for t in ba if t0 <= t <= t1 and t in bb]
    if not common:
        return 0.0
    d = []
    for t in common:
        a, b = ba[t], bb[t]
        h = max(1.0, ((a[3] - a[1]) + (b[3] - b[1])) / 2)
        d.append(np.hypot((a[0] + a[2]) / 2 - (b[0] + b[2]) / 2, a[3] - b[3]) / h)
    return len(common) / (1.0 + float(np.median(d)))


def windows_for(t_start: float, t_end: float, args) -> List[tuple]:
    dur = t_end - t_start
    if dur < args.min_clip_sec:
        return []
    if dur <= args.window_sec * 1.2:
        return [(t_start, t_end)]
    out, t = [], t_start
    while t + args.window_sec <= t_end + 1e-6:
        out.append((t, t + args.window_sec))
        t += args.stride_sec
    return out


def build_clip_samples(session: Path, args, classes: List[str], stats: Counter) -> List[Dict]:
    meta = load_meta(session)
    label = map_label(session_raw_label(meta), args.taxonomy)
    if label is None or label not in classes:
        stats["sessions_no_label"] += 1
        return []
    records = load_records(session)
    if not records:
        stats["sessions_empty"] += 1
        return []
    timeline = replay(records, meta)
    boxes = person_boxes(records)
    fps = float(meta.get("fps") or 30.0)
    t_start = 0.0
    t_end = max(float(meta.get("frames") or 0) / fps, max(float(r["timestamp"]) for r in records))

    samples = []
    for (w0, w1) in windows_for(t_start, t_end, args):
        win_len = w1 - w0
        # 每人在視窗內的骨架覆蓋
        cand = []
        for pid, tl in timeline.items():
            ts = np.array([x[0] for x in tl])
            sel = (ts >= w0) & (ts <= w1)
            if not sel.any():
                continue
            valid = np.array([x[1] is not None for x in tl])[sel]
            if valid.sum() >= 3:
                cand.append(pid)
        if not cand:
            stats["windows_no_skeleton"] += 1
            continue
        if len(cand) == 1:
            pair = [cand[0]]
        else:
            best, pair = -1.0, cand[:2]
            for i in range(len(cand)):
                for j in range(i + 1, len(cand)):
                    s = pair_score(boxes[cand[i]], boxes[cand[j]], w0, w1)
                    if s > best:
                        best, pair = s, [cand[i], cand[j]]
        seqs, valids, xs_list, levels = [], [], [], []
        for pid in pair:
            tl = timeline[pid]
            ts = np.array([x[0] for x in tl])
            lm = np.stack([x[1] if x[1] is not None else np.zeros((NUM_JOINTS, 3), np.float32) for x in tl])
            ok = np.array([x[1] is not None for x in tl])
            xv = np.stack([x[2] for x in tl])
            seq, mask = resample_by_time(ts, lm, ok, w1, win_len, args.seq_len)
            xseq, _ = resample_by_time(ts, xv, np.ones(len(tl), bool), w1, win_len, args.seq_len, max_gap_sec=1e9)
            # 視窗外的時間點不要被外插：超出此人出現範圍的幀視為缺漏
            grid = np.linspace(w0, w1, args.seq_len)
            inside = (grid >= ts.min() - 0.3) & (grid <= ts.max() + 0.3)
            mask = mask * inside
            xseq = xseq * inside[:, None]
            seqs.append(seq)
            valids.append(mask > 0)
            xs_list.append(xseq)
            in_win = (ts >= w0) & (ts <= w1)
            levels.extend(LEVEL_INDEX[x[3]] for x, keep in zip(tl, in_win) if keep)
        coverage = float(np.mean([v.mean() for v in valids]))
        if coverage < args.min_coverage:
            stats["windows_low_coverage"] += 1
            continue
        if len(pair) == 1:
            seqs.append(np.zeros_like(seqs[0]))
            valids.append(np.zeros(args.seq_len, bool))
            xs_list.append(np.zeros_like(xs_list[0]))
            stats["windows_single_person"] += 1
        skel = normalize_pair_sequence(seqs, valids)
        xseq = np.maximum(xs_list[0], xs_list[1]).astype(np.float32)   # 兩人中較危險者的指數
        rule_level = int(max(levels)) if levels else 0
        samples.append({
            "skel": skel, "xseq": xseq, "cur_x": xseq[-1], "next_x": xseq[-1],
            "risk": np.float32(0.0), "cur_risk": np.float32(0.0),
            "level": np.int64(rule_level), "cur_level": np.int64(rule_level), "rule_level": np.int64(rule_level),
            "action": np.int64(classes.index(label)), "pid": "-".join(str(p) for p in pair),
            "t": float(w1), "coverage": coverage, "n_persons": len(pair),
        })
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*", default=[str(config.INTERIM_SESSIONS_DIR)])
    ap.add_argument("--output", default=str(config.DATA_DIR / "processed_clips"))
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--taxonomy", choices=sorted(ATTACK_TAXONOMY), default="coarse")
    ap.add_argument("--window-sec", type=float, default=5.0, help="長影片的視窗長度；RWF-2000 片段本身即 5 秒")
    ap.add_argument("--stride-sec", type=float, default=2.5)
    ap.add_argument("--seq-len", type=int, default=75, help="重取樣幀數（5 秒 × 15 fps）")
    ap.add_argument("--min-clip-sec", type=float, default=1.0, help="短於此長度的影片略過")
    ap.add_argument("--min-coverage", type=float, default=0.25, help="兩人平均骨架覆蓋率下限")
    ap.add_argument("--max-per-session", type=int, default=8)
    args = ap.parse_args()

    roots = [Path(p) for p in args.sessions]
    sessions = find_sessions(roots)
    classes = build_class_list(sessions, args.taxonomy, None)
    print(f"找到 {len(sessions)} 個 session；攻擊類型類別（{args.taxonomy}）：{classes}")

    out_root = Path(args.output)
    if args.clean:
        for split in ("train", "val", "test"):
            shutil.rmtree(out_root / split, ignore_errors=True)
    files, writers = {}, {}
    for split in ("train", "val", "test"):
        (out_root / split / "samples").mkdir(parents=True, exist_ok=True)
        files[split] = open(out_root / split / "labels.csv", "w", newline="", encoding="utf-8")
        writers[split] = csv.writer(files[split])
        writers[split].writerow(["sample_id", "label", "level", "risk", "session", "pid", "t", "label_source",
                                 "rule_level", "coverage", "n_persons"])

    stats: Counter = Counter()
    counts = {s: Counter() for s in ("train", "val", "test")}
    try:
        for i, session in enumerate(sessions, 1):
            meta = load_meta(session)
            split = assign_split(session, meta)
            samples = build_clip_samples(session, args, classes, stats)
            if args.max_per_session and len(samples) > args.max_per_session:
                keep = sorted(set(np.linspace(0, len(samples) - 1, args.max_per_session).round().astype(int).tolist()))
                samples = [samples[k] for k in keep]
            for j, s in enumerate(samples):
                sid = f"{session.name}_w{j:02d}"
                np.savez_compressed(out_root / split / "samples" / f"{sid}.npz",
                                    **{k: v for k, v in s.items() if k not in ("pid", "t", "coverage", "n_persons")})
                writers[split].writerow([sid, int(s["action"]), int(s["level"]), 0.0, session.name, s["pid"],
                                         round(s["t"], 2), "video", int(s["rule_level"]), round(s["coverage"], 3),
                                         s["n_persons"]])
                counts[split][classes[int(s["action"])]] += 1
            if i % 200 == 0 or i == len(sessions):
                print(f"[{i}/{len(sessions)}] {dict((k, sum(v.values())) for k, v in counts.items())}")
    finally:
        for f in files.values():
            f.close()

    model_cfg = dict(config.MODEL_CONFIG)
    model_cfg.update({"action_classes": classes, "num_person": 2, "seq_len": args.seq_len,
                      "window_sec": args.window_sec, "horizon_sec": 0.0, "task": "classify"})
    summary = {
        "config": model_cfg,
        "mode": "clip_pair",
        "taxonomy": args.taxonomy,
        "params": vars(args) | {"sessions": [str(r) for r in roots]},
        "stats": dict(stats),
        "action_counts": {s: dict(c) for s, c in counts.items()},
    }
    (out_root / "dataset_meta.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("stats", "action_counts")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
