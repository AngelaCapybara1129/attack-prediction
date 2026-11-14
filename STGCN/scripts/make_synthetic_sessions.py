"""產生「合成」骨架 session，用來在沒有真實資料集時驗證 抽取 → 前處理 → 訓練 → 評估 流程是否正確。

⚠️ 只是劇本產生的火柴人骨架（backend/app/core/cv/simulator.py 的情境加上隨機變化），
   不能當作模型效果的依據；論文／報告的數據必須用 RWF-2000 等真實資料集。

情境 → 原始標籤：calm / follow / group / loiter → normal；fight → fight；weapon → weapon
每個變體隨機：播放速度、水平鏡像、水平位移、景深（人物大小）、骨架雜訊、武器信心值。

    python scripts/make_synthetic_sessions.py --variants 30
    python scripts/preprocess.py --sessions data/synthetic/sessions --output data/processed_synthetic --clean \
        --window-sec 3 --horizon-sec 1 --seq-len 90
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config  # noqa: E402

import numpy as np  # noqa: E402

from app.core.cv import simulator as sim  # noqa: E402

SCENE_LABELS = {"calm": "normal", "follow": "normal", "group": "normal", "loiter": "normal",
                "fight": "fight", "weapon": "weapon"}


def make_session(scene: str, k: int, rng: np.random.Generator, out_dir: Path, fps: float) -> int:
    dur, fn, _ = sim.SCENES[scene]
    speed = rng.uniform(0.8, 1.25)
    mirror = bool(rng.random() < 0.5)
    shift = rng.uniform(-120, 120)
    dv = rng.uniform(-40, 40)                 # 整體往前/後移，人物大小跟著改變
    noise = rng.uniform(0.5, 2.5)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    n_frames = int(dur / speed * fps)
    with open(out_dir / "frames.jsonl", "w", encoding="utf-8") as f:
        for i in range(n_frames):
            ts = i / fps
            local_t = ts * speed
            weapons = []
            recs = []
            for a in fn(local_t):
                a.u += shift
                a.v = float(np.clip(a.v + dv, sim.HORIZON + 140, sim.H - 5))
                pts = sim.make_skeleton(a, local_t)
                box = sim.bbox_of(pts, a.v)
                if mirror:
                    pts[:, 0] = sim.W - pts[:, 0]
                    box = [sim.W - box[2], box[1], sim.W - box[0], box[3]]
                if box[2] < 0 or box[0] > sim.W:
                    continue
                pts[:, :2] += rng.normal(0.0, noise, size=(33, 2))
                if a.arm == "knife":
                    h = sim.height_at(a.v)
                    w = 16 if a.target_dir > 0 else 15
                    cx, cy = pts[w, 0] + (-1 if mirror else 1) * a.target_dir * 0.08 * h, pts[w, 1]
                    weapons.append({"label": "knife", "conf": round(float(np.clip(rng.normal(0.82, 0.08), 0, 1)), 3),
                                    "bbox": [int(cx - 0.08 * h), int(cy - 0.04 * h), int(cx + 0.08 * h), int(cy + 0.04 * h)]})
                pid = a.actor + 1
                recs.append({
                    "frame_idx": i, "timestamp": ts, "camera_id": "video", "person_id": pid, "track_id": pid,
                    "bbox": [int(v) for v in box], "reid_score": 1.0, "det_conf": 0.9,
                    "frame_size": [sim.W, sim.H], "has_pose": True, "landmarks": None,
                    "landmarks_px": np.round(pts, 2).tolist(), "visibility": [1.0] * 33,
                })
            for r in recs:
                r["weapons"] = weapons
                f.write(json.dumps(r) + "\n")
                n += 1
    meta = {"source": "video", "dataset": "synthetic", "video": f"sim://{scene}#{k}", "label": SCENE_LABELS[scene],
            "split": None, "fps": fps, "frame_size": [sim.W, sim.H], "frames": n_frames, "records": n,
            "augment": {"speed": speed, "mirror": mirror, "shift": shift, "dv": dv, "noise": noise}}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(config.DATA_DIR / "synthetic" / "sessions"))
    ap.add_argument("--variants", type=int, default=30, help="每個情境產生幾個隨機變體")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.output)
    if args.clean:
        shutil.rmtree(out_root, ignore_errors=True)
    rng = np.random.default_rng(args.seed)
    total = 0
    for scene in SCENE_LABELS:
        for k in range(args.variants):
            total += make_session(scene, k, rng, out_root / f"synthetic_{scene}_{k:03d}", args.fps)
    print(f"完成：{len(SCENE_LABELS) * args.variants} 個合成 session、{total} 筆紀錄 → {out_root}")


if __name__ == "__main__":
    main()
