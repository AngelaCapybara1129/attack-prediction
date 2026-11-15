"""影片資料集（RWF-2000 / Hockey / RLVS / SCVD / NTU RGB+D 120 / 自錄影片）→ 骨架序列 session。

對每支影片執行 YOLO26 + ByteTrack + MediaPipe Pose（+ 武器偵測），
輸出格式與後端 DataRecorder 相同（frames.jsonl + meta.json），之後統一交給 preprocess.py 建立樣本。

資料夾結構範例（類別與 split 由路徑推得）：
    data/raw/rwf2000/train/Fight/*.avi
    data/raw/rwf2000/train/NonFight/*.avi
    data/raw/rwf2000/val/Fight/*.avi
標籤由路徑推得（見 config.resolve_raw_label）：資料夾 Fight/NonFight/Violence/Weaponized…、
Hockey 檔名 fi*/no*、NTU 檔名動作代碼 A050 揮拳 / A051 踢 / A052 推 / A107 持刀 / A110 持槍…
用法：
    python scripts/extract_skeletons.py --input data/raw/rwf2000 --dataset rwf2000
    python scripts/extract_skeletons.py --input data/raw/clips --label punch --max-videos 50
    # 平行處理（開 3 個終端機各跑一份）
    python scripts/extract_skeletons.py --input data/raw/rwf2000 --shard 0/3
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.core.cv.lap_compat import install as install_lap_compat  # noqa: E402
from app.core.intent.keypoints import pose_from_result  # noqa: E402
from app.core.intent.skeleton import crop_landmarks_to_frame, expand_box  # noqa: E402
from app.core.paths import WEIGHTS_DIR  # noqa: E402
from app.core.utils.pose_utils import (  # noqa: E402
    POSE_MIN_CONFIDENCE, POSE_UPSCALE_HEIGHT, extract_mediapipe_pose_full, pose_model_exists,
)

VIDEO_EXTS = {".avi", ".mp4", ".mkv", ".mov"}

install_lap_compat()   # ByteTrack 的 lap 原生套件被封鎖時改用 scipy


def infer_label_and_split(path: Path, forced_label=None):
    parts = [p.lower() for p in path.parts]
    label, _ = config.resolve_raw_label(path, forced_label)
    split = None
    for p in parts:
        if p in ("train", "training"):
            split = "train"
        elif p in ("val", "valid", "validation"):
            split = "val"
        elif p in ("test", "testing"):
            split = "test"
    return label, split


def safe_name(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z_\-]+", "_", s)[:120]


def process_video(video: Path, out_dir: Path, label, split, args, weapon_detector, dataset=None) -> int:
    from ultralytics import YOLO

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"  ⚠️ 無法開啟 {video}")
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not np.isfinite(fps) or fps <= 1:
        fps = 30.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or config.DEFAULT_FRAME_SIZE[0]
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or config.DEFAULT_FRAME_SIZE[1]

    weights = args.yolo or ("yolo26n-pose.pt" if args.pose == "yolo" else "yolo26l.pt")
    model = YOLO(str(WEIGHTS_DIR / weights))   # 每支影片新的 tracker 狀態
    tracker_cfg = config.BACKEND_DIR / "app" / "scripts" / "custom_tracker.yaml"
    tracker_arg = str(tracker_cfg) if tracker_cfg.exists() else "bytetrack.yaml"

    out_dir.mkdir(parents=True, exist_ok=True)
    n_records = 0
    frame_idx = 0
    with open(out_dir / "frames.jsonl", "w", encoding="utf-8") as f:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % args.every != 0:
                frame_idx += 1
                continue
            ts = frame_idx / fps
            res = model.track(frame, classes=[0], conf=args.conf, imgsz=640, persist=True,
                              tracker=tracker_arg, verbose=False)[0]
            weapons = []
            if weapon_detector is not None:
                weapons = [{"label": w.label, "conf": round(w.conf, 3), "bbox": list(w.bbox)}
                           for w in weapon_detector.detect(video.name, frame)]
            if res.boxes is not None and res.boxes.id is not None:
                for k, (box, tid, cf) in enumerate(zip(res.boxes.xyxy.int().cpu().tolist(),
                                                       res.boxes.id.int().cpu().tolist(),
                                                       res.boxes.conf.cpu().tolist())):
                    pose_box = expand_box(box, fw, fh)
                    landmarks, vis, lm_px = None, None, None
                    if args.pose == "yolo":
                        lm33, vis33 = pose_from_result(res, k)
                        if lm33 is not None:
                            lm_px = lm33.round(2).tolist()
                            vis = vis33.round(3).tolist()
                    else:
                        x1, y1, x2, y2 = pose_box
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0 and crop.shape[0] >= args.min_pose_h and crop.shape[1] >= args.min_pose_h // 2:
                            landmarks, vis = extract_mediapipe_pose_full(crop)
                        lm_px = crop_landmarks_to_frame(landmarks, pose_box).round(2).tolist() if landmarks else None
                    rec = {
                        "frame_idx": frame_idx, "timestamp": ts, "camera_id": "video",
                        "person_id": int(tid), "track_id": int(tid), "bbox": box,
                        "reid_score": 1.0, "det_conf": float(cf), "frame_size": [fw, fh], "pose_box": pose_box,
                        "has_pose": lm_px is not None, "landmarks": landmarks, "landmarks_px": lm_px,
                        "visibility": vis, "weapons": weapons,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_records += 1
            frame_idx += 1
    cap.release()

    meta = {
        "source": "video", "video": str(video), "dataset": dataset, "label": label, "split": split,
        "fps": fps, "frame_size": [fw, fh], "frames": frame_idx, "records": n_records, "detector": weights, "pose_backend": args.pose,
        "every": args.every, "min_pose_h": args.min_pose_h,
        "pose": {"min_confidence": POSE_MIN_CONFIDENCE, "upscale_height": POSE_UPSCALE_HEIGHT},
        "created_at": time.time(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return n_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=str(config.RAW_DIR), help="影片根目錄（遞迴搜尋）")
    ap.add_argument("--output", type=str, default=str(config.INTERIM_SESSIONS_DIR))
    ap.add_argument("--label", type=str, default=None,
                    help="強制指定原始標籤（normal/fight/punch/kick/push/strike_object/knife/gun/weapon）")
    ap.add_argument("--dataset", type=str, default=None, help="資料集名稱（寫進 meta.json，例如 rwf2000）")
    ap.add_argument("--shard", type=str, default="0/1", help="i/n：只處理第 i 份（多個程序平行抽取）")
    ap.add_argument("--keep-unlabeled", action="store_true", help="無法判斷標籤的影片也抽取（預設略過）")
    ap.add_argument("--pose", choices=["yolo", "mediapipe"], default="yolo",
                    help="骨架來源：yolo＝YOLO26-pose 多人關鍵點（預設，與即時系統相同）；mediapipe＝裁切後 MediaPipe 33 點")
    ap.add_argument("--yolo", type=str, default=None,
                    help="YOLO 權重（預設 --pose yolo 用 yolo26n-pose.pt，mediapipe 用 yolo26l.pt；不存在時自動下載官方權重）")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--every", type=int, default=1, help="每 N 幀處理一次")
    ap.add_argument("--min-pose-h", type=int, default=40, help="人物裁切最小高度（小裁切會先放大到 256 px 再偵測）")
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--no-weapons", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--redo-before", type=str, default=None,
                    help="只重做在此時間之前產生的 session（例如 '2026-11-15 01:50'），之後產生的略過；用於改設定後續跑")
    args = ap.parse_args()
    redo_before = None
    if args.redo_before:
        import datetime as _dt
        redo_before = _dt.datetime.fromisoformat(args.redo_before).timestamp()

    if not pose_model_exists():
        print("❌ 找不到 MediaPipe pose_landmarker 模型（backend/weights/pose_landmarker_lite.task）")
        return

    root = Path(args.input)
    videos = sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    if args.max_videos:
        videos = videos[: args.max_videos]
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    videos = videos[shard_i::shard_n]
    print(f"找到 {len(videos)} 支影片於 {root.resolve()}（shard {shard_i}/{shard_n}）")

    weapon_detector = None
    if not args.no_weapons:
        try:
            from app.core.cv.weapons import WeaponDetector
            weapon_detector = WeaponDetector()
        except Exception as e:
            print(f"⚠️ 武器偵測器無法使用（X1 將為 0）: {e}")

    out_root = Path(args.output)
    total, skipped = 0, 0
    for i, video in enumerate(videos, 1):
        rel = video.relative_to(root)
        label, split = infer_label_and_split(rel, args.label)
        if label is None and not args.keep_unlabeled:
            skipped += 1
            continue
        prefix = f"{args.dataset}_" if args.dataset else ""
        # 名稱含上層資料夾，避免不同資料夾內同名影片互相覆蓋
        name = safe_name(f"{prefix}{split or 'all'}_{label or 'unlabeled'}_{'_'.join(rel.parts[-2:-1])}_{video.stem}")
        out_dir = out_root / name
        meta_path = out_dir / "meta.json"
        if meta_path.exists() and not args.overwrite:
            stale = False
            if redo_before is not None:
                try:
                    stale = float(json.loads(meta_path.read_text(encoding="utf-8")).get("created_at", 0)) < redo_before
                except Exception:
                    stale = True
            if not stale:
                print(f"[{i}/{len(videos)}] 已存在，略過 {name}")
                continue
        t0 = time.time()
        n = process_video(video, out_dir, label, split, args, weapon_detector, args.dataset)
        total += n
        print(f"[{i}/{len(videos)}] {video.name} → {name}  records={n}  ({time.time() - t0:.1f}s)")
    print(f"完成，共 {total} 筆人物-幀紀錄 → {out_root}（無標籤略過 {skipped} 支）")


if __name__ == "__main__":
    main()
