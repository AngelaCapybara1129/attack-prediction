"""測試集評估（研究計畫：攻擊意圖預測模型之成效評估）。

- 攻擊類型：Top-1 Accuracy、Macro-F1、Recall
    * 視窗層級：每個人物的每個時間視窗
    * 影片層級：同一支影片的所有視窗彙整（RWF-2000 等資料集的標準評估方式）
      任一視窗的攻擊類別機率 ≥ --video-threshold → 判為該攻擊類別（取最高者），否則為 normal
- 意圖預測：下一秒 X1–X9 與風險分數的 MAE、RMSE
- 與「持續性基線」比較（假設下一秒 = 現在，即純規則 + AHP 系統的預測能力）

    python scripts/evaluate.py --split test
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from app.core.intent.constants import ACTION_CLASSES, ACTION_NAMES, INDEX_KEYS, LEVELS, NON_ATTACK_CLASSES  # noqa: E402
from app.core.intent.nets import build_model  # noqa: E402
from datasets.skeleton_dataset import IntentWindowDataset  # noqa: E402
from utils.metrics import classification_metrics, per_index_regression, regression_metrics  # noqa: E402


def session_of(split_dir: str):
    out = {}
    with open(os.path.join(split_dir, "labels.csv"), "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["sample_id"].strip()] = row.get("session") or row["sample_id"]
    return out


def video_level(sessions, y_action, probs, classes, threshold):
    attack_idx = [i for i, c in enumerate(classes) if c not in NON_ATTACK_CLASSES]
    normal_idx = classes.index("normal") if "normal" in classes else 0
    by_video = defaultdict(list)
    for s, y, p in zip(sessions, y_action, probs):
        if y >= 0:
            by_video[s].append((y, p))
    yt, yp = [], []
    for items in by_video.values():
        truth = np.bincount([y for y, _ in items], minlength=len(classes)).argmax()
        P = np.stack([p for _, p in items])
        best = max(attack_idx, key=lambda i: P[:, i].max()) if attack_idx else normal_idx
        pred = best if attack_idx and P[:, best].max() >= threshold else normal_idx
        yt.append(int(truth))
        yp.append(int(pred))
    return classification_metrics(np.array(yt), np.array(yp), len(classes), classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--data", default=str(config.PROCESSED_DIR))
    ap.add_argument("--model", default=str(config.CHECKPOINT_DIR / "intent_model.pt"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--video-threshold", type=float, default=0.5)
    ap.add_argument("--out", default=None, help="報告輸出路徑（預設 logs/eval_<split>.json）")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model, map_location=device, weights_only=True)
    cfg = ckpt.get("config", {})
    classes = list(cfg.get("action_classes", ACTION_CLASSES))
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    split_dir = os.path.join(args.data, args.split)
    ds = IntentWindowDataset(split_dir)
    sess_map = session_of(split_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    keys = ["action", "level", "next_x", "risk", "cur_x", "cur_risk", "cur_level"]
    ys = {k: [] for k in keys}
    preds = {"action": [], "action_prob": [], "level": [], "next_x": [], "risk": []}
    with torch.no_grad():
        for b in loader:
            out = model(b["skel"].to(device), b["xseq"].to(device))
            prob = torch.softmax(out["action_logits"], 1).cpu().numpy()
            preds["action_prob"].append(prob)
            preds["action"].append(prob.argmax(1))
            preds["level"].append(out["level_logits"].argmax(1).cpu().numpy())
            preds["next_x"].append(out["next_x"].cpu().numpy())
            preds["risk"].append(out["risk"].cpu().numpy())
            for k in keys:
                ys[k].append(b[k].numpy())
    y = {k: np.concatenate(v) for k, v in ys.items()}
    p = {k: np.concatenate(v) for k, v in preds.items()}
    sessions = [sess_map.get(it["sample_id"], it["sample_id"]) for it in ds.items]
    labeled = y["action"] >= 0

    report = {
        "split": args.split,
        "n": int(len(ds)),
        "n_labeled": int(labeled.sum()),
        "model": args.model,
        "action_classes": classes,
        "action": classification_metrics(y["action"][labeled], p["action"][labeled], len(classes), classes),
        "action_video": video_level(sessions, y["action"], p["action_prob"], classes, args.video_threshold),
        "level": classification_metrics(y["level"], p["level"], len(LEVELS), LEVELS),
        "next_x": {**regression_metrics(y["next_x"], p["next_x"]),
                   "per_index": per_index_regression(y["next_x"], p["next_x"], INDEX_KEYS)},
        "risk": regression_metrics(y["risk"], p["risk"]),
        "baseline_persistence": {
            "next_x": regression_metrics(y["next_x"], y["cur_x"]),
            "risk": regression_metrics(y["risk"], y["cur_risk"]),
            "level": classification_metrics(y["level"], y["cur_level"], len(LEVELS), LEVELS),
        },
    }

    a, av = report["action"], report["action_video"]
    print(f"樣本數: {report['n']}（有攻擊類型標籤 {report['n_labeled']}）  類別: {classes}")
    print(f"[攻擊類型-視窗] Top-1={a['top1']:.4f}  Macro-F1={a['macro_f1']:.4f}  Recall={a['macro_recall']:.4f}")
    print(f"[攻擊類型-影片] Top-1={av['top1']:.4f}  Macro-F1={av['macro_f1']:.4f}  Recall={av['macro_recall']:.4f}"
          f"  (影片數 {av['n']})")
    for c in classes:
        pc = a["per_class"][c]
        print(f"    {c:<14}{ACTION_NAMES.get(c, ''):<6} P={pc['precision']:.3f} R={pc['recall']:.3f} "
              f"F1={pc['f1']:.3f} n={pc['support']}")
    print("    混淆矩陣（列=真值, 欄=預測）:")
    for c, row in zip(classes, a["confusion_matrix"]):
        print(f"      {c:<14}{row}")
    print(f"[風險等級] Acc={report['level']['top1']:.4f}  Macro-F1={report['level']['macro_f1']:.4f}  "
          f"(基線 Acc={report['baseline_persistence']['level']['top1']:.4f})")
    print(f"[下一秒 X1–X9] MAE={report['next_x']['mae']:.4f}  RMSE={report['next_x']['rmse']:.4f}  "
          f"(基線 MAE={report['baseline_persistence']['next_x']['mae']:.4f})")
    print(f"[風險分數] MAE={report['risk']['mae']:.4f}  RMSE={report['risk']['rmse']:.4f}  "
          f"(基線 MAE={report['baseline_persistence']['risk']['mae']:.4f})")

    out = args.out or str(config.LOG_DIR / f"eval_{args.split}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"完整報告 → {out}")


if __name__ == "__main__":
    main()
