"""測試集評估（研究計畫：攻擊意圖預測模型之成效評估）。

- 行為類型：Top-1 Accuracy、Macro-F1、Recall
- 意圖預測：下一秒 X1–X9 與風險分數的 MAE、RMSE
- 與「持續性基線」比較（假設下一秒 = 現在，即純規則 + AHP 系統的預測能力）

    python scripts/evaluate.py --split test
"""
import argparse
import json
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from app.core.intent.constants import ACTION_CLASSES, INDEX_KEYS, LEVELS  # noqa: E402
from app.core.intent.nets import build_model  # noqa: E402
from datasets.skeleton_dataset import IntentWindowDataset  # noqa: E402
from utils.metrics import classification_metrics, per_index_regression, regression_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--data", default=str(config.PROCESSED_DIR))
    ap.add_argument("--model", default=str(config.CHECKPOINT_DIR / "intent_model.pt"))
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model, map_location=device, weights_only=True)
    model = build_model(ckpt.get("config", {})).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = IntentWindowDataset(os.path.join(args.data, args.split))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    keys = ["action", "level", "next_x", "risk", "cur_x", "cur_risk", "cur_level"]
    ys = {k: [] for k in keys}
    preds = {"action": [], "level": [], "next_x": [], "risk": []}
    with torch.no_grad():
        for b in loader:
            out = model(b["skel"].to(device), b["xseq"].to(device))
            preds["action"].append(out["action_logits"].argmax(1).cpu().numpy())
            preds["level"].append(out["level_logits"].argmax(1).cpu().numpy())
            preds["next_x"].append(out["next_x"].cpu().numpy())
            preds["risk"].append(out["risk"].cpu().numpy())
            for k in keys:
                ys[k].append(b[k].numpy())
    y = {k: np.concatenate(v) for k, v in ys.items()}
    p = {k: np.concatenate(v) for k, v in preds.items()}

    report = {
        "split": args.split,
        "n": int(len(ds)),
        "model": args.model,
        "action": classification_metrics(y["action"], p["action"], len(ACTION_CLASSES), ACTION_CLASSES),
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

    print(f"樣本數: {report['n']}")
    print(f"[行為類型] Top-1={report['action']['top1']:.4f}  Macro-F1={report['action']['macro_f1']:.4f}  "
          f"Recall={report['action']['macro_recall']:.4f}")
    print(f"[風險等級] Acc={report['level']['top1']:.4f}  Macro-F1={report['level']['macro_f1']:.4f}  "
          f"(基線 Acc={report['baseline_persistence']['level']['top1']:.4f})")
    print(f"[下一秒 X1–X9] MAE={report['next_x']['mae']:.4f}  RMSE={report['next_x']['rmse']:.4f}  "
          f"(基線 MAE={report['baseline_persistence']['next_x']['mae']:.4f})")
    print(f"[風險分數] MAE={report['risk']['mae']:.4f}  RMSE={report['risk']['rmse']:.4f}  "
          f"(基線 MAE={report['baseline_persistence']['risk']['mae']:.4f})")

    out = config.LOG_DIR / f"eval_{args.split}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完整報告 → {out}")


if __name__ == "__main__":
    main()
