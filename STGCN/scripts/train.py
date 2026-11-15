"""訓練 ST-GCN + LSTM 雙通道攻擊意圖預測模型（多任務）。

    python scripts/train.py                      # 雙通道多任務（預設，輸出 checkpoints/intent_model.pt）
    python scripts/train.py --task action        # 只訓練 ST-GCN 行為分類（輸出 checkpoints/best_stgcn.pth）
    python scripts/train.py --task classify --data data/processed_clips
                                                 # 雙通道模型只訓練攻擊類型（整段影片 × 雙人，preprocess_clips.py）

損失 = CE(攻擊類型) + CE(L0–L4) + 5·MSE(下一秒 X1–X9) + 5·MSE(風險分數/10)
攻擊類型類別清單來自 preprocess 產生的 dataset_meta.json（--taxonomy coarse/fine），並存進 checkpoint；
action = -1（無標籤樣本）不計入攻擊類型損失與指標。
訓練完成後把 intent_model.pt 留在 STGCN/checkpoints/，後端啟動新任務時會自動載入
（或呼叫 POST /api/intent/model/reload）。
"""
import argparse
import csv
import json
import os
import sys
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from app.core.intent.constants import ACTION_CLASSES, INDEX_KEYS, LEVELS, RISK_SCALE  # noqa: E402,F401
from app.core.intent.nets import STGCN, build_model, count_parameters  # noqa: E402
from datasets.skeleton_dataset import IntentWindowDataset, class_weights  # noqa: E402
from utils.metrics import classification_metrics, regression_metrics  # noqa: E402


def to_py(obj):
    """checkpoint 以 weights_only=True 載入，metrics 只能含 Python 原生型別。"""
    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def masked_ce(logits, target, weight=None):
    """忽略 target = -1（無攻擊類型標籤）的樣本；全部無標籤時回傳 0。"""
    if not bool((target >= 0).any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits, target, weight=weight, ignore_index=-1)


def run_epoch(model, loader, device, task, classes, optimizer=None, w_action=None, w_level=None,
              label_smoothing=0.0):
    train = optimizer is not None
    model.train(train)
    totals = {"loss": 0.0, "n": 0}
    ys = {"action": [], "action_pred": [], "level": [], "level_pred": [], "next_x": [], "next_x_pred": [],
          "risk": [], "risk_pred": []}
    for batch in loader:
        skel = batch["skel"].to(device)
        y_action = batch["action"].to(device)
        with torch.set_grad_enabled(train):
            if task == "action":
                logits = model(skel)
                loss = masked_ce(logits, y_action, w_action)
                out = {"action_logits": logits}
            elif task == "classify":
                out = model(skel, batch["xseq"].to(device))
                loss = F.cross_entropy(out["action_logits"], y_action, weight=w_action, ignore_index=-1,
                                       label_smoothing=label_smoothing if train else 0.0)
            else:
                xseq = batch["xseq"].to(device)
                y_level = batch["level"].to(device)
                y_next = batch["next_x"].to(device)
                y_risk = batch["risk"].to(device)
                out = model(skel, xseq)
                loss = (
                    masked_ce(out["action_logits"], y_action, w_action)
                    + F.cross_entropy(out["level_logits"], y_level, weight=w_level)
                    + 5.0 * F.mse_loss(out["next_x"], y_next)
                    + 5.0 * F.mse_loss(out["risk"] / RISK_SCALE, y_risk / RISK_SCALE)
                )
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        bs = skel.size(0)
        totals["loss"] += float(loss.item()) * bs
        totals["n"] += bs
        ys["action"].append(batch["action"].numpy())
        ys["action_pred"].append(out["action_logits"].argmax(1).detach().cpu().numpy())
        if task == "intent":
            ys["level"].append(batch["level"].numpy())
            ys["level_pred"].append(out["level_logits"].argmax(1).detach().cpu().numpy())
            ys["next_x"].append(batch["next_x"].numpy())
            ys["next_x_pred"].append(out["next_x"].detach().cpu().numpy())
            ys["risk"].append(batch["risk"].numpy())
            ys["risk_pred"].append(out["risk"].detach().cpu().numpy())

    metrics = {"loss": totals["loss"] / max(1, totals["n"])}
    ya, pa = np.concatenate(ys["action"]), np.concatenate(ys["action_pred"])
    keep = ya >= 0
    metrics["action"] = classification_metrics(ya[keep], pa[keep], len(classes), classes)
    if task == "intent":
        metrics["level"] = classification_metrics(np.concatenate(ys["level"]), np.concatenate(ys["level_pred"]),
                                                  len(LEVELS), LEVELS)
        metrics["next_x"] = regression_metrics(np.concatenate(ys["next_x"]), np.concatenate(ys["next_x_pred"]))
        metrics["risk"] = regression_metrics(np.concatenate(ys["risk"]), np.concatenate(ys["risk_pred"]))
    return metrics


def selection_score(m, task):
    if task in ("action", "classify"):
        return m["action"]["macro_f1"]
    return m["action"]["macro_f1"] + m["level"]["macro_f1"] - m["next_x"]["mae"] - m["risk"]["mae"] / RISK_SCALE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["intent", "action", "classify"], default="intent",
                    help="intent：攻擊類型 + 下一秒 X/風險/等級；classify：雙通道只訓練攻擊類型；action：單通道 ST-GCN")
    ap.add_argument("--data", type=str, default=str(config.PROCESSED_DIR))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=12, help="驗證分數幾個 epoch 沒進步就提前停止")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dropout", type=float, default=None, help="覆寫模型 dropout（減少過擬合）")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = IntentWindowDataset(os.path.join(args.data, "train"), augment=True, seed=args.seed)
    val_ds = IntentWindowDataset(os.path.join(args.data, "val"))
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    model_cfg = dict(config.MODEL_CONFIG)
    meta_path = os.path.join(args.data, "dataset_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            model_cfg.update(json.load(f).get("config", {}))   # 與資料集一致的 seq_len / window_sec / horizon_sec

    if args.dropout is not None:
        model_cfg["dropout"] = args.dropout
    if args.task == "classify":
        model_cfg["task"] = "classify"
    print(f"模型設定：num_person={model_cfg.get('num_person', 1)} seq_len={model_cfg.get('seq_len')} "
          f"dropout={model_cfg.get('dropout')} task={args.task}")
    classes = list(model_cfg.get("action_classes") or ACTION_CLASSES)
    model_cfg["action_classes"] = classes
    a_counts = train_ds.label_counts("action", len(classes))
    l_counts = train_ds.label_counts("level", len(LEVELS))
    print("攻擊類型分布:", dict(zip(classes, a_counts.tolist())))
    print("風險等級分布:", dict(zip(LEVELS, l_counts.tolist())))
    w_action = class_weights(a_counts)
    w_level = class_weights(l_counts)
    w_action = None if w_action is None else w_action.to(device)
    w_level = None if w_level is None else w_level.to(device)

    if args.task == "action":
        model = STGCN(num_class=len(classes)).to(device)
        out_path = args.out or str(config.CHECKPOINT_DIR / "best_stgcn.pth")
    else:
        model = build_model(model_cfg).to(device)
        default_name = "attack_classifier.pt" if args.task == "classify" else "intent_model.pt"
        out_path = args.out or str(config.CHECKPOINT_DIR / default_name)
    print(f"Model params: {count_parameters(model):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=len(train_ds) > args.batch_size)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    log_path = config.LOG_DIR / f"train_{args.task}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    best, best_epoch, stale = -1e9, 0, 0
    with open(log_path, "w", newline="", encoding="utf-8") as lf:
        writer = csv.writer(lf)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_action_top1", "val_action_f1",
                         "val_level_f1", "val_next_x_mae", "val_risk_mae", "lr"])
        for epoch in range(1, args.epochs + 1):
            tr = run_epoch(model, train_loader, device, args.task, classes, optimizer, w_action, w_level,
                           label_smoothing=args.label_smoothing)
            va = run_epoch(model, val_loader, device, args.task, classes)
            scheduler.step()
            score = selection_score(va, args.task)
            lvl_f1 = va.get("level", {}).get("macro_f1", float("nan"))
            nx = va.get("next_x", {}).get("mae", float("nan"))
            rk = va.get("risk", {}).get("mae", float("nan"))
            print(f"Epoch [{epoch}/{args.epochs}] train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} | "
                  f"action top1={va['action']['top1']:.3f} F1={va['action']['macro_f1']:.3f} | "
                  f"level F1={lvl_f1:.3f} | nextX MAE={nx:.4f} | risk MAE={rk:.3f}")
            writer.writerow([epoch, tr["loss"], va["loss"], va["action"]["top1"], va["action"]["macro_f1"],
                             lvl_f1, nx, rk, optimizer.param_groups[0]["lr"]])
            lf.flush()

            if score > best:
                best, best_epoch, stale = score, epoch, 0
                if args.task == "action":
                    torch.save({"model_state": model.state_dict(), "action_classes": classes,
                                "config": to_py(model_cfg)}, out_path)
                else:
                    torch.save({
                        "model_state": model.state_dict(),
                        "config": to_py(model_cfg),
                        "metrics": to_py({"val": {k: v for k, v in va.items() if k != "loss"}, "epoch": epoch}),
                        "index_keys": INDEX_KEYS,
                    }, out_path)
                print(f"  ✅ best model saved → {out_path}")
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"驗證分數 {args.patience} 個 epoch 未進步，提前停止")
                    break

    print(f"Training finished. Best epoch {best_epoch}, score {best:.4f}. Log: {log_path}")


if __name__ == "__main__":
    main()
