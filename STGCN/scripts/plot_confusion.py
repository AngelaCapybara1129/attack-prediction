"""把 evaluate.py 輸出的報告畫成混淆矩陣圖（攻擊類型：視窗層級 + 影片層級；風險等級 L0–L4）。

    python scripts/evaluate.py --split test                    # 先產生 logs/eval_test.json
    python scripts/plot_confusion.py --report logs/eval_test.json

每格顯示「筆數」與「列百分比（該真值類別中被預測成此類的比例，即 recall 分布）」。
輸出：logs/confusion_<報告名稱>.png
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from app.core.intent.constants import ACTION_NAMES  # noqa: E402

CJK_FONTS = ["Microsoft JhengHei", "Noto Sans CJK TC", "Noto Sans TC", "PingFang TC", "Heiti TC", "SimHei"]


def use_cjk_font():
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in CJK_FONTS:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def draw(ax, cm, labels, title, metrics):
    cm = np.asarray(cm, dtype=np.float64)
    row = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    n = len(labels)
    ax.set_xticks(range(n), labels, rotation=30 if n > 3 else 0, ha="right" if n > 3 else "center")
    ax.set_yticks(range(n), labels)
    ax.set_xlabel("預測")
    ax.set_ylabel("真值")
    for i in range(n):
        for j in range(n):
            color = "white" if pct[i, j] > 0.55 else "#1e293b"
            ax.text(j, i, f"{int(cm[i, j])}\n{pct[i, j]:.0%}", ha="center", va="center", fontsize=10, color=color,
                    fontweight="bold" if i == j else "normal")
    ax.set_title(f"{title}\nTop-1 {metrics['top1']:.3f}  Macro-F1 {metrics['macro_f1']:.3f}  "
                 f"Recall {metrics['macro_recall']:.3f}  (n={metrics['n']})", fontsize=10)
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(config.LOG_DIR / "eval_test.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default=None, help="圖表總標題（例如：RWF-2000 測試集）")
    args = ap.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    classes = report.get("action_classes") or list(report["action"]["per_class"])
    names = [f"{ACTION_NAMES.get(c, c)}\n{c}" for c in classes]
    levels = list(report["level"]["per_class"])

    use_cjk_font()
    panels = [("攻擊類型（視窗層級）", report["action"], names)]
    if report.get("action_video", {}).get("n"):
        panels.append(("攻擊類型（影片層級）", report["action_video"], names))
    panels.append(("風險等級 L0–L4", report["level"], levels))

    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5.2))
    for ax, (title, m, labels) in zip(np.atleast_1d(axes), panels):
        draw(ax, m["confusion_matrix"], labels, title, m)
    fig.suptitle(args.title or f"混淆矩陣 — {Path(report.get('model', '')).name}（{report.get('split', '')}）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))   # 上方留給總標題，避免與子圖標題重疊
    out = args.out or str(config.LOG_DIR / f"confusion_{Path(args.report).stem}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"混淆矩陣 → {out}")

    for title, m, labels in panels:
        print(f"\n{title}（列=真值, 欄=預測）")
        flat = [lbl.replace("\n", " ") for lbl in labels]
        width = max(len(s) for s in flat) + 2
        print(" " * width + "".join(f"{s.split(' ')[-1]:>10}" for s in flat))
        for lbl, rowv in zip(flat, m["confusion_matrix"]):
            print(f"{lbl:<{width}}" + "".join(f"{v:>10}" for v in rowv))


if __name__ == "__main__":
    main()
