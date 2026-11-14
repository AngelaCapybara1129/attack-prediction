"""評估指標（研究計畫：行為類型 Top-1 Accuracy / Macro-F1 / Recall；意圖預測 MAE / RMSE）。純 numpy 實作。"""
from typing import Dict, List, Optional, Sequence

import numpy as np


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int,
                           class_names: Optional[List[str]] = None) -> Dict:
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(yt, yp):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    present = support > 0
    names = class_names or [str(i) for i in range(num_classes)]
    return {
        "top1": float(tp.sum() / max(1, len(yt))),
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "macro_recall": float(recall[present].mean()) if present.any() else 0.0,
        "per_class": {
            names[i]: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]),
                       "support": int(support[i])}
            for i in range(num_classes)
        },
        "confusion_matrix": cm.tolist(),
        "n": int(len(yt)),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.size == 0:
        return {"mae": 0.0, "rmse": 0.0}
    err = yp - yt
    return {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean()))}


def per_index_regression(y_true: np.ndarray, y_pred: np.ndarray, keys: List[str]) -> Dict:
    return {k: regression_metrics(y_true[:, i], y_pred[:, i]) for i, k in enumerate(keys)}
