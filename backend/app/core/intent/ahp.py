"""AHP 層級分析法（Saaty, 2013）：由成對比較矩陣求九項指數權重與一致性比率。

研究計畫公式 (4)：R = Σ w_i · X_i，其中 w_i 為 AHP 權重。
成對比較採 Saaty 1–9 尺度；下方矩陣即專家判斷，可依標註結果調整後重新計算。
計畫書敘述「X₁ 權重最高約 0.25、X₄ 權重較低約 0.05」，本矩陣計算結果約為
X1≈0.27、X2≈0.15、X6≈0.14、X3≈0.09、X5≈0.085、X8≈0.085、X7≈0.07、X4≈0.05、X9≈0.05。
"""
from fractions import Fraction
from typing import Dict, List, Tuple

import numpy as np

from app.core.intent.constants import INDEX_KEYS

_F = Fraction

# 列 i、欄 j：X_i 相對 X_j 的重要程度
PAIRWISE_MATRIX: List[List[Fraction]] = [
    #  X1      X2      X3      X4   X5      X6      X7   X8      X9
    [_F(1),   _F(2),  _F(3),  _F(5), _F(3),  _F(2),  _F(4), _F(3),  _F(5)],   # X1 武器準備
    [_F(1, 2), _F(1), _F(2),  _F(3), _F(2),  _F(1),  _F(2), _F(2),  _F(3)],   # X2 拳擊姿態
    [_F(1, 3), _F(1, 2), _F(1), _F(2), _F(1), _F(1), _F(1), _F(1),  _F(2)],   # X3 追逐衝刺
    [_F(1, 5), _F(1, 3), _F(1, 2), _F(1), _F(1, 2), _F(1, 3), _F(1), _F(1, 2), _F(1)],  # X4 徘徊
    [_F(1, 3), _F(1, 2), _F(1), _F(2), _F(1), _F(1, 2), _F(1), _F(1),  _F(2)],  # X5 尾隨
    [_F(1, 2), _F(1),  _F(1), _F(3), _F(2),  _F(1),  _F(2), _F(2),  _F(3)],   # X6 距離侵略
    [_F(1, 4), _F(1, 2), _F(1), _F(1), _F(1), _F(1, 2), _F(1), _F(1), _F(1)],  # X7 群聚對峙
    [_F(1, 3), _F(1, 2), _F(1), _F(2), _F(1), _F(1, 2), _F(1), _F(1),  _F(2)],  # X8 歷史風險
    [_F(1, 5), _F(1, 3), _F(1, 2), _F(1), _F(1, 2), _F(1, 3), _F(1), _F(1, 2), _F(1)],  # X9 空間脈絡
]

# Saaty 隨機一致性指標
RANDOM_INDEX = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


def _check_reciprocal(matrix: np.ndarray) -> None:
    n = matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if abs(matrix[i, j] * matrix[j, i] - 1.0) > 1e-9:
                raise ValueError(f"AHP 成對比較矩陣不互為倒數: a[{i}][{j}]={matrix[i, j]}, a[{j}][{i}]={matrix[j, i]}")


def compute_ahp(matrix: List[List[Fraction]]) -> Tuple[np.ndarray, float, float, float]:
    """回傳 (weights, lambda_max, CI, CR)。以特徵向量法（power iteration）求主特徵向量。"""
    a = np.array([[float(v) for v in row] for row in matrix], dtype=np.float64)
    if a.shape[0] != a.shape[1]:
        raise ValueError("AHP 矩陣必須為方陣")
    _check_reciprocal(a)

    n = a.shape[0]
    w = np.ones(n) / n
    for _ in range(200):
        nw = a @ w
        nw = nw / nw.sum()
        if np.abs(nw - w).max() < 1e-12:
            w = nw
            break
        w = nw

    lambda_max = float(np.mean((a @ w) / w))
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri = RANDOM_INDEX.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0.0
    return w, lambda_max, ci, cr


_WEIGHTS, LAMBDA_MAX, CONSISTENCY_INDEX, CONSISTENCY_RATIO = compute_ahp(PAIRWISE_MATRIX)

AHP_WEIGHTS: Dict[str, float] = {k: float(v) for k, v in zip(INDEX_KEYS, _WEIGHTS)}
AHP_WEIGHT_VECTOR = np.array([AHP_WEIGHTS[k] for k in INDEX_KEYS], dtype=np.float32)

if CONSISTENCY_RATIO >= 0.1:
    print(f"[AHP] ⚠️ 一致性比率 CR={CONSISTENCY_RATIO:.3f} ≥ 0.1，請重新檢視成對比較矩陣")


def weighted_sum(x: np.ndarray) -> float:
    """Σ w_i X_i（0~1）。x 依 INDEX_KEYS 排序。"""
    return float(np.dot(AHP_WEIGHT_VECTOR, np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)))


def ahp_summary() -> Dict:
    return {
        "weights": AHP_WEIGHTS,
        "lambda_max": LAMBDA_MAX,
        "consistency_index": CONSISTENCY_INDEX,
        "consistency_ratio": CONSISTENCY_RATIO,
        "matrix": [[str(v) for v in row] for row in PAIRWISE_MATRIX],
    }
