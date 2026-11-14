"""ByteTrack（ultralytics）需要的 `lap.lapjv`：原生套件無法載入時，改用 scipy 的純 Python 等效實作。

本機 Windows 的應用程式控制原則會封鎖 lap 套件的 DLL（_lapjv.pyd），ultralytics 匯入 tracker 時就會失敗，
並嘗試自動 pip 安裝（仍然被封鎖）。這裡在匯入 ultralytics 之前先檢查：
  - 原生 lap 可用 → 什麼都不做（Docker / Linux 走這條）
  - 不可用 → 在 sys.modules 註冊同介面的替代模組，被封鎖的 DLL 不會被執行

lapjv(cost, extend_cost=True, cost_limit=c) 的語意（與 gatagat/lap 相同）：
成本大於 cost_limit 的配對視為不配對；回傳 (總成本, x, y)，x[i] = 第 i 列配到的欄（-1 表示未配對），y 反之。
做法同 lap：把矩陣擴充成 (n+m)×(n+m)，以 cost_limit/2 作為「不配對」的成本。
"""
import sys
import types

import numpy as np

_INSTALLED = False


def lapjv(cost, extend_cost=False, cost_limit=np.inf, return_cost=True):
    from scipy.optimize import linear_sum_assignment

    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    if n == 0 or m == 0:
        x, y = np.full(n, -1, dtype=np.int64), np.full(m, -1, dtype=np.int64)
        return (0.0, x, y) if return_cost else (x, y)
    if not extend_cost and n != m and not np.isfinite(cost_limit):
        raise ValueError("Square cost array expected. Pass extend_cost=True if non-square cost is intended.")

    if np.isfinite(cost_limit):
        big = max(1e6, float(np.abs(cost[np.isfinite(cost)]).max(initial=0.0)) * 10 + cost_limit)
        ext = np.full((n + m, n + m), big, dtype=np.float64)
        ext[:n, :m] = np.where(np.isfinite(cost), cost, big)
        ext[np.arange(n), m + np.arange(n)] = cost_limit / 2.0       # 列不配對
        ext[n + np.arange(m), np.arange(m)] = cost_limit / 2.0       # 欄不配對
        ext[n:, m:] = 0.0
        rows, cols = linear_sum_assignment(ext)
        x = np.full(n, -1, dtype=np.int64)
        y = np.full(m, -1, dtype=np.int64)
        for r, c in zip(rows, cols):
            if r < n and c < m and cost[r, c] <= cost_limit:
                x[r], y[c] = c, r
    else:
        rows, cols = linear_sum_assignment(np.where(np.isfinite(cost), cost, 1e12))
        x = np.full(n, -1, dtype=np.int64)
        y = np.full(m, -1, dtype=np.int64)
        x[rows], y[cols] = cols, rows

    total = float(sum(cost[i, x[i]] for i in range(n) if x[i] >= 0))
    return (total, x, y) if return_cost else (x, y)


def install() -> str:
    """在匯入 ultralytics tracker 之前呼叫。回傳使用的實作："native" 或 "scipy"。"""
    global _INSTALLED
    if _INSTALLED:
        return "scipy" if getattr(sys.modules.get("lap"), "__lap_compat__", False) else "native"
    _INSTALLED = True
    try:
        import lap  # noqa: F401

        if getattr(lap, "__version__", None):
            return "native"
    except Exception:
        pass
    sys.modules.pop("lap", None)
    mod = types.ModuleType("lap")
    mod.__version__ = "0.0-scipy-compat"
    mod.__lap_compat__ = True
    mod.lapjv = lapjv
    sys.modules["lap"] = mod
    print("[TRACK] 原生 lap 無法載入（可能被應用程式控制原則封鎖），ByteTrack 改用 scipy 指派演算法")
    return "scipy"
