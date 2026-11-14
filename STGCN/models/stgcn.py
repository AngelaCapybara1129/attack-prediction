"""相容舊路徑：模型實作已移至 backend/app/core/intent/nets.py（訓練與即時推論共用）。

舊版 SimpleSTGCN 只有 1×1 卷積、沒有使用骨架鄰接矩陣，並非真正的圖卷積；
現在的 STGCN 依 Yan et al. (2018) 使用 MediaPipe 33 點的空間構型分群鄰接矩陣。
"""
import config  # noqa: F401  設定 sys.path

from app.core.intent.nets import DualChannelIntentNet, STGCN, STGCNBackbone, build_model  # noqa: E402,F401

SimpleSTGCN = STGCN
