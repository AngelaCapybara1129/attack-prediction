"""STGCN 訓練專案設定。

模型與特徵程式碼統一放在 backend/app/core/intent/（即時推論與訓練共用同一份），
這裡把 backend 加入 sys.path，讓訓練腳本可以 `from app.core.intent... import ...`。
"""
import os
import sys
from pathlib import Path

STGCN_DIR = Path(__file__).resolve().parent
REPO_DIR = STGCN_DIR.parent
BACKEND_DIR = Path(os.getenv("BACKEND_DIR", str(REPO_DIR / "backend")))

for _p in (str(STGCN_DIR), str(BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.core.intent.constants import ACTION_CLASSES, LEVELS, NUM_INDICES  # noqa: E402

DATA_DIR = STGCN_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_SESSIONS_DIR = DATA_DIR / "interim" / "sessions"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = STGCN_DIR / "checkpoints"
LOG_DIR = STGCN_DIR / "logs"

# 後端錄製的 session（DataRecorder 輸出）
BACKEND_SESSION_DIRS = [BACKEND_DIR / "data" / "sessions", BACKEND_DIR / "app" / "data" / "sessions"]

# 滑動視窗（研究計畫：5–10 秒、約 150–300 幀）
SEQ_LEN = 150            # 重取樣後的固定幀數（5 秒 × 30 fps）
WINDOW_SEC = 5.0
HORIZON_SEC = 1.0        # 預測「下一秒」的指數與風險
STRIDE_SEC = 1.0
MIN_VALID_RATIO = 0.5    # 視窗內至少一半時間有骨架

DEFAULT_FRAME_SIZE = (640, 480)

# 資料夾名稱 → 行為類別（RWF-2000: Fight / NonFight；UCF-Crime 可自行擴充）
LABEL_ALIASES = {
    "fight": "dangerous",
    "violence": "dangerous",
    "violent": "dangerous",
    "assault": "dangerous",
    "fighting": "dangerous",
    "abuse": "dangerous",
    "robbery": "dangerous",
    "nonfight": "normal",
    "non-fight": "normal",
    "non_fight": "normal",
    "nonviolence": "normal",
    "normal": "normal",
    "suspicious": "suspicious",
    "dangerous": "dangerous",
}

MODEL_CONFIG = {
    "action_classes": ACTION_CLASSES,
    "num_index": NUM_INDICES,
    "num_level": len(LEVELS),
    "in_channels": 3,
    "num_point": 33,
    "num_person": 1,
    "x_embed": 32,
    "lstm_hidden": 128,
    "lstm_layers": 2,
    "dropout": 0.2,
    "seq_len": SEQ_LEN,
    "window_sec": WINDOW_SEC,
    "horizon_sec": HORIZON_SEC,
}

for _d in (INTERIM_SESSIONS_DIR, PROCESSED_DIR, CHECKPOINT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)
