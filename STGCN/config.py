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

from app.core.intent.constants import ACTION_CLASSES, ATTACK_ORDER, ATTACK_TAXONOMY, LEVELS, NUM_INDICES  # noqa: E402,F401

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

# ---------------------------------------------------------------------------
# 資料集標籤：路徑 → 原始攻擊標籤（normal / fight / punch / kick / push / strike_object / knife / gun / weapon）
# preprocess.py 再依 --taxonomy coarse|fine 合併成實際訓練類別（見 ATTACK_TAXONOMY）。
# ---------------------------------------------------------------------------

# 資料夾名稱（不分大小寫，去掉空白、底線、連字號後比對）
FOLDER_LABELS = {
    # RWF-2000：train|val / Fight|NonFight
    "fight": "fight", "fights": "fight", "nonfight": "normal", "nonfights": "normal", "nofight": "normal",
    "nofights": "normal",
    # Hockey Fights / Movies Fights 合併在 RWF-2000 資料夾中的情況
    "hockeyfight": "fight", "hockeyfights": "fight", "moviefight": "fight", "moviefights": "fight",
    "hockeynonfight": "normal", "movienonfight": "normal", "movienonfights": "normal",
    # Real Life Violence Situations：Violence / NonViolence
    "violence": "fight", "violent": "fight", "nonviolence": "normal", "nonviolent": "normal",
    # SCVD（Smart-City CCTV Violence Detection）：Normal / Violence / Weaponized
    "normal": "normal", "weaponized": "weapon", "weapon": "weapon", "weapons": "weapon",
    "weaponviolence": "weapon",
    # UCF-Crime 等
    "fighting": "fight", "assault": "fight", "abuse": "fight", "robbery": "weapon", "shooting": "gun",
    "normalvideos": "normal", "normalvideosfortraining": "normal", "testingnormalvideosanomaly": "normal",
    # 自錄資料可直接用類別名稱當資料夾
    "punch": "punch", "kick": "kick", "push": "push", "knife": "knife", "gun": "gun",
    "strikeobject": "strike_object",
}

# NTU RGB+D 120 檔名 SsssCcccPpppRrrrAaaa_rgb.avi：雙人互動動作代碼 → 原始標籤
# （其餘非攻擊互動如握手、擁抱、遞物、走近/走開等 → normal；單人動作不使用）
NTU_ACTIONS = {
    50: "punch", 51: "kick", 52: "push", 106: "strike_object", 107: "knife", 108: "push", 110: "gun",
    53: "normal", 54: "normal", 55: "normal", 56: "normal", 57: "normal", 58: "normal", 59: "normal",
    60: "normal", 109: "normal", 111: "normal", 112: "normal", 113: "normal", 114: "normal", 115: "normal",
    116: "normal", 117: "normal", 118: "normal", 119: "normal", 120: "normal",
}

# 檔名前綴（Hockey Fights：fi123_xvid.avi / no123_xvid.avi；RLVS：V_1.mp4 / NV_1.mp4）
FILENAME_PREFIX_LABELS = [("nv_", "normal"), ("no", "normal"), ("fi", "fight"), ("v_", "fight")]


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def resolve_raw_label(rel_path, forced=None, use_prefix=True):
    """由相對路徑推得原始攻擊標籤；回傳 (label, source) 或 (None, None)。"""
    import re
    from pathlib import Path
    if forced:
        return forced, "forced"
    path = Path(rel_path)
    m = re.search(r"A(\d{3})", path.stem)
    if m and re.search(r"S\d{3}C\d{3}P\d{3}R\d{3}", path.stem):
        code = int(m.group(1))
        return (NTU_ACTIONS.get(code), "ntu") if code in NTU_ACTIONS else (None, None)
    for part in reversed(path.parts[:-1]):
        lab = FOLDER_LABELS.get(_norm(part))
        if lab:
            return lab, "folder"
    if use_prefix:
        stem = path.stem.lower()
        for prefix, lab in FILENAME_PREFIX_LABELS:
            if stem.startswith(prefix) and stem[len(prefix):len(prefix) + 1].isdigit():
                return lab, "filename"
    return None, None


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
