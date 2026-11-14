"""集中管理專案路徑，避免各模組各自用 parents[n] 推算。"""
import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]          # backend/
REPO_DIR = BACKEND_DIR.parent                               # attack-prediction/

# 讓 os.getenv 也讀得到 backend/.env（MILVUS_URI、NOTIFY_WEBHOOK_URL、INTENT_MODEL_PATH…）
try:
    from dotenv import load_dotenv

    load_dotenv(BACKEND_DIR / ".env", override=False)
except ImportError:
    pass
STGCN_DIR = Path(os.getenv("STGCN_DIR", str(REPO_DIR / "STGCN")))

WEIGHTS_DIR = BACKEND_DIR / "weights"
DATA_DIR = BACKEND_DIR / "data"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
EVENTS_DIR = DATA_DIR / "events"
RAG_DIR = DATA_DIR / "rag"
CONFIG_DIR = BACKEND_DIR / "app" / "config"

DEFAULT_INTENT_MODEL = STGCN_DIR / "checkpoints" / "intent_model.pt"

for _d in (DATA_DIR, SNAPSHOT_DIR, EVENTS_DIR, RAG_DIR):
    _d.mkdir(parents=True, exist_ok=True)
