"""執行中任務的 IntentEngine 登記表，供 /api/intent/live 查詢。"""
import threading
from typing import Dict, List, Optional

from app.core.intent.engine import IntentEngine

_engines: Dict[str, IntentEngine] = {}
_lock = threading.Lock()


def register(engine: IntentEngine) -> None:
    with _lock:
        # 只保留最近的幾個任務，避免停止後的引擎一直佔用記憶體
        finished = [tid for tid, e in _engines.items() if not e.running]
        for tid in finished[:-2]:
            _engines.pop(tid, None)
        _engines[engine.task_id] = engine


def get(task_id: str) -> Optional[IntentEngine]:
    with _lock:
        return _engines.get(task_id)


def mark_stopped(task_id: str) -> None:
    with _lock:
        e = _engines.get(task_id)
        if e is not None:
            e.running = False


def list_tasks() -> List[Dict]:
    with _lock:
        return [
            {"task_id": tid, "running": e.running, "started_at": e.started_at,
             "frames_processed": e.frames_processed, "alerts_emitted": e.alerts_emitted}
            for tid, e in _engines.items()
        ]
