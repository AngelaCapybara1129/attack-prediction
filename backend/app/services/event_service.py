"""告警事件處理：背景執行緒把告警寫入資料庫、保存關鍵畫面與事件前回放片段、
執行 RAG 相似事件檢索與說明產生、向量入庫，最後透過 WebSocket 廣播與推播。

影像分析主迴圈只呼叫 submit()（非阻塞），不會因 I/O 拖慢 FPS。
"""
import datetime as dt
import queue
import threading
import traceback
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from app.core.paths import BACKEND_DIR, EVENTS_DIR
from app.core.rag.appearance import describe_appearance
from app.core.rag.service import get_rag_service
from app.database.session import SessionLocal
from app.models.event import Event
from app.schemas.event import EventOut
from app.services.broker import broker
from app.services.notifier import notify_async


def _aware(d: Optional[dt.datetime]) -> Optional[dt.datetime]:
    if d is None:
        return None
    return d if d.tzinfo is not None else d.replace(tzinfo=dt.timezone.utc)


def event_to_dict(ev: Event) -> Dict[str, Any]:
    out = EventOut.model_validate(ev).model_copy(update={
        "has_snapshot": bool(ev.snapshot_path),
        "occurred_at": _aware(ev.occurred_at),
        "created_at": _aware(ev.created_at),
        "handled_at": _aware(ev.handled_at),
    })
    return out.model_dump(mode="json")


def _rel(path) -> str:
    try:
        return str(path.relative_to(BACKEND_DIR)).replace("\\", "/")
    except ValueError:
        return str(path)


class EventService:
    def __init__(self, maxsize: int = 256):
        self._queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize=maxsize)
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._worker, name="event-service", daemon=True)
        self._thread.start()
        try:
            get_rag_service()   # 預先建立向量庫與示範情境，避免第一次告警延遲
        except Exception as e:
            print(f"[EVENT] ⚠️ RAG 初始化失敗: {e}")

    def submit(self, alert: Dict[str, Any], frame_raw: Optional[np.ndarray], frame_vis: Optional[np.ndarray],
               clip: Optional[List[bytes]] = None) -> None:
        item = {
            "alert": alert,
            "frame_raw": None if frame_raw is None else frame_raw.copy(),
            "frame_vis": None if frame_vis is None else frame_vis.copy(),
            "clip": list(clip or []),
        }
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            print(f"[EVENT] ⚠️ 事件佇列已滿，丟棄告警 {alert.get('alert_id')}")

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            try:
                self._handle(item)
            except Exception:
                traceback.print_exc()

    def _handle(self, item: Dict[str, Any]) -> None:
        alert = item["alert"]
        occurred = dt.datetime.fromtimestamp(float(alert["ts"]), tz=dt.timezone.utc)
        appearance = describe_appearance(item["frame_raw"], alert.get("bbox") or [0, 0, 0, 0], alert.get("landmarks"))

        db = SessionLocal()
        try:
            ev = Event(
                alert_id=alert["alert_id"],
                task_id=alert.get("task_id"),
                camera_id=alert["camera_id"],
                person_id=int(alert["person_id"]),
                track_id=alert.get("track_id"),
                level=alert["level"],
                score=float(alert["score"]),
                rule_score=alert.get("rule_score"),
                model_score=alert.get("model_score"),
                indices=alert.get("indices") or {},
                indices_peak=alert.get("indices_peak"),
                predicted_next=alert.get("predicted_next"),
                triggers=alert.get("triggers"),
                aux=alert.get("aux"),
                action=alert.get("action"),
                action_conf=alert.get("action_conf"),
                bbox=alert.get("bbox"),
                trajectory=[list(p) for p in alert.get("trail") or []],
                camera_path=alert.get("camera_path"),
                score_history=alert.get("score_history"),
                appearance=appearance,
                occurred_at=occurred,
                status="pending",
            )
            db.add(ev)
            db.commit()
            db.refresh(ev)

            ev_dir = EVENTS_DIR / str(ev.id)
            ev_dir.mkdir(parents=True, exist_ok=True)
            if item["frame_vis"] is not None:
                snap = ev_dir / "snapshot.jpg"
                if cv2.imwrite(str(snap), item["frame_vis"]):
                    ev.snapshot_path = _rel(snap)
            clip = item["clip"]
            if clip:
                clip_dir = ev_dir / "clip"
                clip_dir.mkdir(exist_ok=True)
                for i, jpg in enumerate(clip):
                    (clip_dir / f"{i:04d}.jpg").write_bytes(jpg)
                ev.clip_dir = _rel(clip_dir)
                ev.clip_frames = len(clip)

            rag = get_rag_service()
            similar = rag.similar(alert.get("indices_peak") or alert.get("indices") or {}, alert["level"],
                                  k=3, exclude_key=f"event:{ev.id}")
            explanation, recs = rag.explain(alert, similar)
            ev.similar = similar
            ev.explanation = explanation
            ev.recommendations = recs
            db.commit()
            db.refresh(ev)

            rag.index_event(ev)
            data = event_to_dict(ev)
            broker.publish({"type": "alert", "event": data})
            notify_async(data)
            print(f"[EVENT] #{ev.id} {ev.level} {ev.camera_id} P{ev.person_id} score={ev.score:.2f}")
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()


event_service = EventService()
