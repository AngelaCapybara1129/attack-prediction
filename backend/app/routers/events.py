"""告警事件 API：查詢、處置狀態更新、相似事件、關鍵畫面與事件回放、文字檢索。"""
import datetime as dt
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user, get_user_header_or_query
from app.core.intent.constants import LEVEL_INDEX, LEVELS
from app.core.paths import BACKEND_DIR
from app.core.rag.service import get_rag_service
from app.database.session import get_db
from app.models.event import EVENT_STATUSES, Event
from app.models.user import User
from app.schemas.event import (
    EventListOut, EventOut, EventStats, EventUpdate, TextSearchRequest, TextSearchResponse,
)
from app.services.broker import broker
from app.services.event_service import event_to_dict

router = APIRouter(prefix="/api", tags=["Events"])
SECURITY_SCHEMA = [{"bearer": []}]
NO_CACHE = {"Cache-Control": "private, max-age=3600"}


def _get_event(db: Session, event_id: int) -> Event:
    ev = db.query(Event).filter(Event.id == event_id).first()
    if ev is None:
        raise HTTPException(status_code=404, detail="找不到事件")
    return ev


def _to_utc(d: dt.datetime) -> dt.datetime:
    """occurred_at 以 UTC 儲存；無時區的查詢參數視為伺服器本地時間。"""
    return (d if d.tzinfo else d.astimezone()).astimezone(dt.timezone.utc)


def _safe_path(rel: Optional[str]):
    if not rel:
        return None
    p = (BACKEND_DIR / rel).resolve()
    if BACKEND_DIR.resolve() not in p.parents:
        return None
    return p if p.exists() else None


@router.get("/events", response_model=EventListOut, openapi_extra={"security": SECURITY_SCHEMA})
def list_events(
    level: Optional[str] = Query(default=None, description="只看此等級，例如 L3"),
    min_level: Optional[str] = Query(default=None, description="最低等級，例如 L2"),
    status: Optional[str] = Query(default=None),
    camera_id: Optional[str] = None,
    person_id: Optional[int] = None,
    task_id: Optional[str] = None,
    since: Optional[dt.datetime] = None,
    until: Optional[dt.datetime] = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    q = db.query(Event)
    if level:
        q = q.filter(Event.level == level.upper())
    if min_level and min_level.upper() in LEVEL_INDEX:
        q = q.filter(Event.level.in_(LEVELS[LEVEL_INDEX[min_level.upper()]:]))
    if status:
        q = q.filter(Event.status == status)
    if camera_id:
        q = q.filter(Event.camera_id == camera_id)
    if person_id is not None:
        q = q.filter(Event.person_id == person_id)
    if task_id:
        q = q.filter(Event.task_id == task_id)
    if since:
        q = q.filter(Event.occurred_at >= _to_utc(since))
    if until:
        q = q.filter(Event.occurred_at < _to_utc(until))
    total = q.count()
    rows = q.order_by(Event.occurred_at.desc(), Event.id.desc()).offset(offset).limit(limit).all()
    return {"total": total, "items": [event_to_dict(ev) for ev in rows]}


@router.get("/events/stats", response_model=EventStats, openapi_extra={"security": SECURITY_SCHEMA})
def event_stats(
    hours: int = Query(default=24, ge=1, le=24 * 90),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours)
    base = db.query(Event).filter(Event.occurred_at >= since)
    by_level = {lv: 0 for lv in LEVELS}
    for lv, n in base.with_entities(Event.level, func.count(Event.id)).group_by(Event.level).all():
        by_level[lv] = int(n)
    by_status = {s: 0 for s in EVENT_STATUSES}
    for st, n in base.with_entities(Event.status, func.count(Event.id)).group_by(Event.status).all():
        by_status[st] = int(n)
    total = sum(by_level.values())
    closed = by_status.get("handled", 0) + by_status.get("dismissed", 0)
    return {
        "total": total,
        "by_level": by_level,
        "by_status": by_status,
        "handled_rate": round(closed / total, 4) if total else 1.0,
        "since": since,
    }


@router.get("/events/{event_id}", response_model=EventOut, openapi_extra={"security": SECURITY_SCHEMA})
def get_event(event_id: int, db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    return event_to_dict(_get_event(db, event_id))


@router.patch("/events/{event_id}", response_model=EventOut, openapi_extra={"security": SECURITY_SCHEMA})
def update_event(
    event_id: int,
    body: EventUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    ev = _get_event(db, event_id)
    if body.status is not None:
        if body.status not in EVENT_STATUSES:
            raise HTTPException(status_code=400, detail=f"status 必須為 {', '.join(EVENT_STATUSES)}")
        ev.status = body.status
        ev.handled_by = current_user.email
        ev.handled_at = dt.datetime.now(dt.timezone.utc)
    if body.note is not None:
        ev.note = body.note
    db.commit()
    db.refresh(ev)
    get_rag_service().index_event(ev)   # 處置紀錄回寫向量庫，成為往後的檢索依據
    data = event_to_dict(ev)
    broker.publish({"type": "event_updated", "event": data})
    return data


@router.get("/events/{event_id}/similar", openapi_extra={"security": SECURITY_SCHEMA})
def similar_events(
    event_id: int,
    k: int = Query(default=5, ge=1, le=20),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    ev = _get_event(db, event_id)
    items = get_rag_service().similar(ev.indices_peak or ev.indices or {}, ev.level, k=k,
                                      exclude_key=f"event:{ev.id}")
    return {"event_id": ev.id, "items": items}


@router.get("/events/{event_id}/snapshot")
def event_snapshot(event_id: int, db: Session = Depends(get_db), _: User = Depends(get_user_header_or_query)):
    ev = _get_event(db, event_id)
    p = _safe_path(ev.snapshot_path)
    if p is None:
        raise HTTPException(status_code=404, detail="此事件沒有關鍵畫面")
    return FileResponse(p, media_type="image/jpeg", headers=NO_CACHE)


@router.get("/events/{event_id}/clip/{index}")
def event_clip_frame(event_id: int, index: int, db: Session = Depends(get_db),
                     _: User = Depends(get_user_header_or_query)):
    ev = _get_event(db, event_id)
    if index < 0 or index >= (ev.clip_frames or 0):
        raise HTTPException(status_code=404, detail="影格不存在")
    clip_dir = _safe_path(ev.clip_dir)
    if clip_dir is None:
        raise HTTPException(status_code=404, detail="此事件沒有回放片段")
    p = clip_dir / f"{index:04d}.jpg"
    if not p.exists():
        raise HTTPException(status_code=404, detail="影格不存在")
    return FileResponse(p, media_type="image/jpeg", headers=NO_CACHE)


@router.post("/search/text", response_model=TextSearchResponse, openapi_extra={"security": SECURITY_SCHEMA})
def text_search(body: TextSearchRequest, db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    parsed, hits = get_rag_service().text_search(db, body.query, body.limit)
    ids = [h["event_id"] for h in hits]
    rows = {ev.id: ev for ev in db.query(Event).filter(Event.id.in_(ids)).all()} if ids else {}
    items = []
    for h in hits:
        ev = rows.get(h["event_id"])
        if ev is not None:
            items.append({**h, "event": event_to_dict(ev)})
    return {"query": body.query, "parsed": parsed, "items": items}
