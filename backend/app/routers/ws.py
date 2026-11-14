"""WebSocket 即時告警通道：ws://<host>:8000/ws/alerts?token=<JWT>"""
import asyncio
import time

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.core.dependencies import user_from_token
from app.database.session import SessionLocal
from app.services.broker import broker

router = APIRouter(tags=["WebSocket"])


@router.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket, token: str = Query(default="")):
    db = SessionLocal()
    try:
        user = user_from_token(db, token)
    finally:
        db.close()
    if user is None:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    q = broker.subscribe()
    try:
        await websocket.send_json({"type": "hello", "user": user.email, "ts": time.time()})
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=20.0)
            except asyncio.TimeoutError:
                msg = {"type": "ping", "ts": time.time()}
            await websocket.send_json(msg)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS] 連線結束: {e!r}")
    finally:
        broker.unsubscribe(q)
