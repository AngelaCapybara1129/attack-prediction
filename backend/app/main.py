import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from app.database.session import Base, engine
from app.models import event as _event_model  # noqa: F401  確保 events 資料表註冊到 metadata
from app.models import user as _user_model  # noqa: F401
from app.routers import auth, cameras, events, intent, users, ws, yolo
from app.services.broker import broker
from app.services.event_service import event_service


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # 開發環境沒跑 alembic 時，自動補建缺少的資料表（已存在的表不受影響）
    Base.metadata.create_all(bind=engine)
    broker.bind_loop(asyncio.get_running_loop())
    event_service.start()
    yield


app = FastAPI(title="Attack Prediction API", lifespan=lifespan)

_default_origins = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000"
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", _default_origins).split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(yolo.router)
app.include_router(cameras.router)
app.include_router(events.router)
app.include_router(intent.router)
app.include_router(ws.router)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Attack Prediction API",
        version="1.1.0",
        description="YOLO26 + ByteTrack + MediaPipe Pose + ST-GCN/LSTM + AHP + RAG 攻擊意圖預測系統",
        routes=app.routes,
    )

    if "components" not in openapi_schema:
        openapi_schema["components"] = {}

    openapi_schema["components"]["securitySchemes"] = {
        "bearer": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "請輸入 JWT Token（不需要加 Bearer 前綴）"
        }
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.get("/")
async def root():
    return {"message": "Backend API is running"}
