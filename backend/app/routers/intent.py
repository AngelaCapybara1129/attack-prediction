"""攻擊意圖即時狀態與系統設定 API。"""
from fastapi import APIRouter, Depends, HTTPException

from app.core.dependencies import get_current_user, require_admin
from app.core.intent import registry
from app.core.intent.ahp import ahp_summary
from app.core.intent.constants import (
    INDEX_CRITERIA, INDEX_KEYS, INDEX_NAMES, LEVEL_COLOR_NAME, LEVEL_REACTION, LEVELS, RISK_SCALE, SCORE_THRESHOLDS,
)
from app.core.intent.predictor import get_predictor, predictor_status
from app.core.intent.risk import RULES
from app.core.rag.service import get_rag_service
from app.models.user import User
from app.services.broker import broker

router = APIRouter(prefix="/api/intent", tags=["Intent"])
SECURITY_SCHEMA = [{"bearer": []}]


@router.get("/live/{task_id}", openapi_extra={"security": SECURITY_SCHEMA})
def live_state(task_id: str, _: User = Depends(get_current_user)):
    engine = registry.get(task_id)
    if engine is None:
        raise HTTPException(status_code=404, detail="找不到此任務的分析引擎（任務未啟動或已結束）")
    return engine.snapshot()


@router.get("/live/{task_id}/person/{person_id}", openapi_extra={"security": SECURITY_SCHEMA})
def live_person(task_id: str, person_id: int, _: User = Depends(get_current_user)):
    engine = registry.get(task_id)
    if engine is None:
        raise HTTPException(status_code=404, detail="找不到此任務的分析引擎")
    detail = engine.person_detail(person_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="目前畫面中沒有此人物")
    return detail


@router.get("/tasks", openapi_extra={"security": SECURITY_SCHEMA})
def tasks(_: User = Depends(get_current_user)):
    return {"tasks": registry.list_tasks()}


@router.get("/config")
def intent_config():
    """表 1 指數定義、表 2 規則、AHP 權重與一致性比率（公開資訊，前端顯示用）。"""
    th = {lv: None for lv in LEVELS}
    for lower, lv in SCORE_THRESHOLDS:
        th[lv] = lower
    return {
        "indices": [{"key": k, "name": INDEX_NAMES[k], "criterion": INDEX_CRITERIA[k]} for k in INDEX_KEYS],
        "levels": [
            {"level": lv, "color": LEVEL_COLOR_NAME[lv], "reaction": LEVEL_REACTION[lv], "min_score": th[lv] or 0.0}
            for lv in LEVELS
        ],
        "rules": [{"key": r.key, "level": r.level, "min_duration": r.min_duration, "description": r.description} for r in RULES],
        "risk_scale": RISK_SCALE,
        "ahp": ahp_summary(),
    }


@router.get("/status", openapi_extra={"security": SECURITY_SCHEMA})
def system_status(_: User = Depends(get_current_user)):
    rag = get_rag_service()
    return {
        "model": predictor_status(),
        "vector_store": {"backend": rag.store.backend, "count": rag.store.count()},
        "websocket_subscribers": broker.subscriber_count,
        "tasks": registry.list_tasks(),
    }


@router.post("/model/reload", openapi_extra={"security": SECURITY_SCHEMA})
def reload_model(_: User = Depends(require_admin)):
    """重新載入 STGCN/checkpoints/intent_model.pt（訓練完成後不必重啟後端；下一個任務生效）。"""
    get_predictor(reload=True)
    return predictor_status()
