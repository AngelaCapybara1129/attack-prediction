from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    alert_id: str
    task_id: Optional[str] = None
    camera_id: str
    person_id: int
    track_id: Optional[int] = None
    level: str
    score: float
    rule_score: Optional[float] = None
    model_score: Optional[float] = None
    indices: Dict[str, float] = Field(default_factory=dict)
    indices_peak: Optional[Dict[str, float]] = None
    predicted_next: Optional[Dict[str, float]] = None
    triggers: Optional[List[Dict[str, Any]]] = None
    aux: Optional[Dict[str, Any]] = None
    action: Optional[str] = None
    action_conf: Optional[float] = None
    bbox: Optional[List[int]] = None
    trajectory: Optional[List[List[int]]] = None
    camera_path: Optional[List[Dict[str, Any]]] = None
    score_history: Optional[List[Dict[str, Any]]] = None
    appearance: Optional[Dict[str, Any]] = None
    explanation: Optional[str] = None
    recommendations: Optional[List[str]] = None
    similar: Optional[List[Dict[str, Any]]] = None
    clip_frames: int = 0
    has_snapshot: bool = False
    status: str
    handled_by: Optional[str] = None
    handled_at: Optional[datetime] = None
    note: Optional[str] = None
    occurred_at: datetime
    created_at: Optional[datetime] = None


class EventListOut(BaseModel):
    total: int
    items: List[EventOut]


class EventUpdate(BaseModel):
    status: Optional[str] = Field(default=None, description="pending / handled / escalated / dismissed")
    note: Optional[str] = Field(default=None, max_length=2000)


class EventStats(BaseModel):
    total: int
    by_level: Dict[str, int]
    by_status: Dict[str, int]
    handled_rate: float
    since: datetime


class TextSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=200, examples=["上週穿黑衣揮拳", "深夜尾隨 cam1"])
    limit: int = Field(default=20, ge=1, le=100)


class TextSearchResponse(BaseModel):
    query: str
    parsed: Dict[str, Any]
    items: List[Dict[str, Any]]
