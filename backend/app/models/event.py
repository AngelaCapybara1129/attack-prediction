"""告警事件：儲存告警、人物軌跡、前兆指數 X1–X9、風險等級 L0–L4、處置狀態與關鍵片段（研究計畫：資料庫）。"""
from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.sql import func

from app.database.session import Base

EVENT_STATUSES = ("pending", "handled", "escalated", "dismissed")
EVENT_STATUS_NAMES = {
    "pending": "未處理",
    "handled": "已處理",
    "escalated": "需升級",
    "dismissed": "誤報",
}


class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(String(32), unique=True, index=True, nullable=False)
    task_id = Column(String(32), index=True, nullable=True)
    camera_id = Column(String(64), index=True, nullable=False)
    person_id = Column(Integer, index=True, nullable=False)
    track_id = Column(Integer, nullable=True)

    level = Column(String(4), index=True, nullable=False)
    score = Column(Float, nullable=False, default=0.0)
    rule_score = Column(Float, nullable=True)
    model_score = Column(Float, nullable=True)

    indices = Column(JSON, nullable=False, default=dict)          # 告警當下 X1–X9
    indices_peak = Column(JSON, nullable=True)                    # 前 5 秒峰值
    predicted_next = Column(JSON, nullable=True)                  # LSTM 預測的下一秒 X1–X9
    triggers = Column(JSON, nullable=True)                        # 表 2 觸發規則
    aux = Column(JSON, nullable=True)                             # 可解釋量測值（速度、角度、距離…）
    action = Column(String(32), nullable=True)
    action_conf = Column(Float, nullable=True)

    bbox = Column(JSON, nullable=True)
    trajectory = Column(JSON, nullable=True)                      # 畫面中的移動軌跡
    camera_path = Column(JSON, nullable=True)                     # 跨鏡頭路徑
    score_history = Column(JSON, nullable=True)
    appearance = Column(JSON, nullable=True)                      # 上/下身主色，供文字檢索

    explanation = Column(Text, nullable=True)                     # RAG 產生的判斷說明
    recommendations = Column(JSON, nullable=True)
    similar = Column(JSON, nullable=True)                         # 相似歷史事件/案例

    snapshot_path = Column(String(512), nullable=True)
    clip_dir = Column(String(512), nullable=True)
    clip_frames = Column(Integer, nullable=False, default=0)

    status = Column(String(16), index=True, nullable=False, default="pending")
    handled_by = Column(String(255), nullable=True)
    handled_at = Column(DateTime(timezone=True), nullable=True)
    note = Column(Text, nullable=True)

    occurred_at = Column(DateTime(timezone=True), index=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
