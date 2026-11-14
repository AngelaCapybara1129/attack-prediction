"""RAG 服務：事件向量化入庫、相似事件檢索、可解釋判斷說明與處置建議產生、文字檢索。

「生成」部分以檢索結果 + 指數量測值 + 處置知識庫組成結構化說明（範本式生成），
可離線運作、輸出可追溯；若之後要接 LLM，只需替換 explain()。
"""
import datetime as dt
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.intent.constants import (
    INDEX_KEYS, INDEX_NAMES, LEVEL_COLOR_NAME, LEVEL_INDEX, LEVEL_REACTION, LEVELS,
)
from app.core.rag.knowledge import load_seed_cases, recommend
from app.core.rag.query_parser import parse_query
from app.core.rag.vector_store import VectorStore, create_vector_store

DESCRIPTOR_DIM = len(INDEX_KEYS) + len(LEVELS)
STATUS_ZH = {"pending": "未處理", "handled": "已處理", "escalated": "需升級", "dismissed": "誤報"}


def event_descriptor(indices: Optional[Dict[str, float]], level: str) -> np.ndarray:
    indices = indices or {}
    x = np.array([float(indices.get(k, 0.0)) for k in INDEX_KEYS], dtype=np.float32)
    lv = np.zeros(len(LEVELS), dtype=np.float32)
    lv[LEVEL_INDEX.get(level, 0)] = 0.5
    return np.concatenate([x, lv])


def _fmt(v: Any, nd: int = 2) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def describe_index(key: str, value: float, aux: Dict[str, Any]) -> str:
    name = f"{key} {INDEX_NAMES[key]} {value:.2f}"
    detail = ""
    if key == "X1" and aux.get("weapon"):
        detail = f"偵測到「{aux['weapon']}」，可信度 {_fmt(aux.get('weapon_conf'))}"
    elif key == "X2":
        angles = [a for a in (aux.get("elbow_angle") or []) if a is not None]
        speeds = aux.get("wrist_speed") or []
        parts = []
        if angles:
            parts.append(f"手肘角度 {max(angles):.0f}°")
        if speeds:
            parts.append(f"手腕速度 {max(speeds):.1f} 軀幹長/秒")
        if aux.get("guard_pose"):
            parts.append("雙手護頭架勢")
        detail = "、".join(parts)
    elif key == "X3":
        detail = f"移動速度 {_fmt(aux.get('speed_mps'))} m/s、前傾 {_fmt(aux.get('lean_deg'), 0)}°"
        if aux.get("chasing") is not None:
            detail += f"、朝 P{aux['chasing']} 移動"
    elif key == "X4":
        detail = f"路徑方向熵 {_fmt(aux.get('path_entropy'))} bits、停留 {_fmt(aux.get('dwell_sec'), 0)} 秒"
    elif key == "X5" and aux.get("follow_target") is not None:
        detail = f"跟隨 P{aux['follow_target']} 已 {_fmt(aux.get('follow_sec'), 1)} 秒"
    elif key == "X6" and aux.get("nearest_pid") is not None:
        detail = f"與 P{aux['nearest_pid']} 距離 {_fmt(aux.get('nearest_dist_m'))} m、接近速度 {_fmt(aux.get('closing_mps'))} m/s"
    elif key == "X7":
        detail = f"群聚人數 {aux.get('group_size', 1)}"
        if aux.get("confront_sec"):
            detail += f"、面對面對峙 {_fmt(aux['confront_sec'], 1)} 秒"
    elif key == "X8":
        detail = f"累積風險分數 {_fmt(aux.get('history_score'))}"
    elif key == "X9":
        detail = f"{aux.get('zone', '')}・{aux.get('period', '')}（區域 {_fmt(aux.get('region_weight'))} × 時段 {_fmt(aux.get('time_weight'))}）"
    return f"{name}（{detail}）" if detail else name


class RAGService:
    def __init__(self, store: Optional[VectorStore] = None):
        self.store = store or create_vector_store(DESCRIPTOR_DIM)
        self._lock = threading.Lock()
        self._seed_cases()

    # ------------------------------------------------------------------ 入庫
    def _seed_cases(self) -> None:
        for case in load_seed_cases():
            try:
                self.store.upsert(
                    f"case:{case['id']}",
                    event_descriptor(case.get("indices"), case.get("level", "L0")),
                    {
                        "kind": "case",
                        "title": case.get("title"),
                        "level": case.get("level"),
                        "summary": case.get("summary"),
                        "handling": case.get("handling", []),
                        "references": case.get("references", []),
                        "synthetic": True,
                        "indices": case.get("indices"),
                    },
                )
            except Exception as e:
                print(f"[RAG] ⚠️ 示範情境入庫失敗 {case.get('id')}: {e}")

    def index_event(self, ev) -> None:
        indices = ev.indices_peak or ev.indices or {}
        top = sorted(indices.items(), key=lambda kv: kv[1], reverse=True)[:3]
        summary = "、".join(f"{k}{INDEX_NAMES.get(k, '')} {v:.2f}" for k, v in top if v >= 0.2) or "無明顯前兆"
        payload = {
            "kind": "event",
            "event_id": ev.id,
            "title": f"事件 #{ev.id}（{ev.camera_id} / P{ev.person_id}）",
            "level": ev.level,
            "camera_id": ev.camera_id,
            "person_id": ev.person_id,
            "occurred_at": ev.occurred_at.isoformat() if ev.occurred_at else None,
            "status": ev.status,
            "note": ev.note,
            "summary": summary,
            "appearance": ev.appearance,
            "indices": indices,
        }
        try:
            self.store.upsert(f"event:{ev.id}", event_descriptor(indices, ev.level), payload)
        except Exception as e:
            print(f"[RAG] ⚠️ 事件入庫失敗 #{ev.id}: {e}")

    # ------------------------------------------------------------------ 檢索
    def similar(self, indices: Dict[str, float], level: str, k: int = 3,
                exclude_key: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            hits = self.store.search(event_descriptor(indices, level), k=k + 2, exclude=exclude_key)
        except Exception as e:
            print(f"[RAG] ⚠️ 檢索失敗: {e}")
            return []
        out: List[Dict[str, Any]] = []
        for key, sim, payload in hits:
            out.append({
                "key": key,
                "kind": payload.get("kind"),
                "event_id": payload.get("event_id"),
                "title": payload.get("title"),
                "level": payload.get("level"),
                "similarity": round(float(sim), 4),
                "summary": payload.get("summary"),
                "status": payload.get("status"),
                "note": payload.get("note"),
                "handling": payload.get("handling"),
                "references": payload.get("references"),
                "synthetic": bool(payload.get("synthetic", False)),
                "occurred_at": payload.get("occurred_at"),
            })
            if len(out) >= k:
                break
        return out

    # ------------------------------------------------------------------ 生成說明
    def explain(self, alert: Dict[str, Any], similar: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        level = alert.get("level", "L0")
        x = alert.get("indices_peak") or alert.get("indices") or {}
        aux = alert.get("aux") or {}
        top = [k for k, v in sorted(x.items(), key=lambda kv: kv[1], reverse=True) if v >= 0.3][:3]

        lines: List[str] = []
        lines.append(
            f"【{level} {LEVEL_COLOR_NAME.get(level, '')}・{LEVEL_REACTION.get(level, '')}】"
            f"{alert.get('camera_id')} 人物 P{alert.get('person_id')}，風險分數 {float(alert.get('score', 0)):.2f}"
            + (f"（規則 {float(alert['rule_score']):.2f}／模型 {float(alert['model_score']):.2f}）"
               if alert.get("model_score") is not None else "")
        )
        if top:
            lines.append("主要前兆：" + "；".join(describe_index(k, float(x[k]), aux) for k in top))
        triggers = alert.get("triggers") or []
        if triggers:
            lines.append("觸發規則：" + "；".join(f"{t['description']}（{t.get('duration', 0)} 秒）" for t in triggers))
        if alert.get("action"):
            lines.append(f"ST-GCN 行為判讀：{alert.get('action_name') or alert['action']}（{float(alert.get('action_conf') or 0):.0%}）")
        pred = alert.get("predicted_next")
        if pred:
            rising = [k for k in INDEX_KEYS if float(pred.get(k, 0)) - float(x.get(k, 0)) >= 0.15]
            if rising:
                lines.append("LSTM 預測下一秒升高：" + "、".join(f"{k}{INDEX_NAMES[k]}→{float(pred[k]):.2f}" for k in rising))
        path = alert.get("camera_path") or []
        if len(path) > 1:
            lines.append("跨鏡頭路徑：" + " → ".join(p["camera_id"] for p in path))
        if similar:
            refs = []
            for s in similar:
                tag = "示範情境" if s.get("kind") == "case" else f"歷史事件（{STATUS_ZH.get(s.get('status') or '', '未處理')}）"
                note = f"，處置紀錄：{s['note']}" if s.get("note") else ""
                refs.append(f"{s.get('title')}［{tag}，相似度 {float(s['similarity']):.0%}］{note}")
            lines.append("相似參考：" + "；".join(refs))

        recs = recommend(level, top)
        for s in similar:
            if s.get("kind") == "case" and float(s.get("similarity", 0)) >= 0.85:
                for h in s.get("handling") or []:
                    if h not in recs:
                        recs.append(h)
        return "\n".join(lines), recs

    # ------------------------------------------------------------------ 文字檢索
    def text_search(self, db, query: str, limit: int = 20) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        from app.models.event import Event

        parsed = parse_query(query)
        q = db.query(Event)
        # occurred_at 一律以 UTC 儲存；查詢條件由本地時間換成 UTC
        if parsed["since"] is not None:
            q = q.filter(Event.occurred_at >= parsed["since"].astimezone(dt.timezone.utc))
        if parsed["until"] is not None:
            q = q.filter(Event.occurred_at < parsed["until"].astimezone(dt.timezone.utc))
        if parsed["levels"]:
            q = q.filter(Event.level.in_(parsed["levels"]))
        if parsed["statuses"]:
            q = q.filter(Event.status.in_(parsed["statuses"]))
        if parsed["cameras"]:
            q = q.filter(Event.camera_id.in_(parsed["cameras"]))
        if parsed["person_id"] is not None:
            q = q.filter(Event.person_id == parsed["person_id"])
        candidates = q.order_by(Event.occurred_at.desc()).limit(500).all()

        qvec = None
        if parsed["indices"]:
            qvec = np.array([1.0 if k in parsed["indices"] else 0.0 for k in INDEX_KEYS], dtype=np.float32)
            qvec /= np.linalg.norm(qvec)

        now = dt.datetime.now(dt.timezone.utc)
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for ev in candidates:
            peak = ev.indices_peak or ev.indices or {}
            reasons: List[str] = []
            score = 0.0
            if qvec is not None:
                x = np.array([float(peak.get(k, 0.0)) for k in INDEX_KEYS], dtype=np.float32)
                n = float(np.linalg.norm(x))
                s_b = float(np.dot(qvec, x) / n) if n > 1e-6 else 0.0
                score += s_b
                for k in parsed["indices"]:
                    if float(peak.get(k, 0)) >= 0.3:
                        reasons.append(f"{k}{INDEX_NAMES[k]} {float(peak[k]):.2f}")
                if not reasons:
                    continue
            else:
                score += 0.5
            if parsed["colors"]:
                app = ev.appearance or {}
                if app.get("upper") in parsed["colors"]:
                    score += 0.35
                    reasons.append(f"上身{app.get('upper_zh') or app.get('upper')}色")
                elif app.get("lower") in parsed["colors"]:
                    score += 0.25
                    reasons.append(f"下身{app.get('lower_zh') or app.get('lower')}色")
                else:
                    continue
            occurred = ev.occurred_at
            if occurred is not None:
                if occurred.tzinfo is None:
                    occurred = occurred.replace(tzinfo=dt.timezone.utc)
                age_h = max(0.0, (now - occurred).total_seconds() / 3600.0)
                score += 0.1 / (1.0 + age_h / 24.0)
            score += 0.05 * LEVEL_INDEX.get(ev.level, 0)
            scored.append((score, {"event_id": ev.id, "score": round(score, 4), "reasons": reasons}))

        scored.sort(key=lambda t: t[0], reverse=True)
        parsed_out = {**parsed,
                      "since": parsed["since"].isoformat() if parsed["since"] else None,
                      "until": parsed["until"].isoformat() if parsed["until"] else None}
        return parsed_out, [item for _, item in scored[:limit]]


_service: Optional[RAGService] = None
_service_lock = threading.Lock()


def get_rag_service() -> RAGService:
    global _service
    with _service_lock:
        if _service is None:
            _service = RAGService()
        return _service
