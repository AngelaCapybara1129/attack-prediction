"""通訊軟體推播（研究計畫：L3 推播、L4 立即警告）。

設定 NOTIFY_WEBHOOK_URL 後，等級 ≥ NOTIFY_MIN_LEVEL 的事件會以 JSON POST 推送，
可串接 Slack / Discord / Teams / Telegram bot / LINE Messaging API 中介服務等 webhook。
NOTIFY_WEBHOOK_FORMAT=slack|discord|generic（預設 generic）。
"""
import json
import os
import threading
import urllib.request
from typing import Any, Dict

from app.core.intent.constants import LEVEL_COLOR_NAME, LEVEL_INDEX, LEVEL_REACTION


def _build_body(event: Dict[str, Any], fmt: str) -> Dict[str, Any]:
    level = event.get("level", "L0")
    expl_lines = (event.get("explanation") or "").splitlines()
    main_reason = expl_lines[1] if len(expl_lines) > 1 else ""
    text = (
        f"[{level} {LEVEL_COLOR_NAME.get(level, '')}・{LEVEL_REACTION.get(level, '')}] "
        f"{event.get('camera_id')} 人物 P{event.get('person_id')} 風險分數 {float(event.get('score', 0)):.2f}\n"
        f"{main_reason}"
    ).strip()
    if fmt == "slack":
        return {"text": text}
    if fmt == "discord":
        return {"content": text}
    return {"type": "attack_intent_alert", "text": text, "event": event}


def notify_async(event: Dict[str, Any]) -> None:
    url = os.getenv("NOTIFY_WEBHOOK_URL", "").strip()
    if not url:
        return
    min_level = os.getenv("NOTIFY_MIN_LEVEL", "L3").strip().upper()
    if LEVEL_INDEX.get(event.get("level", "L0"), 0) < LEVEL_INDEX.get(min_level, 3):
        return
    fmt = os.getenv("NOTIFY_WEBHOOK_FORMAT", "generic").strip().lower()

    def _send() -> None:
        try:
            body = json.dumps(_build_body(event, fmt), ensure_ascii=False, default=str).encode("utf-8")
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=5) as resp:
                resp.read()
        except Exception as e:
            print(f"[NOTIFY] ⚠️ 推播失敗: {e}")

    threading.Thread(target=_send, daemon=True).start()
