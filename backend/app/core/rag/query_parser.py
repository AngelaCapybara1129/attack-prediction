"""文字查詢解析：把「上週穿黑衣揮拳」這類自然語言轉成 行為指數 / 外觀顏色 / 時間 / 等級 / 鏡頭 條件。

採關鍵詞 → 行為語義映射，不需額外下載語言模型即可離線運作。
"""
import datetime as dt
import re
from typing import Any, Dict, List, Optional

from app.core.intent.constants import INDEX_KEYS, LEVELS

INDEX_KEYWORDS: Dict[str, List[str]] = {
    "X1": ["刀", "武器", "持刀", "拔刀", "持械", "棍", "球棒", "剪刀", "槍", "knife", "weapon", "bat", "gun"],
    "X2": ["揮拳", "出拳", "拳", "毆打", "打人", "打架", "鬥毆", "攻擊", "punch", "fight", "hit"],
    "X3": ["追逐", "追趕", "追", "衝刺", "奔跑", "狂奔", "跑", "chase", "run"],
    "X4": ["徘徊", "逗留", "遊蕩", "來回", "loiter"],
    "X5": ["尾隨", "跟蹤", "跟隨", "跟著", "follow", "stalk"],
    "X6": ["逼近", "貼近", "侵入", "推擠", "推人", "靠近", "push", "approach"],
    "X7": ["群聚", "聚集", "對峙", "圍圈", "圍毆", "群眾", "crowd", "group"],
    "X8": ["累犯", "再次", "重複", "常客", "repeat"],
    "X9": ["深夜", "凌晨", "夜間", "高風險區", "出入口", "night"],
}

COLOR_KEYWORDS: Dict[str, List[str]] = {
    "black": ["黑", "black"], "white": ["白", "white"], "gray": ["灰", "gray", "grey"],
    "red": ["紅", "red"], "orange": ["橘", "橙", "orange"], "yellow": ["黃", "yellow"],
    "green": ["綠", "green"], "blue": ["藍", "blue"], "purple": ["紫", "purple"],
    "pink": ["粉", "pink"], "brown": ["棕", "咖啡", "褐", "brown"], "cyan": ["青", "cyan"],
}

LEVEL_KEYWORDS: Dict[str, List[str]] = {
    "L4": ["l4", "紅色警告", "立即警告", "最高"],
    "L3": ["l3", "橘色"],
    "L2": ["l2", "黃色"],
    "L1": ["l1", "綠色"],
    "L0": ["l0", "藍色"],
}

STATUS_KEYWORDS: Dict[str, List[str]] = {
    "pending": ["未處理", "待處理"],
    "handled": ["已處理"],
    "escalated": ["需升級", "升級"],
    "dismissed": ["誤報"],
}


def _start_of_day(d: dt.datetime) -> dt.datetime:
    return d.replace(hour=0, minute=0, second=0, microsecond=0)


def _parse_time(q: str, now: dt.datetime) -> Dict[str, Optional[dt.datetime]]:
    since: Optional[dt.datetime] = None
    until: Optional[dt.datetime] = None
    m = re.search(r"(最近|過去|近)\s*(\d+)\s*(分鐘|小時|天|日|週|周)", q)
    if m:
        n = int(m.group(2))
        unit = m.group(3)
        delta = {"分鐘": dt.timedelta(minutes=n), "小時": dt.timedelta(hours=n), "天": dt.timedelta(days=n),
                 "日": dt.timedelta(days=n), "週": dt.timedelta(weeks=n), "周": dt.timedelta(weeks=n)}[unit]
        since = now - delta
    elif "今天" in q or "今日" in q:
        since = _start_of_day(now)
    elif "昨天" in q or "昨日" in q:
        since = _start_of_day(now) - dt.timedelta(days=1)
        until = _start_of_day(now)
    elif "上週" in q or "上周" in q or "上星期" in q:
        start_this_week = _start_of_day(now) - dt.timedelta(days=now.weekday())
        since = start_this_week - dt.timedelta(weeks=1)
        until = start_this_week
    elif "本週" in q or "這週" in q or "本周" in q or "這周" in q:
        since = _start_of_day(now) - dt.timedelta(days=now.weekday())
    elif "本月" in q or "這個月" in q:
        since = _start_of_day(now).replace(day=1)
    return {"since": since, "until": until}


def parse_query(query: str, now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    now = now or dt.datetime.now()
    q = query.strip()
    ql = q.lower()

    indices = [k for k in INDEX_KEYS if any(w in ql for w in INDEX_KEYWORDS[k])]
    colors = [c for c, words in COLOR_KEYWORDS.items() if any(w in ql for w in words)]
    # 「紅色警告」「黃色」等等級用語不應被當成衣服顏色
    levels = [lv for lv in LEVELS if any(w in ql for w in LEVEL_KEYWORDS[lv])]
    if levels:
        for lv in levels:
            for w in LEVEL_KEYWORDS[lv]:
                if w in ql and len(w) > 2:
                    ql = ql.replace(w, "")
        colors = [c for c, words in COLOR_KEYWORDS.items() if any(w in ql for w in words)]
    statuses = [s for s, words in STATUS_KEYWORDS.items() if any(w in q for w in words)]
    cameras = re.findall(r"cam\s*\d+", ql)
    cameras = [c.replace(" ", "") for c in cameras]
    person = re.search(r"(?:person|人物|p)\s*-?\s*(\d+)", ql)
    t = _parse_time(q, now)
    return {
        "indices": indices,
        "colors": colors,
        "levels": levels,
        "statuses": statuses,
        "cameras": cameras,
        "person_id": int(person.group(1)) if person else None,
        "since": t["since"],
        "until": t["until"],
    }
