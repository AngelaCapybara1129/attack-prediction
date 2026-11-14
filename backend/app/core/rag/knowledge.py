"""處置建議知識：依風險等級與主要前兆指數提供的標準作業建議，以及示範情境庫載入。"""
import json
from pathlib import Path
from typing import Dict, List

LEVEL_ACTIONS: Dict[str, List[str]] = {
    "L0": ["持續監控"],
    "L1": ["記錄事件，持續觀察是否升溫"],
    "L2": ["於畫面標示目標，通知現場人員留意"],
    "L3": ["推播通知值勤人員，派遣最近人員前往", "保留事件前後影像"],
    "L4": ["立即通報 110 與現場保全", "疏散周邊民眾並持續鎖定目標"],
}

INDEX_ACTIONS: Dict[str, List[str]] = {
    "X1": ["勿徒手奪械，與持械者保持距離", "確認武器類型並回報"],
    "X2": ["介入降溫，必要時隔開衝突雙方"],
    "X3": ["通知前方鏡頭區域人員攔阻或注意"],
    "X4": ["安排巡邏人員順道關懷詢問"],
    "X5": ["優先保護被尾隨者，確認其安全"],
    "X6": ["派員到場阻止進一步肢體接觸"],
    "X7": ["觀察群體動態，避免單獨人員貿然進入"],
    "X8": ["調閱該人物歷史告警，評估是否為重複行為"],
    "X9": ["加強該區域與時段的巡邏密度"],
}

SEED_CASES_PATH = Path(__file__).resolve().parent / "seed_cases.json"


def load_seed_cases() -> List[Dict]:
    try:
        data = json.loads(SEED_CASES_PATH.read_text(encoding="utf-8"))
        return list(data.get("cases", []))
    except Exception as e:
        print(f"[RAG] ⚠️ 無法讀取示範情境庫 {SEED_CASES_PATH}: {e}")
        return []


def recommend(level: str, top_indices: List[str]) -> List[str]:
    recs: List[str] = list(LEVEL_ACTIONS.get(level, []))
    for key in top_indices:
        for r in INDEX_ACTIONS.get(key, []):
            if r not in recs:
                recs.append(r)
    return recs
