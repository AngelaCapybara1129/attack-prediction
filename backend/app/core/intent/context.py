"""X9 空間脈絡：區域權重 × 時段權重（研究計畫表 1，Vossekuil et al., 2002）。"""
import datetime as _dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core.paths import CONFIG_DIR

DEFAULT_CONFIG_PATH = CONFIG_DIR / "spatial_context.json"


def _point_in_polygon(x: float, y: float, poly: Sequence[Sequence[float]]) -> bool:
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


class SpatialContext:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.default_region_weight = float(self.config.get("default_region_weight", 0.4))
        self.cameras: Dict[str, Dict[str, Any]] = self.config.get("cameras", {})
        self.time_weights: List[Dict[str, Any]] = self.config.get("time_weights", []) or [
            {"start": 0, "end": 24, "weight": 0.5, "label": "全日"}
        ]

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "SpatialContext":
        p = Path(path) if path else DEFAULT_CONFIG_PATH
        if p.is_file():
            try:
                return cls(json.loads(p.read_text(encoding="utf-8")))
            except Exception as e:  # 設定檔壞掉時不要讓整條管線停掉
                print(f"[X9] ⚠️ 無法讀取 {p}: {e}，改用預設值")
        return cls({})

    def region_weight(self, camera_id: str, foot_xy_norm: Tuple[float, float]) -> Tuple[float, str]:
        cam = self.cameras.get(camera_id, {})
        base = float(cam.get("region_weight", self.default_region_weight))
        best, name = base, "一般區域"
        for zone in cam.get("zones", []):
            poly = zone.get("polygon") or []
            if len(poly) >= 3 and _point_in_polygon(foot_xy_norm[0], foot_xy_norm[1], poly):
                w = float(zone.get("weight", base))
                if w > best:
                    best, name = w, str(zone.get("name", "高風險區"))
        return best, name

    def time_weight(self, ts: float) -> Tuple[float, str]:
        hour = _dt.datetime.fromtimestamp(ts).hour + _dt.datetime.fromtimestamp(ts).minute / 60.0
        for tw in self.time_weights:
            if float(tw["start"]) <= hour < float(tw["end"]):
                return float(tw["weight"]), str(tw.get("label", ""))
        return 0.5, ""

    def score(self, camera_id: str, ts: float, foot_xy_norm: Tuple[float, float]) -> Tuple[float, Dict[str, Any]]:
        rw, zone = self.region_weight(camera_id, foot_xy_norm)
        tw, period = self.time_weight(ts)
        return float(max(0.0, min(1.0, rw * tw))), {"zone": zone, "region_weight": rw, "period": period, "time_weight": tw}
