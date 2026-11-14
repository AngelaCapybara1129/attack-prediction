"""X1 武器準備：武器偵測。

預設用 COCO 預訓練的 YOLO26 偵測 knife / scissors / baseball bat；
研究計畫的微調模型（UCF-Crime / RWF-2000 / 自建武器資料集）訓練好後，
設定環境變數 WEAPON_MODEL_PATH 指向該權重，該模型的所有類別都會被視為武器。
"""
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from app.core.intent.features import WeaponObservation
from app.core.paths import WEIGHTS_DIR

COCO_WEAPON_CLASSES: Dict[int, str] = {34: "baseball bat", 43: "knife", 76: "scissors"}
WEAPON_CONF = float(os.getenv("WEAPON_CONF", "0.30"))
WEAPON_EVERY_N = int(os.getenv("WEAPON_EVERY_N", "2"))
IMG_SIZE = 640


class WeaponDetector:
    def __init__(self, model_path: Optional[str] = None, conf: float = WEAPON_CONF, every_n: int = WEAPON_EVERY_N):
        from ultralytics import YOLO

        custom = model_path or os.getenv("WEAPON_MODEL_PATH", "").strip()
        if custom and Path(custom).is_file():
            self.model = YOLO(custom)
            self.class_map: Optional[Dict[int, str]] = None   # 自訂模型：全部類別皆為武器
            print(f"[WEAPON] ✅ 使用自訂武器模型 {custom}")
        else:
            self.model = YOLO(str(WEIGHTS_DIR / "yolo26n.pt"))
            self.class_map = COCO_WEAPON_CLASSES
        self.conf = conf
        self.every_n = max(1, every_n)
        self._counter: Dict[str, int] = {}
        self._cache: Dict[str, List[WeaponObservation]] = {}
        self._lock = threading.Lock()

    def detect(self, camera_id: str, frame: np.ndarray) -> List[WeaponObservation]:
        """每 every_n 幀實際推論一次，其餘幀沿用上一次結果以節省運算。"""
        n = self._counter.get(camera_id, 0)
        self._counter[camera_id] = n + 1
        if n % self.every_n != 0:
            return self._cache.get(camera_id, [])

        kwargs = {"conf": self.conf, "imgsz": IMG_SIZE, "verbose": False}
        if self.class_map is not None:
            kwargs["classes"] = list(self.class_map.keys())
        with self._lock:
            results = self.model.predict(frame, **kwargs)
        out: List[WeaponObservation] = []
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            names = r.names if hasattr(r, "names") else {}
            for box, cls, cf in zip(r.boxes.xyxy.int().cpu().tolist(), r.boxes.cls.int().cpu().tolist(),
                                    r.boxes.conf.cpu().tolist()):
                label = (self.class_map or {}).get(cls) or names.get(cls, str(cls))
                out.append(WeaponObservation(label=label, conf=float(cf), bbox=box))
        self._cache[camera_id] = out
        return out
