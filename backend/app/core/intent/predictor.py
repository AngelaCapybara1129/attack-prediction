"""即時推論時載入訓練好的 DualChannelIntentNet。

找不到 checkpoint 時回傳 None，系統退回「X1–X9 規則 + AHP」基線模式，仍可完整運作。
checkpoint 由 STGCN/scripts/train.py 產生，預設路徑 STGCN/checkpoints/intent_model.pt，
可用環境變數 INTENT_MODEL_PATH 覆寫。
"""
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from app.core.intent.constants import ACTION_CLASSES, LEVELS
from app.core.paths import DEFAULT_INTENT_MODEL

DEFAULT_SEQ_LEN = 150
DEFAULT_WINDOW_SEC = 5.0
DEFAULT_HORIZON_SEC = 1.0


class IntentPredictor:
    def __init__(self, model, config: Dict[str, Any], device, path: Path, metrics: Optional[Dict] = None):
        self.model = model
        self.config = config
        self.device = device
        self.path = path
        self.metrics = metrics or {}
        self.action_classes = list(config.get("action_classes", ACTION_CLASSES))
        self.seq_len = int(config.get("seq_len", DEFAULT_SEQ_LEN))
        self.window_sec = float(config.get("window_sec", DEFAULT_WINDOW_SEC))
        self.horizon_sec = float(config.get("horizon_sec", DEFAULT_HORIZON_SEC))

    @classmethod
    def load(cls, path: Optional[os.PathLike] = None) -> Optional["IntentPredictor"]:
        p = Path(path or os.getenv("INTENT_MODEL_PATH") or DEFAULT_INTENT_MODEL)
        if not p.is_file():
            print(f"[INTENT] 找不到模型 {p}，使用規則 + AHP 基線模式")
            return None
        try:
            import torch
            from app.core.intent.nets import build_model

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ckpt = torch.load(str(p), map_location=device, weights_only=True)
            config = ckpt.get("config", {})
            model = build_model(config).to(device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            print(f"[INTENT] ✅ 已載入雙通道模型 {p}（device={device}）")
            return cls(model, config, device, p, ckpt.get("metrics"))
        except Exception as e:
            print(f"[INTENT] ⚠️ 模型載入失敗 {p}: {e}；改用規則基線")
            return None

    def predict(self, skel: np.ndarray, xseq: np.ndarray) -> Dict[str, Any]:
        """skel: (3,T,33,1)，xseq: (T,9)。"""
        import torch

        from app.core.intent.nets import to_tensor

        with torch.no_grad():
            out = self.model(to_tensor(skel[None], self.device), to_tensor(xseq[None], self.device))
            action_p = torch.softmax(out["action_logits"], dim=1)[0].cpu().numpy()
            level_p = torch.softmax(out["level_logits"], dim=1)[0].cpu().numpy()
            next_x = out["next_x"][0].cpu().numpy()
            risk = float(out["risk"][0].cpu().item())
        a_idx = int(np.argmax(action_p))
        l_idx = int(np.argmax(level_p))
        return {
            "action": self.action_classes[a_idx] if a_idx < len(self.action_classes) else str(a_idx),
            "action_conf": float(action_p[a_idx]),
            "action_probs": {c: float(p) for c, p in zip(self.action_classes, action_p)},
            "next_x": next_x.astype(float).tolist(),
            "risk": risk,
            "level": LEVELS[l_idx] if l_idx < len(LEVELS) else "L0",
            "level_probs": {lv: float(p) for lv, p in zip(LEVELS, level_p)},
        }

    def status(self) -> Dict[str, Any]:
        return {
            "loaded": True,
            "path": str(self.path),
            "device": str(self.device),
            "action_classes": self.action_classes,
            "seq_len": self.seq_len,
            "window_sec": self.window_sec,
            "horizon_sec": self.horizon_sec,
            "metrics": self.metrics,
        }


_predictor: Optional[IntentPredictor] = None
_loaded = False
_lock = threading.Lock()


def get_predictor(reload: bool = False) -> Optional[IntentPredictor]:
    global _predictor, _loaded
    with _lock:
        if reload or not _loaded:
            _predictor = IntentPredictor.load()
            _loaded = True
        return _predictor


def predictor_status() -> Dict[str, Any]:
    p = _predictor if _loaded else None
    if p is None:
        return {"loaded": False, "path": str(Path(os.getenv("INTENT_MODEL_PATH") or DEFAULT_INTENT_MODEL)), "mode": "rule+AHP"}
    return {**p.status(), "mode": "rule+AHP+ST-GCN/LSTM"}
