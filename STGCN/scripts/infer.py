# scripts/infer.py
"""單一樣本推論。

    python scripts/infer.py --sample data/processed/test/samples/xxx.npz              # 雙通道模型
    python scripts/infer.py --sample some_clip.npy --model checkpoints/best_stgcn.pth  # 只有 ST-GCN 行為分類
"""
import argparse
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from app.core.intent.ahp import weighted_sum  # noqa: E402
from app.core.intent.constants import (  # noqa: E402
    ACTION_CLASSES, ACTION_NAMES, INDEX_KEYS, INDEX_NAMES, LEVELS, RISK_SCALE, score_to_level,
)
from app.core.intent.nets import STGCN, build_model  # noqa: E402


def infer_action_only(model_path, sample_path, device):
    data = np.load(sample_path).astype(np.float32)
    if data.ndim != 4:
        raise ValueError(f"Input sample shape must be (C, T, V, M), got {data.shape}")
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    classes = list(ckpt.get("action_classes", ACTION_CLASSES)) if isinstance(ckpt, dict) else ACTION_CLASSES
    model = STGCN(num_class=len(classes)).to(device)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(torch.tensor(data).unsqueeze(0).to(device)), dim=1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    print(f"Predicted class: {idx} ({classes[idx]} {ACTION_NAMES.get(classes[idx], '')})  confidence={probs[idx]:.4f}")
    for i, p in enumerate(probs):
        print(f"  {i} ({classes[i]}): {p:.4f}")


def infer_intent(model_path, sample_path, device):
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    classes = list(ckpt.get("config", {}).get("action_classes", ACTION_CLASSES))
    model = build_model(ckpt.get("config", {})).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with np.load(sample_path) as z:
        skel = torch.tensor(z["skel"][None], dtype=torch.float32, device=device)
        xseq = torch.tensor(z["xseq"][None], dtype=torch.float32, device=device)
        truth = {k: z[k] for k in ("next_x", "risk", "level", "action") if k in z}
    with torch.no_grad():
        out = model(skel, xseq)
    a = F.softmax(out["action_logits"], 1)[0].cpu().numpy()
    lv = F.softmax(out["level_logits"], 1)[0].cpu().numpy()
    nx = out["next_x"][0].cpu().numpy()
    risk = float(out["risk"][0])
    ta = int(truth["action"]) if "action" in truth else -1
    pa = classes[int(a.argmax())]
    print(f"攻擊類型：{pa} {ACTION_NAMES.get(pa, '')}（{a.max():.2%}）  真值={classes[ta] if ta >= 0 else '-'}")
    for c, pc in zip(classes, a):
        print(f"  {c:<14} {pc:.2%}")
    print(f"風險分數（模型）：{risk:.2f} → {score_to_level(risk)}   預測等級：{LEVELS[int(lv.argmax())]}（{lv.max():.2%}）"
          f"  真值={LEVELS[int(truth['level'])] if 'level' in truth else '-'}")
    print(f"AHP(下一秒 X) = {RISK_SCALE * weighted_sum(nx):.2f}")
    print("下一秒 X1–X9 預測：")
    for i, k in enumerate(INDEX_KEYS):
        t = f"{float(truth['next_x'][i]):.3f}" if "next_x" in truth else "-"
        print(f"  {k} {INDEX_NAMES[k]:<4} pred={nx[i]:.3f}  truth={t}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=str, required=True, help=".npz（雙通道）或 .npy (C,T,V,M)")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if args.sample.endswith(".npy"):
        infer_action_only(args.model or str(config.CHECKPOINT_DIR / "best_stgcn.pth"), args.sample, device)
    else:
        infer_intent(args.model or str(config.CHECKPOINT_DIR / "intent_model.pt"), args.sample, device)


if __name__ == "__main__":
    main()
