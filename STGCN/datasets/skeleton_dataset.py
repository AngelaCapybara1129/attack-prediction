# datasets/skeleton_dataset.py
import csv
import os
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

# MediaPipe 33 點左右對稱關節，水平翻轉增強時需交換
_FLIP_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20),
               (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
FLIP_PERM = list(range(33))
for _a, _b in _FLIP_PAIRS:
    FLIP_PERM[_a], FLIP_PERM[_b] = _b, _a


def augment_skeleton(skel: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """skel: (C,T,V,M)。隨機水平翻轉、縮放、小角度旋轉、時間平移、雜訊；多人時隨機交換人物順序。

    全為 0 的人物（缺人）與幀（缺漏）保持為 0，不加雜訊。
    """
    out = skel.copy()
    present = np.abs(out).sum(axis=(0, 2)) > 0            # (T, M)
    if rng.random() < 0.5:
        out = out[:, :, FLIP_PERM, :]
        out[0] = -out[0]
    out[:2] *= rng.uniform(0.9, 1.1)
    theta = np.radians(rng.uniform(-10, 10))
    c, s = np.cos(theta), np.sin(theta)
    x, y = out[0].copy(), out[1].copy()
    out[0], out[1] = c * x - s * y, s * x + c * y
    out += rng.normal(0.0, 0.01, size=out.shape).astype(np.float32)
    shift = int(rng.integers(-out.shape[1] // 10, out.shape[1] // 10 + 1))
    if shift:
        out = np.roll(out, shift, axis=1)
        present = np.roll(present, shift, axis=0)
        if shift > 0:
            present[:shift] = False
        else:
            present[shift:] = False
    out *= present[None, :, None, :]
    if out.shape[3] > 1 and rng.random() < 0.5:
        out = out[..., ::-1]
    return np.ascontiguousarray(out, dtype=np.float32)


class SkeletonDataset(Dataset):
    """舊格式：samples/*.npy (C,T,V,M) + labels.csv(sample_id,label)，供單通道 ST-GCN 行為分類。"""

    def __init__(self, samples_dir, labels_csv, transform=None):
        self.samples_dir = samples_dir
        self.labels_csv = labels_csv
        self.transform = transform
        self.items = []

        if not os.path.exists(samples_dir):
            raise FileNotFoundError(f"Samples directory not found: {samples_dir}")

        if not os.path.exists(labels_csv):
            raise FileNotFoundError(f"Labels file not found: {labels_csv}")

        with open(labels_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample_id = row["sample_id"].strip()
                label = int(row["label"])
                sample_path = None
                for ext in (".npy", ".npz"):
                    p = os.path.join(samples_dir, f"{sample_id}{ext}")
                    if os.path.exists(p):
                        sample_path = p
                        break

                if sample_path:
                    self.items.append({"sample_id": sample_id, "label": label, "sample_path": sample_path})
                else:
                    print(f"[Warning] Sample file not found, skipped: {sample_id}")

        if len(self.items) == 0:
            raise ValueError(f"No valid samples found in {samples_dir} with {labels_csv}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        if item["sample_path"].endswith(".npz"):
            with np.load(item["sample_path"]) as z:
                data = z["skel"].astype(np.float32)
        else:
            data = np.load(item["sample_path"]).astype(np.float32)  # expected: (C, T, V, M)

        if data.ndim != 4:
            raise ValueError(
                f"Sample {item['sample_id']} shape must be 4D (C, T, V, M), got {data.shape}"
            )

        if self.transform is not None:
            data = self.transform(data)

        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(item["label"], dtype=torch.long)

        return data, label


class IntentWindowDataset(Dataset):
    """新格式（scripts/preprocess.py 產生）：samples/*.npz，內含
        skel (3,T,33,1)、xseq (T,9)、cur_x (9)、next_x (9)、risk、level、action、cur_level、cur_risk
    """

    def __init__(self, split_dir: str, augment: bool = False, seed: int = 0):
        self.split_dir = split_dir
        self.samples_dir = os.path.join(split_dir, "samples")
        labels_csv = os.path.join(split_dir, "labels.csv")
        if not os.path.exists(labels_csv):
            raise FileNotFoundError(f"Labels file not found: {labels_csv}（請先執行 scripts/preprocess.py）")
        self.items = []
        with open(labels_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                p = os.path.join(self.samples_dir, f"{row['sample_id'].strip()}.npz")
                if os.path.exists(p):
                    self.items.append({"sample_id": row["sample_id"].strip(), "path": p,
                                       "action": int(row["label"]), "level": int(row["level"])})
        if not self.items:
            raise ValueError(f"No valid samples in {split_dir}")
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def label_counts(self, key: str, num_classes: int) -> np.ndarray:
        counts = np.zeros(num_classes, dtype=np.int64)
        for it in self.items:
            if 0 <= it[key] < num_classes:
                counts[it[key]] += 1
        return counts

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        with np.load(item["path"]) as z:
            skel = z["skel"].astype(np.float32)
            xseq = z["xseq"].astype(np.float32)
            out = {
                "next_x": torch.from_numpy(z["next_x"].astype(np.float32)),
                "cur_x": torch.from_numpy(z["cur_x"].astype(np.float32)),
                "risk": torch.tensor(float(z["risk"]), dtype=torch.float32),
                "cur_risk": torch.tensor(float(z["cur_risk"]), dtype=torch.float32),
                "level": torch.tensor(int(z["level"]), dtype=torch.long),
                "cur_level": torch.tensor(int(z["cur_level"]), dtype=torch.long),
                "action": torch.tensor(int(z["action"]), dtype=torch.long),
            }
        if self.augment:
            skel = augment_skeleton(skel, self.rng)
        out["skel"] = torch.from_numpy(np.ascontiguousarray(skel))
        out["xseq"] = torch.from_numpy(np.ascontiguousarray(xseq))
        return out


def class_weights(counts: np.ndarray) -> Optional[torch.Tensor]:
    if counts.sum() == 0:
        return None
    w = np.where(counts > 0, counts.sum() / (len(counts) * np.maximum(counts, 1)), 0.0)
    return torch.tensor(w, dtype=torch.float32)
