"""ST-GCN（Yan et al., 2018）與 ST-GCN + LSTM 雙通道攻擊意圖預測模型。

- STGCN：骨架行為類型辨識（公式 (1)~(3)：空間圖卷積 → 時間卷積 → 全域池化 + Softmax）
- DualChannelIntentNet：
    通道 A：ST-GCN 輸出的時序高維動作表徵（只在關節維度池化，保留時間軸）
    通道 B：由骨架/軌跡推導的 X1–X9 可解釋行為指數序列
    兩者在時間軸對齊後串接 → LSTM → 多任務輸出：
        action_logits（行為類型）、next_x（下一時間段 X1–X9）、risk（0~10 連續風險分數）、level_logits（L0–L4）
"""
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.core.intent.constants import ACTION_CLASSES, LEVELS, NUM_INDICES, RISK_SCALE
from app.core.intent.skeleton import NUM_JOINTS, build_spatial_adjacency


class GraphConv(nn.Module):
    """空間圖卷積：f_out = Σ_k W_k · f_in · (A_k ⊙ M_k)。"""

    def __init__(self, in_channels: int, out_channels: int, num_partitions: int):
        super().__init__()
        self.k = num_partitions
        self.conv = nn.Conv2d(in_channels, out_channels * num_partitions, kernel_size=1)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        n, _, t, v = x.shape
        x = self.conv(x)
        x = x.view(n, self.k, -1, t, v)
        return torch.einsum("nkctv,kvw->nctw", x, a).contiguous()


class STGCNUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, a_shape: Sequence[int],
                 stride: int = 1, dropout: float = 0.0, residual: bool = True, kernel_t: int = 9):
        super().__init__()
        pad = (kernel_t - 1) // 2
        self.gcn = GraphConv(in_channels, out_channels, a_shape[0])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (kernel_t, 1), (stride, 1), (pad, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        if not residual:
            self.residual = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        # 可學習邊重要性 M
        self.edge_importance = nn.Parameter(torch.ones(*a_shape))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        res = 0 if self.residual is None else self.residual(x)
        x = self.gcn(x, a * self.edge_importance)
        x = self.tcn(x) + res
        return self.relu(x)


class STGCNBackbone(nn.Module):
    """輸入 (N, C, T, V, M) → 輸出 (N, 256, T/4, V)，多人時對 M 取平均。"""

    out_channels = 256

    def __init__(self, in_channels: int = 3, num_point: int = NUM_JOINTS, num_person: int = 1, dropout: float = 0.2):
        super().__init__()
        a = torch.tensor(build_spatial_adjacency(num_point), dtype=torch.float32)
        self.register_buffer("A", a)
        self.num_point = num_point
        self.num_person = num_person
        self.in_channels = in_channels
        self.data_bn = nn.BatchNorm1d(in_channels * num_point * num_person)
        cfg = [
            (in_channels, 64, 1, False),
            (64, 64, 1, True),
            (64, 64, 1, True),
            (64, 128, 2, True),
            (128, 128, 1, True),
            (128, 256, 2, True),
            (256, 256, 1, True),
        ]
        self.units = nn.ModuleList(
            STGCNUnit(cin, cout, a.shape, stride=s, dropout=0.0 if i == 0 else dropout, residual=r)
            for i, (cin, cout, s, r) in enumerate(cfg)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"輸入必須為 (N, C, T, V, M)，收到 {tuple(x.shape)}")
        n, c, t, v, m = x.shape
        if c != self.in_channels or v != self.num_point or m != self.num_person:
            raise ValueError(f"輸入維度不符：C={c}/{self.in_channels} V={v}/{self.num_point} M={m}/{self.num_person}")
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(n, m * v * c, t)
        x = self.data_bn(x)
        x = x.view(n, m, v, c, t).permute(0, 1, 3, 4, 2).contiguous().view(n * m, c, t, v)
        for unit in self.units:
            x = unit(x, self.A)
        _, cc, tt, vv = x.shape
        return x.view(n, m, cc, tt, vv).mean(dim=1)


class STGCN(nn.Module):
    """單通道 ST-GCN 行為分類（與舊版 SimpleSTGCN 介面相容）。"""

    def __init__(self, in_channels: int = 3, num_class: int = len(ACTION_CLASSES), num_point: int = NUM_JOINTS,
                 num_person: int = 1, dropout: float = 0.2):
        super().__init__()
        self.backbone = STGCNBackbone(in_channels, num_point, num_person, dropout)
        self.fc = nn.Linear(STGCNBackbone.out_channels, num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        pooled = feat.mean(dim=(2, 3))
        return self.fc(pooled)


class DualChannelIntentNet(nn.Module):
    def __init__(
        self,
        num_action: int = len(ACTION_CLASSES),
        num_index: int = NUM_INDICES,
        num_level: int = len(LEVELS),
        in_channels: int = 3,
        num_point: int = NUM_JOINTS,
        num_person: int = 1,
        x_embed: int = 32,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone = STGCNBackbone(in_channels, num_point, num_person, dropout)
        feat_dim = STGCNBackbone.out_channels
        self.action_head = nn.Linear(feat_dim, num_action)
        self.x_proj = nn.Sequential(nn.Linear(num_index, x_embed), nn.ReLU(inplace=True))
        self.lstm = nn.LSTM(
            input_size=feat_dim + x_embed,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.next_x_head = nn.Linear(lstm_hidden, num_index)
        self.risk_head = nn.Linear(lstm_hidden, 1)
        self.level_head = nn.Linear(lstm_hidden, num_level)

    def forward(self, skel: torch.Tensor, xseq: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.backbone(skel)                      # (N, 256, T', V)
        action_logits = self.action_head(feat.mean(dim=(2, 3)))
        seq = feat.mean(dim=3).transpose(1, 2)          # (N, T', 256)
        t_out = seq.shape[1]
        xs = F.adaptive_avg_pool1d(xseq.transpose(1, 2), t_out).transpose(1, 2)   # (N, T', 9)
        fused = torch.cat([seq, self.x_proj(xs)], dim=2)
        out, _ = self.lstm(fused)
        h = self.dropout(out[:, -1])
        return {
            "action_logits": action_logits,
            "next_x": torch.sigmoid(self.next_x_head(h)),
            "risk": RISK_SCALE * torch.sigmoid(self.risk_head(h)).squeeze(-1),
            "level_logits": self.level_head(h),
            "embedding": h,
        }


def build_model(config: Optional[Dict] = None) -> DualChannelIntentNet:
    cfg = dict(config or {})
    return DualChannelIntentNet(
        num_action=len(cfg.get("action_classes", ACTION_CLASSES)),
        num_index=cfg.get("num_index", NUM_INDICES),
        num_level=cfg.get("num_level", len(LEVELS)),
        in_channels=cfg.get("in_channels", 3),
        num_point=cfg.get("num_point", NUM_JOINTS),
        num_person=cfg.get("num_person", 1),
        x_embed=cfg.get("x_embed", 32),
        lstm_hidden=cfg.get("lstm_hidden", 128),
        lstm_layers=cfg.get("lstm_layers", 2),
        dropout=cfg.get("dropout", 0.2),
    )


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(arr), dtype=torch.float32, device=device)
