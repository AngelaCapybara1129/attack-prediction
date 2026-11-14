"""攻擊意圖核心單元測試：cd backend && python -m pytest tests -q"""
import datetime as dt

import numpy as np
import pytest

from app.core.intent.ahp import AHP_WEIGHTS, CONSISTENCY_RATIO
from app.core.intent.constants import INDEX_KEYS, score_to_level
from app.core.intent.context import SpatialContext
from app.core.intent.features import FeatureEngine, PersonObservation, WeaponObservation
from app.core.intent.risk import RiskEngine
from app.core.intent.skeleton import (
    NUM_JOINTS, build_spatial_adjacency, normalize_skeleton_sequence, resample_by_time,
)

NEUTRAL = SpatialContext({"default_region_weight": 0.0})


def _box(u: float, h: float = 340.0, foot: float = 700.0):
    return [int(u - h * 0.2), int(foot - h), int(u + h * 0.2), int(foot)]


# --------------------------------------------------------------------------- AHP / 等級
def test_ahp_weights_consistent():
    w = AHP_WEIGHTS
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert max(w, key=w.get) == "X1"
    assert w["X4"] <= 0.06 and w["X9"] <= 0.06
    assert CONSISTENCY_RATIO < 0.1


def test_score_to_level_thresholds():
    assert score_to_level(0.5) == "L0"
    assert score_to_level(1.0) == "L1"
    assert score_to_level(2.5) == "L2"
    assert score_to_level(3.5) == "L3"
    assert score_to_level(7.0) == "L4"


# --------------------------------------------------------------------------- 骨架
def test_adjacency_is_column_normalized():
    a = build_spatial_adjacency()
    assert a.shape == (3, NUM_JOINTS, NUM_JOINTS)
    np.testing.assert_allclose(a.sum(axis=0).sum(axis=0), np.ones(NUM_JOINTS), atol=1e-5)


def test_resample_and_normalize_shapes():
    ts = np.linspace(0, 5, 40)
    frames = np.random.rand(40, NUM_JOINTS, 3).astype(np.float32) * 100
    valid = np.ones(40, bool)
    valid[10:15] = False
    seq, mask = resample_by_time(ts, frames, valid, t_end=5.0, window_sec=5.0, num_samples=150)
    assert seq.shape == (150, NUM_JOINTS, 3) and mask.shape == (150,)
    out = normalize_skeleton_sequence(seq, mask > 0)
    assert out.shape == (3, 150, NUM_JOINTS, 1)
    assert np.isfinite(out).all()


# --------------------------------------------------------------------------- 風險規則
def test_weapon_rule_requires_three_seconds():
    eng = RiskEngine()
    x = np.zeros(9, np.float32)
    x[0] = 0.9
    levels = []
    for i in range(36):                  # 10 fps × 3.6 s
        res = eng.evaluate(1, i * 0.1, x)
        levels.append((round(i * 0.1, 1), res.level, res.rule_level))
    assert all(rl != "L4" for t, _, rl in levels if t < 2.9)
    assert levels[-1][1] == "L4" and levels[-1][2] == "L4"


def test_level_hold_down_hysteresis():
    eng = RiskEngine()
    hot = np.zeros(9, np.float32)
    hot[1] = hot[5] = 0.9                # X2 + X6 → L3
    for i in range(10):
        eng.evaluate(7, i * 0.1, hot)
    assert eng.evaluate(7, 1.0, hot).level == "L3"
    cold = np.zeros(9, np.float32)
    res = eng.evaluate(7, 1.5, cold)
    assert res.level == "L3"             # 不會馬上降級
    for i in range(1, 60):
        res = eng.evaluate(7, 1.5 + i * 0.25, cold)
    assert res.level == "L0"


# --------------------------------------------------------------------------- X1–X9
def test_x6_sudden_intrusion_flags_aggressor():
    fe = FeatureEngine("cam0", NEUTRAL)
    fps = 20
    best_b, best_a = 0.0, 0.0
    for i in range(int(2.0 * fps)):
        t = i / fps
        u_b = max(560.0, 1000.0 - 340.0 * t)          # B 以約 1.7 m/s 衝向 A
        res = fe.update(t, 1280, 720, [
            PersonObservation(1, 1, _box(500.0)),
            PersonObservation(2, 2, _box(u_b)),
        ])
        best_a = max(best_a, float(res[1].x[5]))
        best_b = max(best_b, float(res[2].x[5]))
    assert best_b > 0.5
    assert best_b > best_a


def test_x5_following_behind_target():
    fe = FeatureEngine("cam0", NEUTRAL)
    fps = 10
    res = {}
    for i in range(int(12 * fps)):
        t = i / fps
        u_a = 900.0 + 200.0 * t                        # 1 m/s（0.005 m/px）
        u_b = u_a - 300.0                              # 落後 1.5 m
        res = fe.update(t, 3600, 720, [
            PersonObservation(1, 1, _box(u_a)),
            PersonObservation(2, 2, _box(u_b)),
        ])
    assert res[2].x[4] >= 0.8                          # 尾隨者
    assert res[1].x[4] < 0.2                           # 被尾隨者


def test_x1_weapon_assigned_to_nearest_person():
    fe = FeatureEngine("cam0", NEUTRAL)
    knife = WeaponObservation("knife", 0.85, [560, 500, 600, 540])
    res = fe.update(0.0, 1280, 720, [
        PersonObservation(1, 1, _box(580.0)),
        PersonObservation(2, 2, _box(1000.0)),
    ], [knife])
    assert res[1].x[0] > 0.8
    assert res[2].x[0] == 0.0


def test_x9_context_zone_and_time():
    ctx = SpatialContext({
        "cameras": {"cam0": {"region_weight": 0.2, "zones": [
            {"name": "出入口", "weight": 1.0, "polygon": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]]}]}},
        "time_weights": [{"start": 0, "end": 24, "weight": 0.8}],
    })
    inside, _ = ctx.score("cam0", 0.0, (0.25, 0.5))
    outside, _ = ctx.score("cam0", 0.0, (0.75, 0.5))
    assert inside == pytest.approx(0.8)
    assert outside == pytest.approx(0.16)


# --------------------------------------------------------------------------- RAG
def test_query_parser():
    from app.core.rag.query_parser import parse_query

    now = dt.datetime(2026, 11, 14, 15, 0)            # 週六
    p = parse_query("上週穿黑衣揮拳 cam1", now)
    assert "X2" in p["indices"]
    assert "black" in p["colors"]
    assert p["cameras"] == ["cam1"]
    assert p["since"] == dt.datetime(2026, 11, 2) and p["until"] == dt.datetime(2026, 11, 9)


def test_local_vector_store(tmp_path):
    from app.core.rag.vector_store import LocalVectorStore

    store = LocalVectorStore(dim=3, path=str(tmp_path / "v.json"))
    store.upsert("a", np.array([1, 0, 0]), {"kind": "event"})
    store.upsert("b", np.array([0, 1, 0]), {"kind": "case"})
    hits = store.search(np.array([0.9, 0.1, 0]), k=2)
    assert hits[0][0] == "a"
    assert store.search(np.array([1, 0, 0]), k=1, kind="case")[0][0] == "b"
    reloaded = LocalVectorStore(dim=3, path=str(tmp_path / "v.json"))
    assert reloaded.count() == 2


# --------------------------------------------------------------------------- 模型
def test_dual_channel_forward_shapes():
    torch = pytest.importorskip("torch")
    from app.core.intent.nets import STGCN, build_model

    model = build_model({}).eval()
    skel = torch.randn(2, 3, 150, NUM_JOINTS, 1)
    xseq = torch.rand(2, 150, len(INDEX_KEYS))
    out = model(skel, xseq)
    assert out["action_logits"].shape == (2, 3)
    assert out["next_x"].shape == (2, 9)
    assert out["level_logits"].shape == (2, 5)
    assert float(out["risk"].min()) >= 0.0 and float(out["risk"].max()) <= 10.0
    assert STGCN()(skel).shape == (2, 3)
