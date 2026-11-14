"""模擬鏡頭各情境應觸發預期的風險等級（以模擬時間快速重播，不需等待）。"""
import numpy as np
import pytest

from app.core.cv.simulator import FPS, SCENES, bbox_of, make_skeleton
from app.core.intent.constants import LEVEL_INDEX
from app.core.intent.context import SpatialContext
from app.core.intent.engine import IntentEngine
from app.core.intent.features import PersonObservation, WeaponObservation

NEUTRAL = SpatialContext({"default_region_weight": 0.0})


def replay(scene: str):
    duration, fn, _ = SCENES[scene]
    engine = IntentEngine("test", predictor=None, context=NEUTRAL)
    peak = {}
    rules = set()
    for i in range(int(duration * FPS)):
        t = i / FPS
        persons, weapons = [], []
        for a in fn(t):
            pts = make_skeleton(a, t)
            box = bbox_of(pts, a.v)
            persons.append(PersonObservation(a.actor, a.actor, box, pts, np.ones(33, np.float32)))
            if a.arm == "knife":
                w = 16 if a.target_dir > 0 else 15
                cx, cy = pts[w, 0], pts[w, 1]
                weapons.append(WeaponObservation("knife", 0.86, [int(cx - 10), int(cy - 5), int(cx + 10), int(cy + 5)]))
        states, _ = engine.process("cam0", 1000.0 + t, 1280, 720, persons, weapons)
        for pid, st in states.items():
            if LEVEL_INDEX[st["level"]] > LEVEL_INDEX[peak.get(pid, "L0")]:
                peak[pid] = st["level"]
            rules.update(tr["rule"] for tr in st["triggers"])
    return peak, rules


@pytest.mark.parametrize("scene, expected_level, expected_rule", [
    ("calm", "L0", None),
    ("follow", "L2", "L2_follow"),
    ("fight", "L3", "L3_punch_intrusion"),
    ("weapon", "L4", "L4_weapon"),
    ("group", "L2", "L2_group"),
    ("loiter", "L1", "L1_loiter"),
])
def test_scene_reaches_expected_level(scene, expected_level, expected_rule):
    peak, rules = replay(scene)
    top = max(peak.values(), key=lambda lv: LEVEL_INDEX[lv]) if peak else "L0"
    print(scene, peak, sorted(rules))
    assert top == expected_level
    if expected_rule:
        assert expected_rule in rules
