"""影像來源：攝影機編號、RTSP/HTTP 串流、影片檔（循環＋即時速度播放）、模擬鏡頭 sim://。

read() 一律回傳 (ok, frame, sim)；sim 只有模擬鏡頭才有值 (persons, weapons)。
"""
import os
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.core.cv.simulator import SimulatedSource, is_sim_source, scenario_of
from app.core.intent.features import PersonObservation, WeaponObservation

SimObs = Tuple[List[PersonObservation], List[WeaponObservation]]
VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov", ".m4v", ".webm")


class FrameSource:
    is_simulated = False
    name = ""

    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[SimObs]]:
        raise NotImplementedError

    def release(self) -> None:
        pass


class CaptureSource(FrameSource):
    """攝影機或串流：直接讀取。"""

    def __init__(self, cap: cv2.VideoCapture, name: str):
        self.cap = cap
        self.name = name

    def read(self):
        ok, frame = self.cap.read()
        return ok, frame, None

    def release(self):
        self.cap.release()


class VideoFileSource(FrameSource):
    """影片檔：依影片 FPS 以真實速度播放（處理太慢時跳幀追上），播完自動從頭循環。

    讓速度、持續時間等以秒計的指數與真實鏡頭一致。
    """

    def __init__(self, cap: cv2.VideoCapture, name: str, loop: bool = True):
        self.cap = cap
        self.name = name
        self.loop = loop
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and np.isfinite(fps) and fps > 1 else 30.0
        self.t0: Optional[float] = None
        self.frame_no = 0

    def _rewind(self) -> bool:
        if not self.loop:
            return False
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.t0 = time.time()
        self.frame_no = 0
        return True

    def read(self):
        if self.t0 is None:
            self.t0 = time.time()
        target = int((time.time() - self.t0) * self.fps)
        # 處理速度跟不上影片時，丟掉落後的影格
        while self.frame_no < target - 1:
            if not self.cap.grab():
                if not self._rewind():
                    return False, None, None
                break
            self.frame_no += 1
        # 處理比影片快時，等到該影格的播放時間
        wait = self.t0 + self.frame_no / self.fps - time.time()
        if wait > 0:
            time.sleep(min(wait, 0.5))
        ok, frame = self.cap.read()
        if not ok:
            if not self._rewind():
                return False, None, None
            ok, frame = self.cap.read()
        self.frame_no += 1
        return ok, frame, None

    def release(self):
        self.cap.release()


class SimSource(FrameSource):
    is_simulated = True

    def __init__(self, scenario: str, cam_index: int):
        self.sim = SimulatedSource(scenario, cam_index)
        self.name = f"sim://{self.sim.name}"

    def read(self):
        ok, frame, persons, weapons = self.sim.read_sim()
        return ok, frame, (persons, weapons)


def open_source(source: str, cam_index: int = 0) -> Optional[FrameSource]:
    src = str(source).strip()
    if is_sim_source(src):
        try:
            return SimSource(scenario_of(src), cam_index)
        except ValueError as e:
            print(f"❌ {e}")
            return None
    try:
        idx = int(src)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(idx)
        return CaptureSource(cap, src) if cap.isOpened() else None
    except ValueError:
        pass
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return None
    if os.path.isfile(src) or src.lower().endswith(VIDEO_EXTS):
        return VideoFileSource(cap, src, loop=os.getenv("VIDEO_LOOP", "1") != "0")
    return CaptureSource(cap, src)
