"""多鏡頭即時分析管線（研究計畫圖 2 情境應用步驟）：

  1. 目標鎖定：YOLO26 人物偵測 + ByteTrack（每支鏡頭各自一個 tracker）
  2. 跨鏡頭重識別：OSNet Re-ID 特徵 → 全域 person_id
  3. 骨架提取：MediaPipe Pose 33 點（轉為畫面像素座標）
  4. 武器偵測（X1）
  5. IntentEngine：X1–X9 → (ST-GCN + LSTM) → AHP + 表 2 規則 → L0–L4
  6. 告警：寫入事件庫、RAG 說明、WebSocket 廣播、推播
"""
import time
import threading
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchreid
from ultralytics import YOLO

from app.core.cv.pipelines.data_recorder import DataRecorder
from app.core.cv.simulator import is_sim_source
from app.core.cv.sources import open_source
from app.core.intent import registry
from app.core.intent.constants import LEVEL_BGR
from app.core.intent.engine import IntentEngine
from app.core.intent.features import PersonObservation, WeaponObservation
from app.core.intent.predictor import get_predictor
from app.core.intent.skeleton import crop_landmarks_to_frame
from app.core.paths import BACKEND_DIR, SNAPSHOT_DIR, WEIGHTS_DIR
from app.core.utils.pose_utils import (
    POSE_CONNECTIONS,
    extract_mediapipe_pose_full,
    get_pose_estimator,
    get_pose_task_path,
    pose_model_exists,
)

BASE_DIR = BACKEND_DIR
MODEL_PATH = WEIGHTS_DIR / "yolo26n.pt"
TRACKER_PATH = BASE_DIR / "app" / "scripts" / "custom_tracker.yaml"

CONF_DEFAULT = 0.45
IMG_SIZE = 640
MAX_GALLERY_PER_PERSON = 12
REID_MATCH_THRESHOLD = 0.72
REID_STRONG_THRESHOLD = 0.82
EMB_REFRESH_SEC = 0.5          # 已綁定 person_id 的 track，每 0.5 秒才重新抽一次 Re-ID 特徵
TRACK_CACHE_MAX_IDLE_SEC = 2.5
PERSON_DB_MAX_IDLE_SEC = 60.0
MIN_CROP_H = 120
MIN_CROP_W = 60
MIN_SHARPNESS = 35.0
MIN_POSE_CROP_H = 160
MIN_POSE_CROP_W = 80
CLIP_BUFFER_SEC = 8.0          # 事件回放保留告警前 8 秒
CLIP_FRAME_INTERVAL = 0.2
CLIP_WIDTH = 640

stop_events: Dict[str, bool] = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

reid_model = torchreid.models.build_model(
    name="osnet_x1_0",
    num_classes=1000,
    loss="softmax",
    pretrained=True
)
reid_model.to(device).eval()

reid_transforms = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_global_lock = threading.Lock()

person_db: Dict[int, Dict[str, Any]] = {}
track_to_person: Dict[Tuple[str, int], Dict[str, Any]] = {}
next_person_id = 1


def now_ts() -> float:
    return time.time()


def reset_global_state():
    global person_db, track_to_person, next_person_id
    with _global_lock:
        person_db = {}
        track_to_person = {}
        next_person_id = 1


def open_camera(source: str):
    try:
        idx = int(source)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    except ValueError:
        cap = cv2.VideoCapture(source)   # RTSP / 影片檔

    if not cap.isOpened():
        return None
    return cap


def save_snapshot(task_id: str, cam_name: str, frame: np.ndarray):
    path = SNAPSHOT_DIR / f"{task_id}_{cam_name}_latest.jpg"
    tmp_path = SNAPSHOT_DIR / f"{task_id}_{cam_name}_latest.tmp.jpg"

    ok = cv2.imwrite(str(tmp_path), frame)
    if ok:
        tmp_path.replace(path)


def l2_normalize(vec: np.ndarray, eps: float = 1e-12):
    norm = np.linalg.norm(vec)
    if norm < eps:
        return vec
    return vec / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))


def safe_crop(frame: np.ndarray, box: List[int]):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def expand_box(box: List[int], frame_shape, scale_x=0.18, scale_y=0.22):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(bw * scale_x)
    pad_y = int(bh * scale_y)

    nx1 = max(0, x1 - pad_x)
    ny1 = max(0, y1 - pad_y)
    nx2 = min(w - 1, x2 + pad_x)
    ny2 = min(h - 1, y2 + pad_y)

    return [nx1, ny1, nx2, ny2]


def crop_quality_ok(crop: Optional[np.ndarray]) -> bool:
    if crop is None or crop.size == 0:
        return False

    h, w = crop.shape[:2]
    if h < MIN_CROP_H or w < MIN_CROP_W:
        return False

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    if sharpness < MIN_SHARPNESS:
        return False

    return True


def extract_deep_embedding(person_crop: Optional[np.ndarray]):
    if person_crop is None or person_crop.size == 0:
        return None

    if not crop_quality_ok(person_crop):
        return None

    person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
    img_t = reid_transforms(person_crop_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = reid_model(img_t).squeeze().cpu().numpy()

    return l2_normalize(emb)


def create_new_person(emb: np.ndarray, camera_name: str) -> int:
    global next_person_id
    pid = next_person_id
    next_person_id += 1

    person_db[pid] = {
        "embeddings": deque([emb], maxlen=MAX_GALLERY_PER_PERSON),
        "last_seen": now_ts(),
        "last_camera": camera_name,
        "seen_cameras": {camera_name},
    }
    return pid


def match_existing_person(emb: np.ndarray) -> Tuple[Optional[int], float]:
    best_pid = None
    best_score = -1.0

    for pid, info in person_db.items():
        gallery = info.get("embeddings", [])
        if not gallery:
            continue

        score = max(cosine_similarity(emb, g) for g in gallery)
        if score > best_score:
            best_score = score
            best_pid = pid

    if best_pid is not None and best_score >= REID_MATCH_THRESHOLD:
        return best_pid, best_score

    return None, best_score


def _touch_person(pid: int, camera_name: str, now: float, emb: Optional[np.ndarray]) -> None:
    info = person_db.get(pid)
    if info is None:
        return
    info["last_seen"] = now
    info["last_camera"] = camera_name
    info["seen_cameras"].add(camera_name)
    if emb is not None:
        info["embeddings"].append(emb)


def resolve_person(camera_name: str, track_id: int, frame: np.ndarray, box: List[int]) -> Tuple[Optional[int], str, float]:
    """track_id → 全域 person_id。回傳 (person_id 或 None, 說明, Re-ID 相似度)。

    已綁定的 track 在 EMB_REFRESH_SEC 內直接沿用快取，不重抽特徵；
    裁切品質不佳時若已有快取也沿用，避免人物在時序分析中斷線。
    """
    key = (camera_name, int(track_id))
    now = now_ts()

    with _global_lock:
        cache = track_to_person.get(key)
        if cache is not None and cache["person_id"] in person_db and now - cache.get("emb_ts", 0) < EMB_REFRESH_SEC:
            cache["last_seen"] = now
            _touch_person(cache["person_id"], camera_name, now, None)
            return cache["person_id"], "track-cache", cache.get("score", 1.0)

    emb = extract_deep_embedding(safe_crop(frame, box))

    with _global_lock:
        cache = track_to_person.get(key)
        if cache is not None and cache["person_id"] in person_db:
            cache["last_seen"] = now
            if emb is not None:
                cache["emb_ts"] = now
            _touch_person(cache["person_id"], camera_name, now, emb)
            return cache["person_id"], "track-cache", cache.get("score", 1.0)

        if emb is None:
            return None, "low-quality", 0.0

        matched_pid, score = match_existing_person(emb)
        if matched_pid is not None:
            track_to_person[key] = {"person_id": matched_pid, "last_seen": now, "emb_ts": now, "score": score}
            _touch_person(matched_pid, camera_name, now, emb)
            tag = "reid-strong" if score >= REID_STRONG_THRESHOLD else f"reid:{score:.2f}"
            return matched_pid, tag, score

        pid = create_new_person(emb, camera_name)
        track_to_person[key] = {"person_id": pid, "last_seen": now, "emb_ts": now, "score": 1.0}
        return pid, "new", 1.0


def cleanup_stale_tracks():
    now = now_ts()

    with _global_lock:
        stale_track_keys = [k for k, info in track_to_person.items()
                            if now - info.get("last_seen", 0) > TRACK_CACHE_MAX_IDLE_SEC]
        for key in stale_track_keys:
            track_to_person.pop(key, None)

        stale_person_ids = [pid for pid, info in person_db.items()
                            if now - info.get("last_seen", 0) > PERSON_DB_MAX_IDLE_SEC]
        for pid in stale_person_ids:
            person_db.pop(pid, None)


# ---------------------------------------------------------------------------
# 繪圖
# ---------------------------------------------------------------------------

def draw_pose_px(frame: np.ndarray, lm_px: np.ndarray, color=(0, 255, 255)):
    h, w = frame.shape[:2]
    pts = [(int(np.clip(p[0], 0, w - 1)), int(np.clip(p[1], 0, h - 1))) for p in lm_px]
    for a, b in POSE_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for p in pts:
        cv2.circle(frame, p, 2, (255, 255, 0), -1)


def draw_person_state(frame: np.ndarray, state: Dict[str, Any], note: str):
    x1, y1, x2, y2 = state["bbox"]
    level = state["level"]
    color = LEVEL_BGR[level]
    thick = 2 if level in ("L0", "L1") else 3
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

    trail = state.get("trail") or []
    if len(trail) >= 2:
        cv2.polylines(frame, [np.array(trail, dtype=np.int32).reshape(-1, 1, 2)], False, color, 2)

    label_1 = f"P{state['person_id']} {level} R{state['score']:.1f}"
    tops = state.get("top_indices") or []
    label_2 = " ".join(f"{t['key']}:{t['value']:.2f}" for t in tops[:2]) or note
    if state.get("action"):
        label_2 = f"{label_2} {state['action']}"

    (tw, th), _ = cv2.getTextSize(label_1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    top = max(0, y1 - 42)
    cv2.rectangle(frame, (x1, top), (x1 + max(tw, 10) + 8, top + 22), color, -1)
    cv2.putText(frame, label_1, (x1 + 4, top + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(frame, label_2, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_weapons(frame: np.ndarray, weapons: List[WeaponObservation]):
    for w in weapons:
        x1, y1, x2, y2 = [int(v) for v in w.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(frame, f"{w.label} {w.conf:.2f}", (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)


# ---------------------------------------------------------------------------
# 單鏡頭單幀處理
# ---------------------------------------------------------------------------

def process_camera_frame(
    model: Optional[YOLO],
    frame: np.ndarray,
    camera_name: str,
    cam_index: int,
    engine: Optional[IntentEngine] = None,
    weapon_detector=None,
    recorder: Optional[DataRecorder] = None,
    frame_idx: int = 0,
    ts: Optional[float] = None,
    sim: Optional[Tuple[List[PersonObservation], List[WeaponObservation]]] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """sim 不為 None 時（模擬鏡頭）直接使用劇本給的人物與武器，略過 YOLO / Re-ID / MediaPipe。"""
    tracker_arg = str(TRACKER_PATH) if TRACKER_PATH.exists() else "bytetrack.yaml"
    ts = now_ts() if ts is None else ts
    fh, fw = frame.shape[:2]
    vis = frame.copy()

    weapons: List[WeaponObservation] = []
    observations: List[PersonObservation] = []
    notes: Dict[int, str] = {}
    record_items: List[Dict[str, Any]] = []

    if sim is not None:
        observations, weapons = list(sim[0]), list(sim[1])
        for obs in observations:
            notes[obs.person_id] = "sim"
            if obs.landmarks is not None:
                draw_pose_px(vis, obs.landmarks)
            record_items.append({
                "person_id": obs.person_id, "track_id": obs.track_id, "bbox": list(obs.bbox),
                "reid_score": 1.0, "det_conf": obs.det_conf, "landmarks": None, "pose_box": None,
                "visibility": None if obs.visibility is None else obs.visibility.round(3).tolist(),
                "landmarks_px": None if obs.landmarks is None else np.asarray(obs.landmarks).round(2).tolist(),
            })
        result = None
    else:
        result = model.track(
            frame,
            classes=[0],
            conf=CONF_DEFAULT,
            imgsz=IMG_SIZE,
            persist=True,
            tracker=tracker_arg,
            verbose=False,
        )[0]
        if weapon_detector is not None:
            try:
                weapons = weapon_detector.detect(camera_name, frame)
            except Exception as e:
                print(f"[WEAPON] 偵測失敗: {e}")

    if result is not None and result.boxes is not None and result.boxes.id is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.int().cpu().tolist()
        track_ids = result.boxes.id.int().cpu().tolist()
        confs = result.boxes.conf.cpu().tolist()

        seen_pids = set()
        for box, track_id, det_conf in zip(boxes, track_ids, confs):
            person_id, note, reid_score = resolve_person(camera_name, int(track_id), frame, box)
            if person_id is None or person_id in seen_pids:
                x1, y1, x2, y2 = box
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 180, 255), 1)
                cv2.putText(vis, "LOW-QUALITY" if person_id is None else "DUP", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)
                continue
            seen_pids.add(person_id)

            pose_box = expand_box(box, frame.shape, scale_x=0.18, scale_y=0.22)
            pose_crop = safe_crop(frame, pose_box)
            landmarks, visibility, lm_px = None, None, None
            if pose_crop is not None:
                ph, pw = pose_crop.shape[:2]
                if ph >= MIN_POSE_CROP_H and pw >= MIN_POSE_CROP_W:
                    landmarks, visibility = extract_mediapipe_pose_full(pose_crop)
            if landmarks:
                lm_px = crop_landmarks_to_frame(landmarks, pose_box)
                draw_pose_px(vis, lm_px)

            observations.append(PersonObservation(
                person_id=int(person_id),
                track_id=int(track_id),
                bbox=box,
                landmarks=lm_px,
                visibility=None if visibility is None else np.asarray(visibility, dtype=np.float32),
                det_conf=float(det_conf),
            ))
            notes[int(person_id)] = note
            record_items.append({
                "person_id": int(person_id), "track_id": int(track_id), "bbox": box,
                "reid_score": float(reid_score), "det_conf": float(det_conf),
                "landmarks": landmarks, "visibility": visibility, "pose_box": pose_box,
                "landmarks_px": None if lm_px is None else lm_px.round(2).tolist(),
            })

    states: Dict[int, Dict[str, Any]] = {}
    alerts: List[Dict[str, Any]] = []
    if engine is not None:
        try:
            states, alerts = engine.process(camera_name, ts, fw, fh, observations, weapons)
        except Exception as e:
            print(f"[INTENT] 分析失敗: {e!r}")

    for obs in observations:
        st = states.get(obs.person_id)
        if st is not None:
            draw_person_state(vis, st, notes.get(obs.person_id, ""))
        else:
            x1, y1, x2, y2 = obs.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (200, 200, 200), 2)
            cv2.putText(vis, f"P{obs.person_id} {notes.get(obs.person_id, '')}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)
    draw_weapons(vis, weapons)

    if recorder is not None:
        weapons_rec = [{"label": w.label, "conf": round(w.conf, 3), "bbox": list(w.bbox)} for w in weapons]
        for item in record_items:
            st = states.get(item["person_id"], {})
            recorder.record_frame(
                frame_idx=frame_idx,
                timestamp=ts,
                camera_id=camera_name,
                person_id=item["person_id"],
                track_id=item["track_id"],
                bbox=item["bbox"],
                reid_score=item["reid_score"],
                landmarks=item["landmarks"],
                landmarks_px=item["landmarks_px"],
                visibility=item["visibility"],
                pose_box=item["pose_box"],
                frame_size=[fw, fh],
                det_conf=item["det_conf"],
                weapons=weapons_rec,
                indices=st.get("indices"),
                level=st.get("level"),
                score=st.get("score"),
            )

    return vis, alerts


def _encode_clip_frame(vis: np.ndarray) -> Optional[bytes]:
    h, w = vis.shape[:2]
    if w > CLIP_WIDTH:
        vis = cv2.resize(vis, (CLIP_WIDTH, int(h * CLIP_WIDTH / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    return buf.tobytes() if ok else None


# ---------------------------------------------------------------------------
# 任務主迴圈
# ---------------------------------------------------------------------------

def run_detection_multi_reid(
    sources: List[str],
    conf: float = CONF_DEFAULT,
    task_id: str = "default",
):
    global CONF_DEFAULT
    CONF_DEFAULT = conf
    stop_events[task_id] = False

    if not MODEL_PATH.exists():
        print(f"❌ 找不到 YOLO 權重檔: {MODEL_PATH}")
        stop_events.pop(task_id, None)
        return

    if not pose_model_exists():
        print(f"⚠️ 找不到 MediaPipe Pose task 檔: {get_pose_task_path()}")
        print("⚠️ 系統仍可執行 YOLO + ReID，但不會有骨架相關指數 (X2/X7 與 ST-GCN)")

    reset_global_state()

    try:
        get_pose_estimator()
        print(f"✅ PoseLandmarker loaded: {get_pose_task_path()}")
    except Exception as e:
        print(f"⚠️ PoseLandmarker 載入失敗: {e}")

    real_sources = [not is_sim_source(s) for s in sources]
    try:
        # 每支真實鏡頭一個 YOLO 實例：ByteTrack 狀態綁在 predictor 上，共用會把不同鏡頭的 track 混在一起
        models: List[Optional[YOLO]] = [YOLO(str(MODEL_PATH)) if real else None for real in real_sources]
        print(f"✅ YOLO loaded for multi_reid ({sum(real_sources)} trackers, {len(sources) - sum(real_sources)} simulated)")
    except Exception as e:
        print(f"❌ YOLO 載入失敗: {e}")
        stop_events.pop(task_id, None)
        return

    weapon_detector = None
    if any(real_sources):
        try:
            from app.core.cv.weapons import WeaponDetector
            weapon_detector = WeaponDetector()
            print("✅ WeaponDetector ready (X1)")
        except Exception as e:
            print(f"⚠️ 武器偵測器載入失敗，X1 將維持 0: {e}")

    engine = IntentEngine(task_id=task_id, predictor=get_predictor())
    registry.register(engine)

    try:
        from app.services.event_service import event_service
    except Exception as e:
        event_service = None
        print(f"⚠️ 事件服務無法使用，告警只會顯示在畫面上: {e}")

    caps = []
    camera_names = []
    frame_indices = []

    for i, src in enumerate(sources):
        cam_name = f"cam{i}"
        cap = open_source(src, cam_index=i)
        if cap is None:
            print(f"❌ 無法開啟來源: {src} ({cam_name})")
            for opened in caps:
                opened.release()
            stop_events.pop(task_id, None)
            registry.mark_stopped(task_id)
            return
        caps.append(cap)
        camera_names.append(cam_name)
        frame_indices.append(0)

    recorder = None
    try:
        recorder = DataRecorder(base_dir=str(BASE_DIR), session_name=task_id)
        print("✅ DataRecorder initialized")
    except Exception as e:
        print(f"⚠️ DataRecorder 初始化失敗（不影響即時分析）: {e}")

    clip_buffers: Dict[str, Deque[Tuple[float, bytes]]] = {name: deque() for name in camera_names}
    last_tick = {name: now_ts() for name in camera_names}
    fps_ema = {name: 0.0 for name in camera_names}

    try:
        while not stop_events.get(task_id, False):
            frames = []
            ok_all = True

            for cap in caps:
                ret, frame, sim = cap.read()
                if not ret:
                    ok_all = False
                    break
                frames.append((frame, sim))

            if not ok_all:
                print("[multi_reid] 有鏡頭讀取失敗，任務結束")
                break

            for i, (frame, sim) in enumerate(frames):
                cam_name = camera_names[i]
                ts = now_ts()
                vis, alerts = process_camera_frame(
                    model=models[i],
                    frame=frame,
                    camera_name=cam_name,
                    cam_index=i,
                    engine=engine,
                    weapon_detector=weapon_detector,
                    recorder=recorder,
                    frame_idx=frame_indices[i],
                    ts=ts,
                    sim=sim,
                )

                now = now_ts()
                inst = 1.0 / max(now - last_tick[cam_name], 1e-6)
                last_tick[cam_name] = now
                fps_ema[cam_name] = inst if fps_ema[cam_name] == 0 else 0.9 * fps_ema[cam_name] + 0.1 * inst
                cv2.putText(vis, f"{cam_name.upper()} FPS:{fps_ema[cam_name]:.1f}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                buf = clip_buffers[cam_name]
                if not buf or ts - buf[-1][0] >= CLIP_FRAME_INTERVAL:
                    jpg = _encode_clip_frame(vis)
                    if jpg is not None:
                        buf.append((ts, jpg))
                while buf and ts - buf[0][0] > CLIP_BUFFER_SEC:
                    buf.popleft()

                if alerts and event_service is not None:
                    clip = [jpg for _, jpg in buf]
                    for alert in alerts:
                        event_service.submit(alert, frame, vis, clip)

                save_snapshot(task_id, cam_name, vis)
                frame_indices[i] += 1

            cleanup_stale_tracks()
            time.sleep(0.005)

    except Exception as e:
        print(f"⚠️ multi_reid 偵測過程中發生錯誤: {repr(e)}")
    finally:
        for cap in caps:
            try:
                cap.release()
            except Exception:
                pass

        stop_events.pop(task_id, None)
        registry.mark_stopped(task_id)

        if recorder is not None:
            try:
                recorder.finalize()
            except Exception as e:
                print(f"⚠️ recorder.finalize() 失敗: {e}")

        print(f"[multi_reid] 任務 {task_id} 已安全結束")
