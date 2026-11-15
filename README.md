# Attack Prediction 全專案說明

## 實作對照（研究計畫 → 程式碼）

| 研究計畫 | 實作位置 |
|---|---|
| YOLO26 + ByteTrack 目標鎖定（每鏡頭獨立 tracker） | `backend/app/core/cv/pipelines/multi_reid.py` |
| 跨鏡頭 Re-ID（OSNet，全域 person_id） | 同上 `resolve_person()` |
| MediaPipe Pose 33 點骨架（轉畫面像素座標） | `backend/app/core/utils/pose_utils.py`、`core/intent/skeleton.py` |
| 表 1：X1–X9 攻擊意圖指數 | `backend/app/core/intent/features.py`（X9 區域/時段設定：`app/config/spatial_context.json`） |
| X1 武器偵測 | `backend/app/core/cv/weapons.py`（預設 COCO knife/scissors/bat，可換自訂權重 `WEAPON_MODEL_PATH`） |
| 公式 (4) AHP 權重 + 一致性比率 | `backend/app/core/intent/ahp.py` |
| 表 2：L0–L4 規則（含持續時間）與分數門檻 | `backend/app/core/intent/risk.py` |
| ST-GCN（空間圖卷積 + 時間卷積 + 全域池化） | `backend/app/core/intent/nets.py` → `STGCN` |
| ST-GCN + LSTM 雙通道、多任務（行為/下一秒 X1–X9/風險分數/L0–L4） | 同上 `DualChannelIntentNet` |
| 即時融合與告警 | `backend/app/core/intent/engine.py` |
| RAG：向量庫（Milvus 或本機）、相似事件、說明與處置建議、文字檢索 | `backend/app/core/rag/` |
| PostgreSQL/SQLite 事件庫（X1–X9、L0–L4、軌跡、處置狀態、關鍵片段） | `backend/app/models/event.py` |
| REST / WebSocket | `backend/app/routers/events.py`、`intent.py`、`ws.py` |
| 推播（L3 以上） | `backend/app/services/notifier.py`（`NOTIFY_WEBHOOK_URL`） |
| 戰情儀表板、告警卡、事件回放、人物風險、文字檢索 | `frontend/attack-frontend/src/pages/Dashboard.tsx`、`EventsPage.tsx`、`components/` |
| 資料集處理、訓練、評估（Top-1/Macro-F1/Recall、MAE/RMSE） | `STGCN/scripts/` |

> 總風險分數 R = 10 × Σ wᵢXᵢ（0–10）。AHP 權重和為 1、Xᵢ ∈ [0,1]，乘以 10 才能對應表 2 的 1.0/2.0/3.5/5.0 門檻。
> 最終等級取「分數等級」與「表 2 規則觸發等級」較高者，降級需持續 4 秒以避免閃爍。

## 執行步驟

```powershell
# 後端
cd backend
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git   # torchreid
copy .env.example .env   # 第一次：填 SECRET_KEY 等
alembic upgrade head
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
python -m pytest tests -q   # 單元測試

# 前端
cd frontend/attack-frontend
pnpm install; pnpm dev        # http://localhost:5173
```

### 模擬鏡頭（沒有攝影機也能測試）

鏡頭清單預設會出現「模擬鏡頭（demo）」，選它並按「開始監測」即可。畫面中的人物會依劇本做出攻擊前兆：

| 來源 | 情境 | 預期 |
|---|---|---|
| `sim://demo` | 路人 → 尾隨 → 衝刺揮拳 → 持刀逼近 → 群聚（約 82 秒循環） | 依序 L0 → L2 → L3 → L4 → L2 |
| `sim://calm` / `follow` / `fight` / `weapon` / `group` / `loiter` | 單一情境 | L0 / L2 / L3 / L4 / L2 / L1 |

- 設定 `SIM_CAMERAS=sim://demo,sim://fight` 可同時列出多個模擬鏡頭（多鏡頭模式會一起跑）；設為空字串則關閉。
- 模擬鏡頭直接產生骨架，**略過 YOLO / Re-ID / MediaPipe**，用來驗證 X1–X9 → L0–L4 → 告警 → RAG → 前端。
- 測試偵測端請用真實影片：`EXTRA_SOURCES=<影片路徑>`，影片會以實際速度循環播放。
- `tests/test_simulator.py` 會快速重播每個情境，確認都達到預期等級。

### Docker（GPU）

在 `attack-prediction` 資料夾執行。啟動後前端在 http://localhost:3000，後端 API 在 http://localhost:8000/docs。

| 要做的事 | 指令 |
|---|---|
| 啟動 | `docker compose up -d` |
| 停止 | `docker compose down` |
| 修改程式後重新建置 | `docker compose up -d --build` |
| 查看後端紀錄 | `docker compose logs -f backend` |
| 只改了 Python 程式（不重建映像） | `docker compose restart backend` |

- 第一次啟動前要先建置映像（`docker compose up -d --build`），後端映像約 13 GB，第一次建置需要一段時間。
- 後端容器直接讀取 `backend` 資料夾，但不會自動重新載入；改了 Python 程式後要 `docker compose restart backend` 才會生效。改了 `requirements.txt` 或前端程式則要 `--build` 重新建置。
- Docker 與本機執行（`start.ps1`）都使用 port 8000，**兩者不能同時執行**：切換前先 `docker compose down` 或關掉本機的後端／前端視窗。
- 後端映像使用 CUDA 12.6 版 PyTorch，需 Docker Desktop（WSL2）+ NVIDIA 驅動；沒有 NVIDIA GPU 時刪除 `docker-compose.yml` 中的 `deploy:` 區塊。
- 預設沿用 `backend/app.db`（SQLite，保留既有帳號）。改用 PostgreSQL：取消 `DATABASE_URL` 註解後 `docker compose --profile postgres up -d`。
- **Windows 版 Docker 無法讀取 USB 攝影機**。容器內請用 RTSP 或影片：把影片放到 `backend/data/videos/`，在 `backend/.env` 設 `EXTRA_SOURCES=/app/data/videos/xxx.mp4`。要用實體攝影機展示時，請改用上面的本機執行方式。

沒有訓練好的模型時，系統以「X1–X9 規則 + AHP」基線運作；訓練完成後放上 checkpoint，系統會自動切換成「規則 + ST-GCN/LSTM 融合」。

### 訓練雙通道模型（攻擊類型分類 + 意圖預測）

模型同時輸出：攻擊類型、下一秒 X1–X9、風險分數、L0–L4。攻擊類型有兩種分類體系（`preprocess.py --taxonomy`）：

| taxonomy | 類別 | 適用資料集 |
|---|---|---|
| `coarse`（預設） | normal 正常 / fight 肢體攻擊 / weapon 持械攻擊 | RWF-2000、Hockey Fights、Real Life Violence Situations、SCVD |
| `fine` | normal / punch 揮拳 / kick 踢擊 / push 推撞 / strike_object 持物毆打 / knife 持刀 / gun 持槍 | NTU RGB+D 120 雙人互動動作、自錄影片 |

資料放在 `STGCN/data/raw/<資料集>/`，標籤由路徑自動判斷（`STGCN/config.py`）：

| 資料集 | 取得方式 | 放置結構 → 標籤 |
|---|---|---|
| RWF-2000 | 向作者簽署使用協議（GitHub: mchengny/RWF2000-Video-Database-for-Violence-Detection） | `rwf2000/train/Fight`、`train/NonFight`、`val/...` → fight / normal（val 再對半分成 val/test） |
| Hockey Fights | 公開下載 | 檔名 `fi*.avi` / `no*.avi` → fight / normal |
| Real Life Violence Situations | Kaggle | `Violence/`、`NonViolence/` → fight / normal |
| SCVD | Kaggle | `Normal/`、`Violence/`、`Weaponized/` → normal / fight / weapon |
| NTU RGB+D 120 | 向 ROSE Lab 申請 | 檔名動作代碼 A050 揮拳、A051 踢、A052/A108 推撞、A106 持物毆打、A107 持刀、A110 持槍，其餘雙人互動 → normal |
| 自錄影片 | — | 資料夾名稱直接用類別：`punch/`、`kick/`、`knife/`、`normal/`… |

```powershell
cd STGCN
$env:PYTHONIOENCODING="utf-8"
# 1) 影片 → 骨架 session（YOLO26 + ByteTrack + MediaPipe + 武器偵測，GPU）
#    影片多時可開多個終端機平行：--shard 0/3、--shard 1/3、--shard 2/3
python scripts/extract_skeletons.py --input data/raw/rwf2000 --dataset rwf2000
# 2) 建立樣本：RWF-2000 每段只有 5 秒，需縮短視窗；Hockey 只有約 1.5 秒，用 --window-sec 1 --horizon-sec 0 --seq-len 30
python scripts/preprocess.py --clean --taxonomy coarse --window-sec 3 --horizon-sec 1 --seq-len 90
# 3) 訓練、評估（視窗層級 + 影片層級 Top-1 / Macro-F1 / Recall、MAE / RMSE）、單筆推論
python scripts/train.py --epochs 80
python scripts/evaluate.py --split test
python scripts/infer.py --sample data/processed/test/samples/<id>.npz
```

- 攻擊影片中的旁觀者：`preprocess.py` 預設會排除「身邊 1.2 個身高內沒有其他人、也沒有攻擊相關指數」的視窗（`--attack-filter none` 可關閉）。
- 沒有標籤的後端錄影（`backend/data/sessions`）只用來訓練指數／風險預測，不計入攻擊類型損失。
- **沒有資料集時的流程測試**：`python scripts/make_synthetic_sessions.py --clean` 產生模擬骨架，再以 `--sessions data/synthetic/sessions --output data/processed_synthetic` 前處理並訓練。結果只代表程式能正確執行，不能當作模型成效。

產生的 `STGCN/checkpoints/intent_model.pt` 會在下一次開始監測時自動載入（或呼叫 `POST /api/intent/model/reload`），攻擊類型會顯示在告警與人物面板上。

### 主要 API

```text
POST  /yolo/start | /yolo/start/multi | /yolo/upload     開始分析（攝影機 / RTSP / 上傳影片）
GET   /api/intent/live/{task_id}                         畫面中每位人物的 X1–X9、分數、等級
GET   /api/intent/config                                 表 1、表 2、AHP 權重與 CR
GET   /api/events | /api/events/stats | /api/events/{id}
PATCH /api/events/{id}                                   處置狀態（未處理/已處理/需升級/誤報）與紀錄
GET   /api/events/{id}/similar | /snapshot | /clip/{n}
POST  /api/search/text                                   「上週穿黑衣揮拳」文字檢索
WS    /ws/alerts?token=<JWT>                             即時告警
```

---

這份文件整理了攻擊意圖預測專案的整體架構、技術棧、目錄規劃、部署方式與 ST-GCN 模組定位，適合作為專案 README 或系統設計總覽文件。[1][2]

## 專案概述

本專案目標是建立一套結合即時影像分析、骨架時序建模、風險分級與檢索增強的攻擊意圖預測系統，核心 AI 流程包含 YOLO26、ByteTrack、MediaPipe Pose、ST-GCN、LSTM、RAG 與 Milvus。[1][2]
系統同時支援 Desktop、Tablet 與 Mobile 使用情境，並透過 WebSocket 提供約 1 秒級的即時告警更新，特別針對 L3 與 L4 高風險事件做快速呈現。[1]

## 技術架構

前端採用 React 18、Vite、TypeScript、Tailwind 與 React Query，後端採用 FastAPI 0.104、SQLAlchemy 與 Pydantic，形成典型的前後端分離架構。[1]
AI 與資料層整合了 YOLO26、ByteTrack、MediaPipe、ST-GCN、LSTM、PostgreSQL、Milvus、Redis、MinIO、Docker Compose、Nginx 與 Gunicorn/Uvicorn，以支援即時推論、資料查詢與部署維運需求。[1][2]

## 核心流程

系統從 RTSP 或監視器影像取得畫面後，先進行人員偵測與追蹤，再透過 MediaPipe Pose 擷取骨架序列，接著將骨架時序送入 ST-GCN 與 LSTM 進行危險行為辨識與攻擊意圖預測。[2][1]
推論結果會進一步結合 AHP 權重與 L0-L4 風險分級規則，並透過 RAG 與 Milvus 查詢相似事件，讓值勤人員能在告警出現時同時取得上下文資訊。[2][1]

## 專案結構

從你提供的內容可看出目前專案工作區中包含應用程式碼與虛擬環境套件，且 `venv` 內部已展開大量 PyTorch、TorchVision 與相依套件目錄，因此 README 應將「專案原始碼」與「venv 產生內容」清楚區分。[1]
正式版本建議只在 README 中保留業務相關目錄，而不要把 `venv/Lib/site-packages/...` 的深層樹狀結構當成主要專案架構，因為那些內容屬於安裝依賴，不是你需要維護的業務程式。[1]

```text
attack-prediction/
├─ backend/
│  ├─ main.py
│  ├─ api/
│  ├─ core/
│  ├─ db/
│  ├─ ml/
│  │  ├─ detection/
│  │  ├─ pose/
│  │  ├─ tracking/
│  │  ├─ stgcn/
│  │  ├─ lstm/
│  │  └─ rag/
│  ├─ models/
│  ├─ schemas/
│  └─ scripts/
├─ frontend/
│  ├─ src/
│  │  ├─ components/
│  │  ├─ hooks/
│  │  ├─ pages/
│  │  ├─ services/
│  │  └─ types/
│  └─ public/
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ embeddings/
├─ models/
│  ├─ yolo/
│  ├─ stgcn/
│  └─ lstm/
├─ docker-compose.yml
├─ docker-compose.prod.yml
├─ .env.example
└─ README.md
```

這種切法與你貼出的 `backend`、`frontend`、`api`、`core`、`models`、`hooks`、`Dashboard`、`EventDetail` 等資訊一致，也更符合你實際在做的全端 AI 監控平台形式。[1]

## 後端設計

後端以 FastAPI 為核心，提供 REST API、Swagger UI、ReDoc 與 WebSocket 告警通道，對外介面包括事件建立、事件列表、單筆事件查詢、相似事件查詢與即時告警流。[1]
你貼出的介面資訊顯示常見端點包含 `POST /api/events`、`GET /api/events`、`GET /api/events/{id}`、`GET /api/events/{id}/similar` 與 `ws://localhost:8000/ws/alerts`，這很適合前端即時儀表板與事件詳情頁整合。[1]

### 後端模組建議

- `api/`：路由與 API 定義。
- `core/`：設定檔、權限驗證、JWT、共用設定。
- `db/`：SQLAlchemy model、session、migration。
- `ml/detection/`：YOLO26 偵測。
- `ml/tracking/`：ByteTrack 追蹤。
- `ml/pose/`：MediaPipe Pose 骨架擷取。
- `ml/stgcn/`：ST-GCN 訓練與推論模組。
- `ml/lstm/`：長時序意圖預測。
- `ml/rag/`：Milvus 相似檢索與事件輔助解釋。[1][2]

## 前端設計

前端以 React 18 + Vite + TypeScript 建立，搭配 Tailwind 與 React Query，適合建立即時儀表板、告警卡片、攝影機拼接畫面與事件詳情介面。[1]
從你貼出的內容可看出重要元件包含 `CameraTile`、`AlertCard`、`Dashboard`、`EventDetail`，以及 `useEvents`、`useWebSocket` 等 hooks，這表示前端主要已圍繞即時監控與事件管理兩條主線設計。[1]

### 前端模組建議

- `components/CameraTile.tsx`：單路攝影機畫面與狀態顯示。
- `components/AlertCard.tsx`：即時告警卡片與風險等級標示。
- `pages/Dashboard.tsx`：監控總覽頁。
- `pages/EventDetail.tsx`：事件詳情與時間軸頁。
- `hooks/useEvents.ts`：事件列表查詢。
- `hooks/useWebSocket.ts`：即時告警訂閱。
- `services/api.ts`：封裝 REST API 請求。[1]

## ST-GCN 模組定位

在整個攻擊預測系統中，ST-GCN 負責處理骨架時序資料，將單人或多人姿態變化轉換為動作辨識特徵，作為危險行為分類與風險推估的核心模型之一。[2]
你的研究文件中也明確將 ST-GCN 與 LSTM 視為中樞時序模型，並以滑動視窗、固定長度片段與 L0-L4 分級作為設計方向，因此 README 應把 ST-GCN 模組寫成系統核心，而不是附加功能。[2]

### ST-GCN 子專案建議結構

```text
backend/ml/stgcn/
├─ data/
│  ├─ raw/
│  ├─ processed/
│  │  ├─ train/
│  │  ├─ val/
│  │  └─ test/
│  └─ splits/
├─ datasets/
│  └─ skeleton_dataset.py
├─ models/
│  └─ stgcn.py
├─ scripts/
│  ├─ preprocess.py
│  ├─ train.py
│  └─ infer.py
├─ checkpoints/
├─ logs/
└─ requirements.txt
```

這樣可以與整體後端架構整合，同時保留模型訓練所需的獨立性。[2][1]

## 系統功能

根據你貼出的專案描述，系統至少涵蓋 12 項核心能力，包含即時告警、事件追蹤、RAG 相似案例查詢、角色權限、RTSP 攝影機接入與多端裝置支援。[1]
系統設計也強調 10 秒內的時序分析、X1-X9 類型化風險特徵與高風險事件的快速呈現，這對校園或公共空間中的預警應用特別重要。[1][2]

### 代表性能力

- 即時影像串流與多鏡頭監看。[1]
- YOLO26 + ByteTrack 人員偵測與追蹤。[1][2]
- MediaPipe Pose 骨架抽取。[2]
- ST-GCN / LSTM 動作與意圖分析。[2]
- AHP 權重風險分級與 L0-L4 告警。[2][1]
- WebSocket 即時告警推送。[1]
- RAG + Milvus 相似事件檢索。[1][2]
- 管理員、主管、警戒人員等角色權限分流。[1]

## 角色與權限

你提供的內容顯示系統至少有 `officer`、`supervisor` 與 `admin` 等角色，且不同角色對監看、管理與設定功能有不同權限範圍。[1]
README 建議將角色權限表明確列出，讓前後端在 API 授權與畫面顯示上都能保持一致。[1]

| 角色 | 主要權限 |
|---|---|
| officer | 觀看告警、查看指定事件、監控攝影機畫面 [1] |
| supervisor | 檢視風險事件、審核結果、跨場域調度資訊 [1] |
| admin | 使用者管理、系統設定、資料源設定、全域權限控管 [1] |

## 開發環境

你貼出的部署資訊指出本專案開發環境建議使用 Docker、Docker Compose、GPU、NVIDIA CUDA 12、Node.js 18 與 Python 3.11。[1]
這代表系統從一開始就以可容器化部署與 GPU 推論為目標，而不是僅限於單機腳本測試。[1]

### 本機啟動

```bash
cp .env.example .env
docker-compose up -d
docker-compose exec backend alembic upgrade head
docker-compose exec backend python scripts/seed-data.py
```

完成後可透過以下入口使用系統：[1]

- 前端：`http://localhost:3000` [1]
- API 文件：`http://localhost:8000/docs` [1]
- ReDoc：`http://localhost:8000/redoc` [1]
- WebSocket：`ws://localhost:8000/ws/alerts` [1]

## 環境變數

你提供的 `.env.example` 摘要中已包含 PostgreSQL、Redis、Milvus 與 JWT 相關設定，因此 README 應保留最小必要環境變數說明。[1]

```env
DATABASE_URL=postgresql://postgres:password@postgres:5432/attackintent
REDIS_URL=redis://redis:6379/0
MILVUS_URL=milvus://milvus:19530
JWT_SECRET=your-super-secret-jwt-key-here
```

若系統要正式上線，還應補充 MinIO、RTSP 來源、模型路徑與管理員初始帳密等設定。[1]

## 測試帳號

你貼出的內容顯示系統提供預設登入帳號，例如 `admin/admin` 與 `officer/officer`，方便本機展示或初期驗證。[1]
正式環境不應保留預設密碼，README 也應提醒部署後立即更換。[1]

## 效能目標

專案描述中提到 1 秒級 WebSocket 告警更新、12 路攝影機、1080p RTSP、30 FPS、L3/L4 高風險事件偵測率 85%、延遲約 500ms、8GB RAM 與 1 張 RTX 4060 GPU 等性能條件。[1]
這些數字可作為 README 中的系統容量參考與硬體規劃基準，但在正式文件中最好再補充測試條件、批次大小與推論配置說明。[1]

## API 範例

README 可保留最常用的 API 與 WebSocket 入口，方便前端與測試人員快速串接。[1]

```text
POST   /api/events
GET    /api/events
GET    /api/events/{id}
GET    /api/events/{id}/similar
WS     /ws/alerts
```

若你之後把 ST-GCN 單獨包成模型服務，也可以再加上例如 `/api/infer/skeleton` 或 `/api/infer/risk-score` 類型的介面。[2]

## 開發流程

你貼出的協作流程包含 Fork、切 branch、commit、push 與 pull request，這代表專案適合用標準 Git flow 來管理功能開發。[1]
如果未來你和隊友會同時改前端、後端與 AI 模型，README 最好再加上命名規範、commit convention 與模型版本管理原則。[1]

```bash
1. Fork
2. git checkout -b feature-xxx
3. git commit -m "feat: xxx"
4. git push origin feature-xxx
5. Pull Request
```
