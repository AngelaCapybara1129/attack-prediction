#!/usr/bin/env bash
# RWF-2000 完整流程：骨架抽取（3 個程序平行）→ 前處理 → 訓練 → 測試集評估 → 混淆矩陣
# 用法（在 STGCN/ 目錄，Git Bash）：
#   bash scripts/run_rwf2000_pipeline.sh            # 從抽取開始（已抽過的影片會略過）
#   EXTRACT=overwrite bash scripts/run_rwf2000_pipeline.sh   # 重新抽取全部影片
#   EXTRACT=skip bash scripts/run_rwf2000_pipeline.sh        # 只做前處理以後的步驟
set -uo pipefail
cd "$(dirname "$0")/.."
PY=../backend/.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1          # 讓進度即時寫進 log
SHARDS=${SHARDS:-3}
EXTRACT=${EXTRACT:-resume}
MODEL=checkpoints/intent_model_rwf2000.pt
ts() { date +%H:%M:%S; }

if [ "$EXTRACT" != "skip" ]; then
  extra=""
  [ "$EXTRACT" = "overwrite" ] && extra="--overwrite"
  echo "[$(ts)] 骨架抽取（$SHARDS 個程序平行，$EXTRACT）"
  pids=()
  for ((i = 0; i < SHARDS; i++)); do
    $PY scripts/extract_skeletons.py --input data/raw/rwf2000 --dataset rwf2000 --every 2 \
      --shard "$i/$SHARDS" ${REDO_BEFORE:+--redo-before "$REDO_BEFORE"}       $( [ "$EXTRACT" = "overwrite" ] && echo --overwrite ) > "logs/extract_shard$i.log" 2>&1 &
    pids+=($!)
  done
  fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  for ((i = 0; i < SHARDS; i++)); do grep -a "^完成" "logs/extract_shard$i.log" || fail=1; done
  if [ "$fail" -ne 0 ]; then echo "[$(ts)] ❌ 骨架抽取有程序失敗，請查看 logs/extract_shard*.log"; exit 1; fi
fi
echo "[$(ts)] 抽取完成：$(ls data/interim/sessions | grep -c '^rwf2000_') 個 session"

echo "[$(ts)] 前處理（3 秒視窗、預測下一秒、90 幀、步長 0.5 秒、骨架覆蓋 ≥ 35%、coarse 分類）"
$PY scripts/preprocess.py --sessions data/interim/sessions --output data/processed --clean \
  --taxonomy coarse --window-sec 3 --horizon-sec 1 --seq-len 90 --stride-sec 0.5 --min-valid-ratio 0.35 \
  > logs/preprocess_rwf2000.log 2>&1 || { echo "[$(ts)] ❌ 前處理失敗，見 logs/preprocess_rwf2000.log"; exit 1; }
grep -A40 '"action_counts"' data/processed/dataset_meta.json | head -20

echo "[$(ts)] 訓練"
$PY scripts/train.py --data data/processed --epochs 80 --batch-size 32 --patience 15 --out "$MODEL"
code=$?
# Windows 上 Python 結束時偶爾回傳非 0（CUDA/MediaPipe 釋放資源），只要模型有存下來就繼續
if [ ! -f "$MODEL" ]; then echo "[$(ts)] ❌ 訓練失敗（exit $code），沒有產生模型"; exit 1; fi
[ "$code" -ne 0 ] && echo "[$(ts)] ⚠️ 訓練程序結束碼 $code，但模型已存檔，繼續評估"

echo "[$(ts)] 測試集評估"
$PY scripts/evaluate.py --data data/processed --split test --model "$MODEL" --out logs/eval_rwf2000_test.json
$PY scripts/plot_confusion.py --report logs/eval_rwf2000_test.json --title "混淆矩陣 — RWF-2000 測試集"
echo "[$(ts)] 全部完成"
