#!/usr/bin/env bash
# RWF-2000 攻擊類型分類（整段影片 × 雙人）完整流程：
#   YOLO26-pose 骨架抽取（3 程序平行）→ preprocess_clips → train --task classify → 評估 → 混淆矩陣
# 用法（在 STGCN/ 目錄，Git Bash）：
#   bash scripts/run_rwf2000_clips.sh                 # 預設 YOLO26-pose
#   POSE=mediapipe bash scripts/run_rwf2000_clips.sh  # 改用 MediaPipe（研究計畫原始做法，用於比較）
#   EXTRACT=skip bash scripts/run_rwf2000_clips.sh    # 已抽取過，只重做前處理以後
set -uo pipefail
cd "$(dirname "$0")/.."
PY=../backend/.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
POSE=${POSE:-yolo}
SHARDS=${SHARDS:-3}
EXTRACT=${EXTRACT:-resume}
if [ "$POSE" = "yolo" ]; then TAG=yolopose; SESS=data/interim/sessions_yolopose; else TAG=mediapipe; SESS=data/interim/sessions; fi
DATA=data/processed_clips_$TAG
MODEL=checkpoints/attack_classifier_rwf2000_$TAG.pt
ts() { date +%H:%M:%S; }

if [ "$EXTRACT" != "skip" ]; then
  echo "[$(ts)] 骨架抽取（$POSE，$SHARDS 個程序平行）→ $SESS"
  pids=()
  for ((i = 0; i < SHARDS; i++)); do
    $PY scripts/extract_skeletons.py --input data/raw/rwf2000 --dataset rwf2000 --every 2 --pose "$POSE" \
      --output "$SESS" --shard "$i/$SHARDS" > "logs/extract_${TAG}_shard$i.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  fail=0
  for ((i = 0; i < SHARDS; i++)); do grep -a "^完成" "logs/extract_${TAG}_shard$i.log" || fail=1; done
  if [ "$fail" -ne 0 ]; then echo "[$(ts)] ❌ 骨架抽取有程序失敗，見 logs/extract_${TAG}_shard*.log"; exit 1; fi
fi

echo "[$(ts)] 前處理（整段影片 × 雙人）→ $DATA"
$PY scripts/preprocess_clips.py --sessions "$SESS" --output "$DATA" --clean \
  > "logs/preprocess_clips_$TAG.log" 2>&1 || { echo "❌ 前處理失敗，見 logs/preprocess_clips_$TAG.log"; exit 1; }
tail -25 "logs/preprocess_clips_$TAG.log"

echo "[$(ts)] 訓練（只訓練攻擊類型）"
$PY scripts/train.py --task classify --data "$DATA" --epochs 100 --batch-size 32 --lr 1e-3 \
  --weight-decay 5e-4 --dropout 0.4 --label-smoothing 0.1 --patience 20 --out "$MODEL"
code=$?
if [ ! -f "$MODEL" ]; then echo "[$(ts)] ❌ 訓練失敗（exit $code）"; exit 1; fi

echo "[$(ts)] 測試集評估"
$PY scripts/evaluate.py --data "$DATA" --split test --model "$MODEL" --out "logs/eval_clips_${TAG}_test.json"
$PY scripts/plot_confusion.py --report "logs/eval_clips_${TAG}_test.json" \
  --title "混淆矩陣 — RWF-2000 測試集（整段影片 × 雙人，骨架：$TAG）"
echo "[$(ts)] 全部完成"
