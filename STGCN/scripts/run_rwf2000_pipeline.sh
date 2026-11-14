#!/usr/bin/env bash
# RWF-2000 完整流程：等骨架抽取（3 個 shard）完成 → 前處理 → 訓練 → 測試集評估 → 混淆矩陣
# 用法（在 STGCN/ 目錄，Git Bash）：bash scripts/run_rwf2000_pipeline.sh
# 若尚未抽取，先執行：python scripts/extract_skeletons.py --input data/raw/rwf2000 --dataset rwf2000 --every 2 --shard i/3
set -euo pipefail
cd "$(dirname "$0")/.."
PY=../backend/.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
SHARDS=${SHARDS:-3}

echo "[$(date +%H:%M:%S)] 等待骨架抽取完成（logs/extract_shard*.log 出現「完成」）..."
while true; do
  done_n=0
  for ((i = 0; i < SHARDS; i++)); do
    grep -q "^完成" "logs/extract_shard$i.log" 2>/dev/null && done_n=$((done_n + 1))
  done
  [ "$done_n" -ge "$SHARDS" ] && break
  sleep 60
done
echo "[$(date +%H:%M:%S)] 抽取完成：$(ls data/interim/sessions | grep -c '^rwf2000_') 個 session"

echo "[$(date +%H:%M:%S)] 前處理（3 秒視窗、預測下一秒、90 幀、coarse 分類）"
$PY scripts/preprocess.py --sessions data/interim/sessions --output data/processed --clean \
  --taxonomy coarse --window-sec 3 --horizon-sec 1 --seq-len 90

echo "[$(date +%H:%M:%S)] 訓練"
$PY scripts/train.py --data data/processed --epochs 80 --batch-size 32 --patience 12 \
  --out checkpoints/intent_model_rwf2000.pt

echo "[$(date +%H:%M:%S)] 測試集評估"
$PY scripts/evaluate.py --data data/processed --split test --model checkpoints/intent_model_rwf2000.pt \
  --out logs/eval_rwf2000_test.json

$PY scripts/plot_confusion.py --report logs/eval_rwf2000_test.json --title "混淆矩陣 — RWF-2000 測試集"
echo "[$(date +%H:%M:%S)] 全部完成"
