"""攻擊意圖預測核心：X1–X9 指數、AHP 風險聚合、L0–L4 分級、ST-GCN + LSTM 雙通道模型。

此套件刻意不在 import 時載入 torch / ultralytics，
讓 STGCN 訓練腳本可以單獨 import 特徵與風險引擎。
"""
