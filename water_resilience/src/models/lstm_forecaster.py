#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LSTM 单指标(压力)短期预测 + 动态异常度(供吸收能力指标使用)。

复用 rnn_llm_system 的 LSTM 设计思路(2 层 LSTM + Dropout)。每个乡镇在日尺度压力序列上
训练 LSTM 预测下一日, 计算 MAE/RMSE; 并用预测残差聚合为 乡镇×季度 的动态异常度。CPU 训练。
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import wr_config as C
from simulate.timeseries import default_path as ts_path


class LSTMForecaster(nn.Module):
    def __init__(self, hidden=32, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _windows(series, w):
    X, y = [], []
    for i in range(len(series) - w):
        X.append(series[i:i + w])
        y.append(series[i + w])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_town(series, window=14, epochs=60, seed=42):
    torch.manual_seed(seed)
    mu, sd = series.mean(), series.std() + 1e-6
    s = (series - mu) / sd
    X, y = _windows(s, window)
    n_tr = int(len(X) * 0.8)
    Xtr = torch.tensor(X[:n_tr]).unsqueeze(-1)
    ytr = torch.tensor(y[:n_tr]).unsqueeze(-1)
    Xte = torch.tensor(X[n_tr:]).unsqueeze(-1)
    yte = y[n_tr:]
    model = LSTMForecaster()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    lossf = nn.MSELoss()
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        loss = lossf(model(Xtr), ytr)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xte).squeeze(-1).numpy()
    pred_real = pred * sd + mu
    yte_real = yte * sd + mu
    mae = float(np.mean(np.abs(pred_real - yte_real)))
    rmse = float(np.sqrt(np.mean((pred_real - yte_real) ** 2)))
    # 全序列残差(用于动态异常度)
    with torch.no_grad():
        allpred = model(torch.tensor(X).unsqueeze(-1)).squeeze(-1).numpy() * sd + mu
    resid = np.abs((y * sd + mu) - allpred)
    return {"mae": round(mae, 4), "rmse": round(rmse, 4)}, resid, window


def run_lstm(seed=42):
    df = pd.read_csv(ts_path(), parse_dates=["timestamp"])
    df["quarter"] = df["timestamp"].dt.year.astype(str) + "Q" + df["timestamp"].dt.quarter.astype(str)
    towns = C.TOWNS
    per_town_metrics, anomaly_rows = {}, []
    for town in towns:
        sub = df[df["town"] == town].sort_values("timestamp")
        daily = sub.set_index("timestamp")["pressure"].resample("D").mean().dropna()
        series = daily.values.astype(np.float32)
        metrics, resid, window = train_town(series, seed=seed)
        per_town_metrics[town] = metrics
        # 残差对齐到日期(前 window 天无残差)
        resid_dates = daily.index[window:]
        rq = pd.DataFrame({"date": resid_dates, "resid": resid})
        rq["quarter"] = rq["date"].dt.year.astype(str) + "Q" + rq["date"].dt.quarter.astype(str)
        g = rq.groupby("quarter")["resid"].mean()
        for q, val in g.items():
            anomaly_rows.append({"town": town, "quarter": q,
                                 "lstm_dynamic_anomaly": round(float(val), 5), "SIMULATED": 1})
    anomaly_df = pd.DataFrame(anomaly_rows)
    return {"per_town": per_town_metrics, "SIMULATED": True}, anomaly_df


if __name__ == "__main__":
    C.set_seed(C.SEED)
    res, anomaly_df = run_lstm(seed=C.SEED)
    with open(os.path.join(C.PATHS["outputs"], "lstm_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    anomaly_df.to_csv(os.path.join(C.PATHS["generated"], "lstm_anomaly_quarterly.csv"), index=False)
    maes = [m["mae"] for m in res["per_town"].values()]
    print(f"[lstm] towns={len(res['per_town'])} mean_MAE={np.mean(maes):.4f}")
    print("[lstm] sample:", list(res["per_town"].items())[0])
    print("[lstm] saved lstm_results.json + lstm_anomaly_quarterly.csv")
