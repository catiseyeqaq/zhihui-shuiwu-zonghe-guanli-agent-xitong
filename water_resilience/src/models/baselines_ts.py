#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LSTM 对照基线: 持续性预测(persistence)、移动平均(MA)、AR(p)。

私有镜像无 statsmodels, 故以 numpy 最小二乘实现轻量 AR(p) 代替 ARIMA; 与 LSTM 在同一日尺度
压力序列、同一 80/20 划分上比较 MAE/RMSE。满足申报书"MA/持续/ARIMA 中一至两个基线"的要求。
"""

import json
import os

import numpy as np
import pandas as pd

import wr_config as C
from simulate.timeseries import default_path as ts_path


def _metrics(pred, true):
    pred, true = np.asarray(pred), np.asarray(true)
    return {"mae": round(float(np.mean(np.abs(pred - true))), 4),
            "rmse": round(float(np.sqrt(np.mean((pred - true) ** 2))), 4)}


def persistence(series, n_tr):
    true = series[n_tr:]
    pred = series[n_tr - 1:-1]
    return _metrics(pred, true)


def moving_average(series, n_tr, w=14):
    preds, trues = [], []
    for i in range(n_tr, len(series)):
        preds.append(series[i - w:i].mean())
        trues.append(series[i])
    return _metrics(preds, trues)


def ar_p(series, n_tr, p=7):
    # 训练集构造滞后设计矩阵, 最小二乘求 AR 系数
    def design(s):
        X, y = [], []
        for i in range(p, len(s)):
            X.append(s[i - p:i][::-1]); y.append(s[i])
        return np.array(X), np.array(y)
    Xtr, ytr = design(series[:n_tr])
    A = np.column_stack([np.ones(len(Xtr)), Xtr])
    coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    preds, trues = [], []
    for i in range(n_tr, len(series)):
        lag = series[i - p:i][::-1]
        preds.append(coef[0] + coef[1:] @ lag)
        trues.append(series[i])
    return _metrics(preds, trues)


def run_baselines_ts(seed=42):
    df = pd.read_csv(ts_path(), parse_dates=["timestamp"])
    out = {}
    for town in C.TOWNS:
        sub = df[df["town"] == town].sort_values("timestamp")
        daily = sub.set_index("timestamp")["pressure"].resample("D").mean().dropna().values.astype(float)
        n_tr = int(len(daily) * 0.8)
        out[town] = {"Persistence": persistence(daily, n_tr),
                     "MA": moving_average(daily, n_tr),
                     "AR(p)": ar_p(daily, n_tr)}
    # 汇总平均
    summary = {}
    for method in ["Persistence", "MA", "AR(p)"]:
        maes = [out[t][method]["mae"] for t in C.TOWNS]
        rmses = [out[t][method]["rmse"] for t in C.TOWNS]
        summary[method] = {"mean_mae": round(float(np.mean(maes)), 4),
                           "mean_rmse": round(float(np.mean(rmses)), 4)}
    return {"per_town": out, "summary": summary, "SIMULATED": True}


if __name__ == "__main__":
    C.set_seed(C.SEED)
    res = run_baselines_ts(seed=C.SEED)
    with open(os.path.join(C.PATHS["outputs"], "lstm_baselines.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("[ts-baseline] summary (mean MAE/RMSE):")
    for m, v in res["summary"].items():
        print(f"  {m}: {v}")
    print("[ts-baseline] saved -> lstm_baselines.json")
