#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""供水监测时序仿真(SIMULATED): 每个乡镇 2 年逐小时的压力/流量/余氯。

含日周期(用水高峰压力下降)+ 年季节性 + 噪声 + 随机注入异常(压力骤降),
供 LSTM 单指标(压力)预测与动态异常度评估使用。非真实监测数据。
"""

import os

import numpy as np
import pandas as pd

import wr_config as C


def simulate(settings=None):
    settings = settings or C.SETTINGS
    ts = settings["timeseries"]
    towns = settings["towns"]
    rng = np.random.default_rng(int(settings.get("random_seed", 42)))
    hours = int(ts["years"] * 365 * 24)
    idx = pd.date_range("2024-01-01", periods=hours, freq="h")
    t = np.arange(hours)
    rows = []
    for ti, town in enumerate(towns):
        base_p = ts["pressure_base_mpa"] + rng.uniform(-0.03, 0.05)
        diurnal = 0.03 * np.sin(2 * np.pi * (t % 24) / 24 - np.pi / 3)
        seasonal = 0.02 * np.sin(2 * np.pi * t / (365 * 24))
        noise = rng.normal(0, 0.006, hours)
        pressure = base_p + diurnal + seasonal + noise
        # 注入异常(压力骤降)
        n_anom = int(hours * ts["anomaly_rate"] / 12)
        anom_flag = np.zeros(hours, dtype=int)
        for _ in range(n_anom):
            start = rng.integers(0, hours - 6)
            dur = rng.integers(2, 6)
            pressure[start:start + dur] -= rng.uniform(0.05, 0.12)
            anom_flag[start:start + dur] = 1
        flow = 40 + 25 * (0.32 - pressure) / 0.05 + rng.normal(0, 2, hours)
        chlorine = 0.32 + 0.05 * np.sin(2 * np.pi * (t % 24) / 24) + rng.normal(0, 0.02, hours)
        df = pd.DataFrame({
            "timestamp": idx, "town": town,
            "pressure": pressure.round(4), "flow": flow.round(2),
            "chlorine": chlorine.round(4), "is_anomaly": anom_flag,
        })
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["SIMULATED"] = 1
    return out


def default_path():
    return os.path.join(C.PATHS["generated"], "timeseries.csv")


if __name__ == "__main__":
    C.set_seed(C.SEED)
    from data_sources import csv_or_simulate
    df, is_real = csv_or_simulate("timeseries.csv", simulate)
    df.to_csv(default_path(), index=False)
    src = "REAL" if is_real else "SIMULATED"
    anom = int(df["is_anomaly"].sum()) if "is_anomaly" in df.columns else 0
    print(f"[timeseries] rows={len(df)} towns={df['town'].nunique()} anomalies={anom} source={src}")
    print("[timeseries] saved ->", default_path())
