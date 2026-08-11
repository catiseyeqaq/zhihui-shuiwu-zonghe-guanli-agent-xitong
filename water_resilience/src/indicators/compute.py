#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""四维韧性指标计算: 融合拓扑/时序/GNN/LSTM/维修/管理, 构建 乡镇×季度 × 14指标 矩阵。

各指标数值方向与 config/indicators.yaml 一致(positive 越大越好 / negative 越大越差),
方向的正负理想解处理在 entropy_topsis 中完成。
"""

import json
import os

import numpy as np
import pandas as pd

import wr_config as C
from graph.topology_builder import load_topology, default_path as topo_path


def _town_topology_indicators(G):
    out = {}
    for t in C.TOWNS:
        nodes = [n for n in G.nodes() if G.nodes[n]["town"] == t]
        sub = G.subgraph(nodes)
        edges = list(sub.edges(data=True))
        mrisk = np.mean([e["material_risk"] for _, _, e in edges]) if edges else 0.5
        age = np.mean([e["age"] for _, _, e in edges]) if edges else 15
        degs = [d for _, d in sub.degree()]
        n_valve = sum(1 for n in nodes if G.nodes[n]["node_type"] == "valve")
        out[t] = {
            "pipe_material_reliability": round(1 - float(mrisk), 4),
            "pipe_age_health": round(1 - float(age) / 30.0, 4),
            "network_connectivity": round(float(np.mean(degs)) if degs else 0.0, 4),
            "supply_redundancy": round(float(np.mean([1 if d >= 2 else 0 for d in degs])) if degs else 0.0, 4),
            "valve_isolation": round(n_valve / max(1, len(nodes)), 4),
        }
    return out


def _gnn_spatial_risk(path):
    with open(path, "r", encoding="utf-8") as f:
        res = json.load(f)
    rows = {}
    for n, d in res["node_pred"].items():
        rows.setdefault(d["town"], []).append(d["risk_prob_high"])
    return {t: round(float(np.mean(v)), 4) for t, v in rows.items()}


def _pressure_stability(ts_csv):
    df = pd.read_csv(ts_csv, parse_dates=["timestamp"])
    df["quarter"] = df["timestamp"].dt.year.astype(str) + "Q" + df["timestamp"].dt.quarter.astype(str)
    g = df.groupby(["town", "quarter"])["pressure"].std().reset_index()
    g["pressure_stability"] = (1.0 / (1.0 + g["pressure"])).round(4)
    return g[["town", "quarter", "pressure_stability"]]


def build_matrix():
    G = load_topology(topo_path())
    topo_ind = _town_topology_indicators(G)
    gnn_risk = _gnn_spatial_risk(os.path.join(C.PATHS["outputs"], "gnn_results.json"))
    press = _pressure_stability(os.path.join(C.PATHS["generated"], "timeseries.csv"))
    lstm = pd.read_csv(os.path.join(C.PATHS["generated"], "lstm_anomaly_quarterly.csv"))
    maint = pd.read_csv(os.path.join(C.PATHS["generated"], "maintenance_quarterly.csv"))
    manage = pd.read_csv(os.path.join(C.PATHS["generated"], "management.csv"))

    rows = []
    for t in C.TOWNS:
        for q in C.QUARTERS:
            row = {"town": t, "quarter": q}
            row.update(topo_ind[t])
            row["gnn_spatial_risk"] = gnn_risk.get(t, 0.0)
            pv = press[(press.town == t) & (press.quarter == q)]["pressure_stability"]
            row["pressure_stability"] = float(pv.iloc[0]) if len(pv) else 0.0
            lv = lstm[(lstm.town == t) & (lstm.quarter == q)]["lstm_dynamic_anomaly"]
            row["lstm_dynamic_anomaly"] = float(lv.iloc[0]) if len(lv) else 0.0
            mv = maint[(maint.town == t) & (maint.quarter == q)]
            for c in ["fault_discovery_time", "avg_repair_time", "supply_recovery_time", "repeat_fault_improvement"]:
                row[c] = float(mv[c].iloc[0]) if len(mv) else 0.0
            gv = manage[(manage.town == t) & (manage.quarter == q)]
            for c in ["inspection_completeness", "training_knowledge_update"]:
                row[c] = float(gv[c].iloc[0]) if len(gv) else 0.0
            row["SIMULATED"] = 1
            rows.append(row)
    return pd.DataFrame(rows)


def default_path():
    return os.path.join(C.PATHS["generated"], "indicator_matrix.csv")


if __name__ == "__main__":
    C.set_seed(C.SEED)
    df = build_matrix()
    df.to_csv(default_path(), index=False)
    ind = C.load_indicators()["indicators"]
    print(f"[indicators] matrix {df.shape[0]} rows x {len(ind)} indicators")
    print("[indicators] columns:", [c for c in df.columns if c not in ('town','quarter','SIMULATED')])
    print("[indicators] saved ->", default_path())
