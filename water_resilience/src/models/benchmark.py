#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练、评估与基线对比的一键脚本。

GNN 空间风险识别(三级分类): GraphSAGE(主) vs GCN(对比) vs 图中心性(基线) vs RandomForest(非图基线)
  → 指标: Accuracy / Macro-F1。
LSTM 单指标(压力)时序预测: LSTM vs 持续性(Persistence) vs 移动平均(MA) vs AR(p)
  → 指标: MAE / RMSE(各乡镇均值)。
输出: 终端对比表 + water_resilience/outputs/benchmark_report.md + benchmark_metrics.csv。

说明: 复用已有训练/基线模块(gnn_train / baselines_graph / lstm_forecaster / baselines_ts)，
本脚本只做统一编排与对比表汇总。数据为规则+仿真(SIMULATED)，仅用于方法演示。

用法: PYTHONPATH=water_resilience/src python3 water_resilience/src/models/benchmark.py
"""

import csv
import os

import wr_config as C
from graph.topology_builder import build_topology, load_topology, save_topology, default_path as topo_path
from graph.risk_labeler import label_topology
from simulate.timeseries import simulate as simulate_ts, default_path as ts_path
from models.gnn_train import run_gnn
from models.baselines_graph import run_baselines as run_gnn_baselines
from models.lstm_forecaster import run_lstm
from models.baselines_ts import run_baselines_ts


def _ensure_data():
    """确保基准所需数据就绪(带风险标签的拓扑 + 压力时序), 缺失则规则+仿真生成。"""
    tp = topo_path()
    need_topo = True
    if os.path.exists(tp):
        try:
            G = load_topology(tp)
            if G.number_of_nodes() and all("risk_label" in G.nodes[n] for n in G.nodes()):
                need_topo = False
        except Exception:
            need_topo = True
    if need_topo:
        G = build_topology()
        label_topology(G)
        save_topology(G, tp)
        print("[data] 生成带风险标签的管网拓扑 ->", tp)
    if not os.path.exists(ts_path()):
        simulate_ts().to_csv(ts_path(), index=False)
        print("[data] 生成压力/流量/余氯时序 ->", ts_path())


def _fmt(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)


def run_benchmark(seed=None):
    seed = C.SEED if seed is None else seed
    C.set_seed(seed)
    _ensure_data()

    # ---- GNN 轨: 空间风险三级分类 ----
    print("[benchmark] 训练 GNN(GraphSAGE/GCN) 与基线(中心性/RandomForest)...")
    gnn = run_gnn(seed=seed)
    gbl = run_gnn_baselines(seed=seed)
    gnn_rows = []  # (模型, 类别, Accuracy, Macro-F1)
    for name, m in gnn["models"].items():
        gnn_rows.append((name, "GNN", m["test_acc"], m["test_macro_f1"]))
    gnn_rows.append(("Centrality", "基线(图中心性)", gbl["Centrality"]["test_acc"], gbl["Centrality"]["test_macro_f1"]))
    gnn_rows.append(("RandomForest", "基线(非图)", gbl["RandomForest"]["test_acc"], gbl["RandomForest"]["test_macro_f1"]))

    # ---- LSTM 轨: 单指标(压力)时序预测 ----
    print("[benchmark] 训练 LSTM 与时序基线(持续/MA/AR)...")
    lstm, _ = run_lstm(seed=seed)
    tsb = run_baselines_ts(seed=seed)
    maes = [v["mae"] for v in lstm["per_town"].values()]
    rmses = [v["rmse"] for v in lstm["per_town"].values()]
    ts_rows = [("LSTM", sum(maes) / len(maes), sum(rmses) / len(rmses))]
    for method, s in tsb["summary"].items():
        ts_rows.append((method, s["mean_mae"], s["mean_rmse"]))

    _print_and_save(gnn_rows, ts_rows, seed)
    return gnn_rows, ts_rows


def _print_and_save(gnn_rows, ts_rows, seed):
    lines = [
        "# 供水管网韧性 - 模型评估与基线对比",
        f"> 随机种子={seed}; 数据为规则+仿真(SIMULATED), 仅方法验证。",
        "",
        "## 1. GNN 空间风险识别(三级分类)",
        "模型 | 类别 | Accuracy | Macro-F1",
        "--- | --- | --- | ---",
    ]
    for name, cat, acc, f1 in gnn_rows:
        lines.append(f"{name} | {cat} | {_fmt(acc)} | {_fmt(f1)}")
    lines += [
        "",
        "## 2. LSTM 压力时序预测(各乡镇均值)",
        "方法 | MAE | RMSE",
        "--- | --- | ---",
    ]
    for name, mae, rmse in ts_rows:
        lines.append(f"{name} | {_fmt(mae)} | {_fmt(rmse)}")
    report = "\n".join(lines)
    print("\n" + report + "\n")

    out = C.PATHS["outputs"]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "benchmark_report.md"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    with open(os.path.join(out, "benchmark_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["track", "model", "category", "metric1", "value1", "metric2", "value2", "SIMULATED"])
        for name, cat, acc, f1 in gnn_rows:
            w.writerow(["gnn_risk_cls", name, cat, "accuracy", acc, "macro_f1", f1, 1])
        for name, mae, rmse in ts_rows:
            w.writerow(["lstm_pressure_forecast", name, "", "mae", mae, "rmse", rmse, 1])
    print("[benchmark] 已保存 -> outputs/benchmark_report.md + benchmark_metrics.csv")


if __name__ == "__main__":
    run_benchmark()
