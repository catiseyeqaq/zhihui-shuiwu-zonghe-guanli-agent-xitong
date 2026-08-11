#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端编排：依次运行拓扑、标注、仿真、GNN、LSTM、抽取、指标、熵权 TOPSIS 和成果导出。

每步作为子进程运行各模块(与单独运行行为一致)。默认离线(种子风险关系); --use-llm 启用 Qwen 抽取。
用法: python water_resilience/src/pipeline.py [--use-llm]
"""

import argparse
import os
import subprocess
import sys
import time

SRC = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("① 管网物理拓扑生成", "graph/topology_builder.py"),
    ("② 多因子风险标注", "graph/risk_labeler.py"),
    ("③ 压力/流量/余氯时序仿真", "simulate/timeseries.py"),
    ("④ 维修记录仿真", "simulate/maintenance.py"),
    ("⑤ 管理指标仿真", "simulate/management.py"),
    ("⑥ GNN 空间风险识别(GraphSAGE/GCN)", "models/gnn_train.py"),
    ("⑦ GNN 基线(中心性/RandomForest)", "models/baselines_graph.py"),
    ("⑧ LSTM 压力预测", "models/lstm_forecaster.py"),
    ("⑨ 时序基线(持续/MA/AR)", "models/baselines_ts.py"),
    ("⑩ 风险知识关系抽取", "extraction/risk_relations.py"),
    ("⑪ 四维指标矩阵", "indicators/compute.py"),
    ("⑫ 熵权-TOPSIS 综合评价", "evaluation/entropy_topsis.py"),
    ("⑬ 成果导出(清单/报告/看板)", "report.py"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-llm", action="store_true", help="启用 PPU 上的 Qwen 进行风险关系抽取")
    args = ap.parse_args()

    env = dict(os.environ, PYTHONPATH=SRC, PYTHONUNBUFFERED="1")
    print("=" * 64)
    print("智慧水务综合管理 Agent - 端到端流水线")
    print("数据说明：拓扑、时序、维修和管理记录均为仿真数据，仅用于功能演示")
    print("=" * 64)
    t0 = time.time()
    for name, rel in STEPS:
        cmd = [sys.executable, os.path.join(SRC, rel)]
        if rel.endswith("risk_relations.py") and args.use_llm:
            cmd.append("--use-llm")
        print(f"\n>>> {name}")
        r = subprocess.run(cmd, env=env)
        if r.returncode != 0:
            print(f"!!! 步骤失败: {name} (rc={r.returncode})")
            sys.exit(1)
    print("\n" + "=" * 64)
    print(f"流水线完成, 用时 {time.time() - t0:.1f}s")
    print("成果目录: water_resilience/outputs/  (dashboard.html / report.md / *.csv / *.json)")
    print("=" * 64)


if __name__ == "__main__":
    main()
