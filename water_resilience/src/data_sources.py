#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实数据接入层: 各数据源模块优先读取 REAL_DIR 下的真实数据, 缺失时回退规则+仿真。

约定: 真实文件与生成文件同名, 放在 config paths.real 指向的目录(默认 water_resilience/data/real)。
  - topology.json           真实管网拓扑(NetworkX node-link; 节点须含 node_type/town/elevation/x/y, 边含 material/age 等)
  - timeseries.csv          真实监测时序(列: timestamp,town,pressure,flow,chlorine[,is_anomaly])
  - maintenance_quarterly.csv 真实季度维修聚合(列: town,quarter,fault_discovery_time,avg_repair_time,supply_recovery_time,repeat_fault_improvement)
  - management.csv          真实管理评分(列: town,quarter,inspection_completeness,training_knowledge_update)
真实数据统一标记 SIMULATED=0; 仿真回退标记 SIMULATED=1。指标体系与流水线无需改动。
"""

import json
import os

import networkx as nx
import pandas as pd

import wr_config as C

REAL_DIR = C.PATHS.get("real") or os.path.join(C.PROJECT_DIR, "data", "real")


def real_path(fname: str):
    """返回 REAL_DIR/fname 若存在, 否则 None。"""
    p = os.path.join(REAL_DIR, fname)
    return p if os.path.exists(p) else None


def csv_or_simulate(fname: str, simulate_fn):
    """优先读取真实 CSV(REAL_DIR/fname, 标记 SIMULATED=0); 否则调用 simulate_fn() 生成。

    返回 (df, is_real)。simulate_fn 需返回带 SIMULATED 列的 DataFrame。
    """
    rp = real_path(fname)
    if rp:
        df = pd.read_csv(rp)
        df["SIMULATED"] = 0
        return df, True
    return simulate_fn(), False


def topology_or_build(fname: str, build_fn):
    """优先读取真实拓扑(REAL_DIR/fname, NetworkX node-link JSON); 否则调用 build_fn() 生成。

    返回 (G, is_real)。真实图标记 graph['SIMULATED']=False。派生节点属性(度/中心性/管段聚合)
    由调用方在加载后按需 _annotate 补齐。
    """
    rp = real_path(fname)
    if rp:
        with open(rp, "r", encoding="utf-8") as f:
            G = nx.node_link_graph(json.load(f))
        G.graph["SIMULATED"] = False
        return G, True
    return build_fn(), False


def detect_real_data() -> dict:
    """检测所有可用真实数据文件, 返回各数据源状态。"""
    result = {}
    for fname, key in [("topology.json", "topology"),
                       ("timeseries.csv", "timeseries"),
                       ("maintenance_quarterly.csv", "maintenance"),
                       ("management.csv", "management")]:
        p = real_path(fname)
        result[key] = {"exists": p is not None, "path": p}
    return result


def data_source_declaration() -> str:
    """动态生成数据源声明文本(诚实标注真实/仿真来源)。"""
    det = detect_real_data()
    sim_parts = []
    if not det["topology"]["exists"]:
        sim_parts.append("拓扑仿真")
    if not det["timeseries"]["exists"]:
        sim_parts.append("时序仿真")
    if not det["maintenance"]["exists"]:
        sim_parts.append("维修台账仿真")
    if not det["management"]["exists"]:
        sim_parts.append("管理评分仿真")
    sim_str = " + ".join(sim_parts) if sim_parts else "无"
    return ("本报告默认数据为合成(SIMULATED), 仅用于方法验证; "
            f"若部署侧提供真实数据则优先读取({sim_str}回退)。"
            f"数据路径: {REAL_DIR}")
