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


def load_shizhu_diagnosis():
    """加载示例真实管网诊断数据(data/real/shizhu_diagnosis.json)。

    返回 dict[pipe_id → feature_dict] 或 None。
    """
    rp = real_path("shizhu_diagnosis.json")
    if rp:
        with open(rp, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_shizhu_monitoring():
    """加载示例真实水厂监测数据(data/real/shizhu_monitoring.json)。

    返回 dict[plant_name → feature_dict] 或 None。
    """
    rp = real_path("shizhu_monitoring.json")
    if rp:
        with open(rp, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ── 数据源声明(真实数据检测 + 回退 + 动态声明) ───────────────────────────

def detect_real_data() -> dict:
    """检测所有可用真实数据文件, 返回各数据源状态。

    Returns:
        dict: {
            "shizhu_diagnosis": {"exists": bool, "count": int, "path": str|None},
            "shizhu_monitoring": {"exists": bool, "count": int, "path": str|None},
            "topology": {"exists": bool, "path": str|None},
            "timeseries": {"exists": bool, "path": str|None},
            "maintenance": {"exists": bool, "path": str|None},
            "management": {"exists": bool, "path": str|None},
        }
    """
    result = {}

    # 示例管网诊断
    diag_path = real_path("shizhu_diagnosis.json")
    if diag_path:
        with open(diag_path, "r", encoding="utf-8") as f:
            diag = json.load(f)
        result["shizhu_diagnosis"] = {"exists": True, "count": len(diag), "path": diag_path}
    else:
        result["shizhu_diagnosis"] = {"exists": False, "count": 0, "path": None}

    # 示例水厂监测
    monit_path = real_path("shizhu_monitoring.json")
    if monit_path:
        with open(monit_path, "r", encoding="utf-8") as f:
            monit = json.load(f)
        result["shizhu_monitoring"] = {"exists": True, "count": len(monit), "path": monit_path}
    else:
        result["shizhu_monitoring"] = {"exists": False, "count": 0, "path": None}

    # 拓扑/时序/维修/管理(原有)
    for fname, key in [("topology.json", "topology"),
                       ("timeseries.csv", "timeseries"),
                       ("maintenance_quarterly.csv", "maintenance"),
                       ("management.csv", "management")]:
        p = real_path(fname)
        result[key] = {"exists": p is not None, "path": p}

    return result


def data_source_declaration() -> str:
    """动态生成数据源声明文本。

    检测真实数据文件存在性, 生成类似:
    "本报告使用示例县真实供水数据44条(shizhu_diagnosis 25条 + shizhu_monitoring 15条 + 预警4条)+仿真数据XX条"
    """
    det = detect_real_data()
    real_parts = []
    real_total = 0

    if det["shizhu_diagnosis"]["exists"]:
        n = det["shizhu_diagnosis"]["count"]
        real_parts.append(f"管网诊断{n}条")
        real_total += n

    if det["shizhu_monitoring"]["exists"]:
        n = det["shizhu_monitoring"]["count"]
        real_parts.append(f"水厂监测{n}条")
        real_total += n

    # 预警规则(硬编码在 warning_rules.py, 始终可用)
    try:
        import sys
        _ws = os.path.join(os.path.dirname(REAL_DIR), "..", "..", "services")
        if _ws not in sys.path:
            sys.path.insert(0, _ws)
        from warning_rules import SHIZHU_REAL_WARNINGS
        nw = len(SHIZHU_REAL_WARNINGS)
        real_parts.append(f"预警记录{nw}条")
        real_total += nw
    except ImportError:
        pass

    sim_parts = []
    if not det["topology"]["exists"]:
        sim_parts.append("拓扑仿真")
    if not det["timeseries"]["exists"]:
        sim_parts.append("时序仿真")
    if not det["maintenance"]["exists"]:
        sim_parts.append("维修台账仿真")
    if not det["management"]["exists"]:
        sim_parts.append("管理评分仿真")

    if real_parts:
        real_str = " + ".join(real_parts)
        sim_str = " + ".join(sim_parts) if sim_parts else "无"
        return (f"本报告使用示例县真实供水数据{real_total}条({real_str})"
                f"+ 仿真数据({sim_str})。"
                f"真实数据路径: {REAL_DIR}")
    else:
        return (f"本报告全部数据为仿真(SIMULATED), 仅用于方法验证, 不代表示例真实结论。"
                f"仿真路径: {REAL_DIR}")


def has_real_shizhu_data() -> bool:
    """快速判断是否有示例真实数据(至少一个文件存在)。"""
    det = detect_real_data()
    return det["shizhu_diagnosis"]["exists"] or det["shizhu_monitoring"]["exists"]
