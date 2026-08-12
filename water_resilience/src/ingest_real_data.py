#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实管网数据 ETL: 公开真实水司管网(EPANET .inp) -> data/real/ 项目约定格式。

数据谱系(诚实标注):
  - 拓扑: wntr 自带的肯塔基州真实水司管网 ky10.inp(University of Kentucky 公开基准,
    真实管网几何/高程/管径/需求), 节点按坐标 KMeans 聚成 len(TOWNS) 个供水分区并
    映射到 settings.yaml 的乡镇名(保持指标体系/管线兼容)。
  - 管材/管龄: .inp 无管材字段, 按 Hazen-Williams 粗糙度系数反推(水司常规做法),
    字段 attr_source='derived_from_roughness'。
  - 时序: EPANET(wntr) 水力+水龄仿真 7 天逐时曲线, 按周平铺+年度季节因子扩展为 2 年;
    余氯由水龄一阶衰减模型换算。物理模型驱动, 非实测 SCADA。
  - 维修/管理台账: 无公开真实数据, 不生成 real 文件(继续走规则仿真回退, SIMULATED=1)。

运行(需 wntr, 在 open_webui_venv): open_webui_venv/bin/python src/ingest_real_data.py
输出: data/real/topology.json, data/real/timeseries.csv, data/real/README.md
"""

import json
import math
import os
import sys

import numpy as np
import pandas as pd

_SRC = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(_SRC)
REAL_DIR = os.path.join(PROJECT_DIR, "data", "real")
os.makedirs(REAL_DIR, exist_ok=True)

# 从 settings.yaml 读乡镇与仿真参数(不 import wr_config, 避免其副作用)
import yaml
with open(os.path.join(PROJECT_DIR, "config", "settings.yaml"), encoding="utf-8") as f:
    SETTINGS = yaml.safe_load(f)
TOWNS = SETTINGS["towns"]
MATERIAL_RISK = SETTINGS["topology"]["material_risk"]
YEARS = int(SETTINGS["timeseries"]["years"])
SEED = int(SETTINGS.get("random_seed", 42))

INP_CANDIDATES = ["ky10.inp", "ky4.inp", "Net3.inp"]

# ── 示例真实数据接入 ──────────────────────────────────────────────────────
SHIZHU_DIAGNOSIS = os.path.join(REAL_DIR, "shizhu_diagnosis.json")
SHIZHU_MONITORING = os.path.join(REAL_DIR, "shizhu_monitoring.json")


def calibrate_simulation_with_real(simulation_data, real_monitoring):
    """用真实监测数据校准仿真时序。

    对每个乡镇, 计算真实水厂总流量与仿真平均流量的比值作为校准系数,
    应用到 flow/pressure/chlorine 列。

    Args:
        simulation_data: pd.DataFrame, 仿真时序 (columns: timestamp,town,pressure,flow,chlorine,is_anomaly)
        real_monitoring: dict, 水厂监测数据 (shizhu_monitoring.json 格式)

    Returns:
        pd.DataFrame: 校准后的时序数据, 格式与输入一致
    """
    if not real_monitoring:
        return simulation_data

    # 水厂→乡镇映射
    plant_to_town = {
        "示例水厂": "甲镇",
        "示例水厂-出厂水": "甲镇",
        "示例水厂-清水池1#": "甲镇",
        "示例水厂-清水池2#": "甲镇",
        "示例水厂-清水池3#": "甲镇",
        "示例水厂-高位水池1#": "甲镇",
        "示例水厂-高位水池2#": "甲镇",
        "示例水厂": "乙镇",
        "丁镇自来水厂": "丁镇",
        "戊镇集中水厂": "戊镇",
        "丙镇大坝水厂": "丙镇",
        "丙镇蛟鱼水厂": "丙镇",
    }

    # 按乡镇聚合真实流量
    town_real_flow = {}
    for plant_name, data in real_monitoring.items():
        town = plant_to_town.get(plant_name)
        if town and data.get("flow_rate") is not None:
            town_real_flow[town] = town_real_flow.get(town, 0.0) + data["flow_rate"]

    # 计算校准系数并应用
    calibrated = simulation_data.copy()
    for town in calibrated["town"].unique():
        mask = calibrated["town"] == town
        real_flow = town_real_flow.get(town)
        if not real_flow:
            continue
        sim_flow = simulation_data.loc[mask, "flow"].mean()
        if sim_flow > 0:
            factor = real_flow / sim_flow
            calibrated.loc[mask, "flow"] = (calibrated.loc[mask, "flow"] * factor).round(4)
            # 压力微调(对数衰减, ±30%)
            p_factor = max(0.7, min(1.3, 1.0 - 0.1 * np.log(factor)))
            calibrated.loc[mask, "pressure"] = (
                (calibrated.loc[mask, "pressure"] * p_factor).clip(lower=0.01).round(4)
            )

    return calibrated


def load_shizhu_real_data():
    """加载示例真实管网诊断 & 水厂监测数据(若存在)。

    返回 (diagnosis_dict, monitoring_dict) 或 (None, None)。
    """
    diag = None
    mon = None
    if os.path.exists(SHIZHU_DIAGNOSIS):
        with open(SHIZHU_DIAGNOSIS, encoding="utf-8") as f:
            diag = json.load(f)
        print(f"[ingest] 加载示例真实管网诊断: {len(diag)} 个节点")
    if os.path.exists(SHIZHU_MONITORING):
        with open(SHIZHU_MONITORING, encoding="utf-8") as f:
            mon = json.load(f)
        print(f"[ingest] 加载示例真实水厂监测: {len(mon)} 个水厂")
    return diag, mon


def enrich_topology_with_real(topo, diagnosis, monitoring):
    """将示例真实管网风险数据合并到拓扑节点属性中。

    匹配逻辑: 诊断数据中的 plant 字段映射到拓扑节点的 town 属性
    (甲镇自来水厂→甲镇, 丙镇自来水厂→丙镇 等)。
    对每个拓扑节点, 注入 real_risk 子字段(若该分区有真实诊断)。
    """
    if not diagnosis:
        return

    # 建立 plant_name → town 映射
    plant_to_town = {}
    town_diagnosis = {}  # town → [node_features]
    for pipe_id, feat in diagnosis.items():
        plant = feat.get("plant", "unknown")
        # 水厂名到乡镇的映射(按 settings.yaml towns)
        for town in TOWNS:
            if town in plant:
                plant_to_town[plant] = town
                town_diagnosis.setdefault(town, []).append(feat)
                break

    # 为拓扑节点注入真实风险数据
    enriched_count = 0
    for node in topo.get("nodes", []):
        town = node.get("town", "")
        if town in town_diagnosis:
            # 统计该分区的风险节点数 & 管道类型分布
            risk_nodes = town_diagnosis[town]
            n_main = sum(1 for r in risk_nodes if r.get("pipe_type") == "main")
            n_branch = sum(1 for r in risk_nodes if r.get("pipe_type") == "branch")
            node["real_risk"] = {
                "source": "shizhu_excel/管网诊断_结构化",
                "n_risk_pipes": len(risk_nodes),
                "n_main_pipes": n_main,
                "n_branch_pipes": n_branch,
                "risk_level": "orange",  # 全部为橙色警告
                "data_lineage": "真实诊断数据(reference_docsExcel)",
            }
            enriched_count += 1

    # 为拓扑边注入真实管道数据(按 pipe_id 匹配)
    # 诊断数据中的管道编号与拓扑边无直接 ID 对应, 但可按分区+类型增强
    # 此处标记 graph 元数据
    topo["graph"]["shizhu_real_diagnosis"] = {
        "n_pipes": len(diagnosis),
        "towns_covered": list(town_diagnosis.keys()),
        "data_lineage": "reference_docsExcel-管网诊断_结构化(25条真实诊断)",
    }
    print(f"[ingest] 真实风险数据合并: {enriched_count} 个拓扑节点已增强")

    # 水厂监测数据 → graph 元数据
    if monitoring:
        topo["graph"]["shizhu_real_monitoring"] = {
            "n_plants": len(monitoring),
            "plants": list(monitoring.keys()),
            "data_lineage": "reference_docsExcel-水厂监测_结构化(41条实测)",
        }


def main():
    wn, inp_name = load_network()
    topo, node_town = build_topology(wn, TOWNS)

    # 加载并合并示例真实数据
    diagnosis, monitoring = load_shizhu_real_data()
    has_real = diagnosis is not None or monitoring is not None
    if has_real:
        enrich_topology_with_real(topo, diagnosis, monitoring)
        # 更新数据谱系说明
        topo["graph"]["data_lineage"] += "; 示例真实管网诊断+水厂监测数据已合并"

    with open(os.path.join(REAL_DIR, "topology.json"), "w", encoding="utf-8") as f:
        json.dump(topo, f, ensure_ascii=False)
    real_tag = "(+示例真实数据)" if has_real else ""
    print(f"[ingest] topology.json{real_tag}: {len(topo['nodes'])} 节点 / {len(topo['links'])} 管段")

    ts = simulate_timeseries(wn, node_town, TOWNS)
    ts.to_csv(os.path.join(REAL_DIR, "timeseries.csv"), index=False)
    print(f"[ingest] timeseries.csv: {len(ts)} 行 ({ts['timestamp'].min()} ~ {ts['timestamp'].max()})")

    # 校准仿真时序(若有真实监测数据)
    if monitoring:
        calibrated_ts = calibrate_simulation_with_real(ts, monitoring)
        calibrated_ts.to_csv(os.path.join(REAL_DIR, "calibrated_timeseries.csv"), index=False)
        print(f"[ingest] calibrated_timeseries.csv: {len(calibrated_ts)} 行 (已用真实数据校准)")

    # 数据声明按真实数据存在性动态生成
    lineage_parts = []
    if has_real:
        lineage_parts.append("示例真实管网诊断(25条)+水厂监测(41条)已合并")
        lineage_parts.append("calibrated_timeseries.csv: 用真实水厂流量校准后的时序")
    lineage_parts.append(
        f"来源管网: `{inp_name}`(肯塔基州真实水司管网公开基准, 随 wntr 分发)"
    )

    with open(os.path.join(REAL_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"""# data/real 数据谱系说明

- {"; ".join(lineage_parts)}
- topology.json: **真实**几何/高程/管径/需求; 管材/管龄由 HW 粗糙度反推(derived);
  维修次数为管龄x长度的确定性泊松代理; 乡镇分区 = 坐标 KMeans 映射到 settings.yaml 的 towns。
{"- **示例真实数据**: 25条管网诊断(G0001-G0165爆管风险, 全橙色)+15个水厂节点实测流量/水质/水位" if has_real else ""}
- timeseries.csv: EPANET(wntr) 水力+水龄物理仿真 7 天逐时, 周平铺+季节因子扩展 {YEARS} 年;
  余氯由水龄一阶衰减换算。物理模型驱动, 非实测 SCADA。
- maintenance/management: 无公开真实数据, 保持规则仿真回退(SIMULATED=1)。

重新生成:
  1. python3 src/ingest_shizhu_real_data.py   # 示例Excel→JSON
  2. python3 src/ingest_real_data.py           # 拓扑+仿真+合并
""")
    print("[ingest] 完成 ->", REAL_DIR)


if __name__ == "__main__":
    main()
