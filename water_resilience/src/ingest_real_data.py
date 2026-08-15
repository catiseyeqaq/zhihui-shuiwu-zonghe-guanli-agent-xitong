#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""公开基准管网数据 ETL: 公开真实水司管网(EPANET .inp) -> data/real/ 项目约定格式。

数据谱系(诚实标注):
  - 拓扑: wntr 自带的肯塔基州真实水司管网 ky10.inp(University of Kentucky 公开基准,
    真实管网几何/高程/管径/需求), 节点按坐标 KMeans 聚成 len(TOWNS) 个供水分区并
    映射到 settings.yaml 的乡镇名(保持指标体系/管线兼容)。
  - 管材/管龄: .inp 无管材字段, 按 Hazen-Williams 粗糙度系数反推(水司常规做法),
    字段 attr_source='derived_from_roughness'。
  - 时序: EPANET(wntr) 水力+水龄仿真 7 天逐时曲线, 按周平铺+年度季节因子扩展为 2 年;
    余氯由水龄一阶衰减模型换算。物理模型驱动, 非实测 SCADA。
  - 维修/管理台账: 无公开真实数据, 不生成 real 文件(继续走规则仿真回退, SIMULATED=1)。

运行(需 wntr): python3 src/ingest_real_data.py
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


def main():
    wn, inp_name = load_network()
    topo, node_town = build_topology(wn, TOWNS)

    with open(os.path.join(REAL_DIR, "topology.json"), "w", encoding="utf-8") as f:
        json.dump(topo, f, ensure_ascii=False)
    print(f"[ingest] topology.json: {len(topo['nodes'])} 节点 / {len(topo['links'])} 管段")

    ts = simulate_timeseries(wn, node_town, TOWNS)
    ts.to_csv(os.path.join(REAL_DIR, "timeseries.csv"), index=False)
    print(f"[ingest] timeseries.csv: {len(ts)} 行 ({ts['timestamp'].min()} ~ {ts['timestamp'].max()})")

    # 数据声明(诚实标注来源)
    lineage_parts = [
        f"来源管网: `{inp_name}`(肯塔基州真实水司管网公开基准, 随 wntr 分发)",
    ]

    with open(os.path.join(REAL_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"""# data/real 数据谱系说明

- {"; ".join(lineage_parts)}
- topology.json: **真实**几何/高程/管径/需求; 管材/管龄由 HW 粗糙度反推(derived);
  维修次数为管龄x长度的确定性泊松代理; 乡镇分区 = 坐标 KMeans 映射到 settings.yaml 的 towns。
- timeseries.csv: EPANET(wntr) 水力+水龄物理仿真 7 天逐时, 周平铺+季节因子扩展 {YEARS} 年;
  余氯由水龄一阶衰减换算。物理模型驱动, 非实测 SCADA。
- maintenance/management: 无公开真实数据, 保持规则仿真回退(SIMULATED=1)。

重新生成:
  python3 src/ingest_real_data.py   # 拓扑+仿真
""")
    print("[ingest] 完成 ->", REAL_DIR)


if __name__ == "__main__":
    main()
