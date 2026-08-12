#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""管理类指标仿真(SIMULATED, 专家评分): 各乡镇各季度的巡检制度完善度与培训水平。

以 0-1 专家评分表示适应能力中的管理维度, 随季度小幅提升(体现运维能力成长)。非真实评分。
"""

import os

import numpy as np
import pandas as pd

import wr_config as C


def simulate(settings=None):
    settings = settings or C.SETTINGS
    towns, quarters = settings["towns"], settings["quarters"]
    rng = np.random.default_rng(int(settings.get("random_seed", 42)) + 13)
    rows = []
    for town in towns:
        insp0 = rng.uniform(0.45, 0.7)
        train0 = rng.uniform(0.4, 0.65)
        for qi, q in enumerate(quarters):
            rows.append({
                "town": town, "quarter": q,
                "inspection_completeness": round(min(0.98, insp0 + 0.02 * qi + rng.normal(0, 0.02)), 4),
                "training_knowledge_update": round(min(0.98, train0 + 0.025 * qi + rng.normal(0, 0.02)), 4),
                "SIMULATED": 1,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    C.set_seed(C.SEED)
    from data_sources import csv_or_simulate
    df, is_real = csv_or_simulate("management.csv", simulate)
    df.to_csv(os.path.join(C.PATHS["generated"], "management.csv"), index=False)
    print(f"[management] rows={len(df)} source={'REAL' if is_real else 'SIMULATED'}")
    print("[management] saved management.csv")
