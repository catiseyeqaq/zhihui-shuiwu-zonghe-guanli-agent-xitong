#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""运维/维修记录仿真(SIMULATED): 各乡镇各季度的故障事件与恢复能力指标。

生成故障事件(发现/抢修/恢复时长、是否重复故障), 并按 乡镇×季度 聚合为恢复能力相关指标:
故障发现时间、平均抢修时间、供水恢复时间、重复故障率、重复故障改善(环比下降)。非真实台账。
"""

import os

import numpy as np
import pandas as pd

import wr_config as C


def simulate(settings=None):
    settings = settings or C.SETTINGS
    towns, quarters = settings["towns"], settings["quarters"]
    rng = np.random.default_rng(int(settings.get("random_seed", 42)) + 7)
    events = []
    for town in towns:
        # 各乡镇基线运维能力不同(山地差异)
        disc_base = rng.uniform(2, 8)
        repair_base = rng.uniform(4, 12)
        for qi, q in enumerate(quarters):
            n_fault = rng.integers(3, 10)
            # 运维随时间小幅改善
            improve = 1.0 - 0.03 * qi
            for _ in range(int(n_fault)):
                disc = max(0.5, rng.normal(disc_base * improve, 1.0))
                repair = max(1.0, rng.normal(repair_base * improve, 2.0))
                recover = disc + repair + max(0.0, rng.normal(2, 1))
                events.append({
                    "town": town, "quarter": q,
                    "discovery_h": round(disc, 2), "repair_h": round(repair, 2),
                    "recovery_h": round(recover, 2),
                    "fault_type": rng.choice(["爆管", "渗漏", "水质", "设备"]),
                    "is_repeat": int(rng.random() < max(0.05, 0.35 - 0.03 * qi)),
                    "SIMULATED": 1,
                })
    ev = pd.DataFrame(events)
    agg = ev.groupby(["town", "quarter"]).agg(
        fault_discovery_time=("discovery_h", "mean"),
        avg_repair_time=("repair_h", "mean"),
        supply_recovery_time=("recovery_h", "mean"),
        repeat_rate=("is_repeat", "mean"),
        n_faults=("is_repeat", "count"),
    ).reset_index()
    # 重复故障改善 = 上一季度重复率 - 本季度(正=改善)
    agg = agg.sort_values(["town", "quarter"]).reset_index(drop=True)
    agg["repeat_fault_improvement"] = agg.groupby("town")["repeat_rate"].diff().fillna(0.0) * -1
    for c in ["fault_discovery_time", "avg_repair_time", "supply_recovery_time", "repeat_rate", "repeat_fault_improvement"]:
        agg[c] = agg[c].round(4)
    return ev, agg


if __name__ == "__main__":
    C.set_seed(C.SEED)
    from data_sources import real_path
    import pandas as pd
    gen = C.PATHS["generated"]
    rq = real_path("maintenance_quarterly.csv")
    if rq:
        agg = pd.read_csv(rq)
        agg["SIMULATED"] = 0
        agg.to_csv(os.path.join(gen, "maintenance_quarterly.csv"), index=False)
        print(f"[maintenance] quarterly rows={len(agg)} source=REAL")
    else:
        ev, agg = simulate()
        ev.to_csv(os.path.join(gen, "maintenance_events.csv"), index=False)
        agg.to_csv(os.path.join(gen, "maintenance_quarterly.csv"), index=False)
        print(f"[maintenance] events={len(ev)} town-quarter rows={len(agg)} source=SIMULATED")
    print("[maintenance] saved maintenance_quarterly.csv")
