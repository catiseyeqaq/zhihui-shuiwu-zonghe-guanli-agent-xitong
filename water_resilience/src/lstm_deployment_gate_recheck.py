#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LSTM 部署门复检（统一预测协议口径）

依据 configs/model_registry.json 的门禁定义（基线超越 + 区间覆盖率），
使用统一 168h 预测的正式聚合指标复检 lstm_forecast 是否可切训练态。

门禁标准：
  Gate A  lstm_beats_primary_baseline : LSTM town-macro MAE ≤ WeeklySN，全部 horizon
  Gate B  mc_coverage_within_tolerance: MC Dropout PICP 与名义 0.95 偏差 ≤ 0.05
全部通过 => deployment_gate_passed=true（注册表自动切训练态），否则维持研究态。

说明：统一协议的 LSTM168 为冻结协议下公平训练的 1×LSTM(32)；注册表权重
kg_gnn_enhanced_lstm（KG/GNN 静态特征增强）未按统一预测协议重训，不在本次复检范围，
其 2026-07-29 旧口径 MC 覆盖率作为遗留记录引用。
输出: outputs/lstm_deployment_gate_recheck_v1.json（只读冻结资产，绝不修改）
"""
from __future__ import annotations

import csv
import datetime
import json
import os

EVIDENCE_DIR = os.environ.get(
    "EVIDENCE_ASSET_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "evidence_reports"),
)
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
OLD_RECEIPT = os.path.join(OUT_DIR, "kg_gnn_lstm_validation.json")
COVERAGE_TOLERANCE = 0.05


def load_unified_forecast_aggregate():
    path = os.path.join(EVIDENCE_DIR, "unified_forecast_aggregate_metrics_v1.csv")
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("aggregation") == "town_macro_average":
                rows.append(r)
    return rows


def main() -> int:
    rows = load_unified_forecast_aggregate()
    horizons = sorted({int(r["horizon_h"]) for r in rows})
    per_horizon = {}
    gate_a = True
    for h in horizons:
        cells = {r["model"]: r for r in rows if int(r["horizon_h"]) == h}
        lstm_mae = float(cells["LSTM168"]["mae"])
        wsn_mae = float(cells["WeeklySN"]["mae"])
        beats = lstm_mae <= wsn_mae
        gate_a = gate_a and beats
        per_horizon[str(h)] = {
            "lstm_mae": lstm_mae,
            "weeklysn_mae": wsn_mae,
            "ridge_mae": float(cells["Ridge168"]["mae"]),
            "lstm_beats_weeklysn": beats,
            "mae_ratio_lstm_over_wsn": round(lstm_mae / wsn_mae, 3),
        }

    # Gate B：遗留 MC Dropout 覆盖率（2026-07-29 旧口径，非统一预测协议产物）
    gate_b = None
    picp = None
    try:
        with open(OLD_RECEIPT, "r", encoding="utf-8") as f:
            old = json.load(f)
        mc = old.get("evaluation", old).get("mc_dropout") or {}
        picp = mc.get("picp")
        if picp is not None:
            gate_b = abs(picp - mc.get("nominal_coverage", 0.95)) <= COVERAGE_TOLERANCE
    except Exception:
        pass

    passed = bool(gate_a and gate_b)
    interpretation = (
        "统一预测协议下 LSTM 未超越 WeeklySN 周季节基线（全部 horizon MAE 约为基线 5-10 倍），"
        "基线超越门未通过，LSTM 维持研究态；不得据此声称可部署。"
        if not passed
        else "全部通过，可切训练态。"
    )
    receipt = {
        "pipeline": "semantic_KG_to_GAT_embeddings_to_GNNEnhancedLSTM",
        "recheck_of": "kg_gnn_lstm_validation.json (2026-07-29)",
        "protocol": "UNIFIED_168H_FAIR_V1",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "data_provenance": {
            "data_class": "CALIBRATED_SIMULATED_PRESSURE_TIMESERIES_NOT_RAW_SCADA",
            "unified_forecast_report": os.path.join(EVIDENCE_DIR, "unified_forecast_report_v1.md"),
            "unified_forecast_aggregate_metrics": os.path.join(EVIDENCE_DIR, "unified_forecast_aggregate_metrics_v1.csv"),
            "unified_forecast_verdict": "REPRODUCTION_PASS_WITH_BASELINE_DOMINANCE",
        },
        "evaluation": {
            "status": "research_only_gate_failed" if not passed else "deployment_gate_passed",
            "per_horizon_town_macro": per_horizon,
            "gates": {
                "lstm_beats_weeklysn_all_horizons": bool(gate_a),
                "mc_coverage_within_tolerance": gate_b,
                "coverage_tolerance": COVERAGE_TOLERANCE,
                "mc_picp_legacy_2026_07_29": picp,
            },
            "deployment_gate_passed": passed,
            "interpretation": interpretation,
            "scope_note": "统一协议的 LSTM168 为冻结协议下公平训练的 1xLSTM(32)；注册表 KG/GNN 增强权重未按统一预测协议重训，不在复检范围。",
        },
    }
    out_path = os.path.join(OUT_DIR, "lstm_deployment_gate_recheck_v1.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2)
    print(f"[gate] deployment_gate_passed={passed} (Gate A={gate_a}, Gate B={gate_b})")
    print(f"[gate] 回执: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
