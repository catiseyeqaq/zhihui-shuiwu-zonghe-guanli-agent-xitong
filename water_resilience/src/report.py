#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出公开版水务智能体的合成数据分析摘要。"""

import json
import html
import os

import pandas as pd

import wr_config as C


def _json(name: str) -> dict:
    path = os.path.join(C.PATHS["outputs"], name)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _csv(name: str) -> pd.DataFrame:
    path = os.path.join(C.PATHS["outputs"], name)
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def build_report() -> str:
    scores = _csv("resilience_scores.csv")
    risks = _csv("high_risk_nodes.csv")
    gnn = _json("gnn_results.json")
    lstm = _json("lstm_results.json")
    weights = _json("entropy_weights.json")
    lines = [
        "# 智慧水务综合管理 Agent 分析报告",
        "",
        "> 数据边界：本公开示例默认使用合成拓扑、时序、维修和管理记录，仅用于方法验证。",
        "> 结果不代表任何真实水务系统，也不直接生成生产调度或阀门控制指令。",
        "",
    ]
    if not scores.empty:
        lines += ["## 分区韧性评价", "", "| 分区 | 周期 | 综合得分 | 等级 |", "|---|---|---:|---|"]
        for _, row in scores.head(20).iterrows():
            lines.append(
                f"| {row.get('town', '')} | {row.get('quarter', '')} | "
                f"{row.get('closeness', row.get('score', ''))} | {row.get('grade', '')} |"
            )
        lines.append("")
    if not risks.empty:
        lines += ["## 高风险节点", "", "| 节点 | 分区 | 类型 | 高风险概率 |", "|---|---|---|---:|"]
        for _, row in risks.head(15).iterrows():
            lines.append(
                f"| {row.get('name', row.get('node_id', ''))} | {row.get('town', '')} | "
                f"{row.get('node_type', '')} | {row.get('risk_prob_high', '')} |"
            )
        lines.append("")
    if gnn:
        lines += ["## GNN 空间风险识别", "", f"```json\n{json.dumps(gnn, ensure_ascii=False, indent=2, default=str)}\n```", ""]
    if lstm:
        lines += ["## LSTM 时序预测", "", f"```json\n{json.dumps(lstm, ensure_ascii=False, indent=2, default=str)}\n```", ""]
    if weights:
        lines += ["## 熵权结果", "", f"```json\n{json.dumps(weights, ensure_ascii=False, indent=2)}\n```", ""]
    return "\n".join(lines)


def main() -> None:
    generate_all()


def generate_all() -> int:
    """生成 Markdown 摘要和无需额外前端依赖的本地 HTML 看板。"""
    os.makedirs(C.PATHS["outputs"], exist_ok=True)
    path = os.path.join(C.PATHS["outputs"], "report.md")
    report = build_report()
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    dashboard_path = os.path.join(C.PATHS["outputs"], "dashboard.html")
    page = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>智慧水务综合管理 Agent</title>
<style>body{margin:0;background:#f5f7fb;color:#172033;font-family:system-ui,-apple-system,"Microsoft YaHei",sans-serif}
main{max-width:980px;margin:40px auto;padding:0 20px}section{background:#fff;border-radius:14px;padding:24px;box-shadow:0 8px 30px #1e2d4a12}
h1{margin-top:0;color:#1769aa}pre{white-space:pre-wrap;line-height:1.7;font-family:inherit}</style></head>
<body><main><section><h1>智慧水务综合管理 Agent</h1>
<p>本看板由公开示例流水线生成，默认输入均为合成数据。</p>
<pre>""" + html.escape(report) + """</pre></section></main></body></html>
"""
    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"[report] saved -> {path}")
    print(f"[report] dashboard -> {dashboard_path}")
    risks = _csv("high_risk_nodes.csv")
    return len(risks)


if __name__ == "__main__":
    main()
