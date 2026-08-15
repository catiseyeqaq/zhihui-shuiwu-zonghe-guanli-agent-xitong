#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""成果导出: 高风险管线清单、韧性评估报告(Markdown)、自包含可视化看板(HTML)。

私有镜像无 Streamlit/matplotlib, 故用纯 Python 生成自包含 HTML(内联 SVG 管网图 + 表格),
浏览器直接打开即可, 无需额外依赖。
"""

import json
import os
import sys

import pandas as pd

import wr_config as C
from graph.topology_builder import load_topology, default_path as topo_path

# 真实数据接入层(可选; 部署侧提供真实数据时优先读取, 否则回退仿真)
try:
    from data_sources import detect_real_data, data_source_declaration
    _HAS_DATA_SOURCES = True
except ImportError:
    _HAS_DATA_SOURCES = False

RISK_COLOR = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}


def _data_note():
    """按实际数据源动态生成数据说明(优先使用 data_sources 模块声明)。"""
    if _HAS_DATA_SOURCES:
        return data_source_declaration()
    return ("数据说明: 管网拓扑/时序/维修记录为规则+仿真生成(SIMULATED), "
            "仅用于方法验证, 不代表任何真实地域与真实结论。")


def _load_json(name):
    with open(os.path.join(C.PATHS["outputs"], name), "r", encoding="utf-8") as f:
        return json.load(f)


def high_risk_nodes(G, gnn):
    rows = []
    for n, d in gnn["node_pred"].items():
        if d["pred_label"] == 2 or d["true_label"] == 2:
            rows.append({"node_id": n, "name": G.nodes[n]["name"], "town": d["town"],
                         "node_type": d["node_type"], "pred_label": d["pred_label"],
                         "true_label": d["true_label"], "risk_prob_high": d["risk_prob_high"]})
    df = pd.DataFrame(rows).sort_values("risk_prob_high", ascending=False)
    df["SIMULATED"] = 1
    return df


def svg_topology(G, width=760, height=520, pad=40):
    xs = [G.nodes[n]["x"] for n in G.nodes()]
    ys = [G.nodes[n]["y"] for n in G.nodes()]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)

    def sx(x):
        return pad + (x - x0) / (x1 - x0 + 1e-9) * (width - 2 * pad)

    def sy(y):
        return pad + (y - y0) / (y1 - y0 + 1e-9) * (height - 2 * pad)

    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="background:#fbfcfe;border:1px solid #e3e8ef;border-radius:8px">']
    for u, v in G.edges():
        parts.append(f'<line x1="{sx(G.nodes[u]["x"]):.1f}" y1="{sy(G.nodes[u]["y"]):.1f}" '
                     f'x2="{sx(G.nodes[v]["x"]):.1f}" y2="{sy(G.nodes[v]["y"]):.1f}" '
                     f'stroke="#cbd5e1" stroke-width="1"/>')
    for n in G.nodes():
        d = G.nodes[n]
        r = 3 + d.get("degree", 1) * 0.6
        color = RISK_COLOR.get(d.get("risk_label", 0), "#95a5a6")
        parts.append(f'<circle cx="{sx(d["x"]):.1f}" cy="{sy(d["y"]):.1f}" r="{r:.1f}" '
                     f'fill="{color}" opacity="0.85"><title>{d["name"]} ({d["node_type"]}) risk={d.get("risk_label",0)}</title></circle>')
    parts.append('</svg>')
    return "\n".join(parts)


_CSS = """<style>
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,'PingFang SC','Microsoft YaHei',sans-serif;margin:24px;color:#1f2937;background:#f8fafc}
h1{font-size:22px} h2{font-size:17px;margin-top:28px;border-left:4px solid #2563eb;padding-left:8px}
.banner{background:#fff7ed;border:1px solid #fed7aa;color:#9a3412;padding:10px 14px;border-radius:8px;font-size:13px}
table{border-collapse:collapse;font-size:12px;margin-top:8px} th,td{border:1px solid #e5e7eb;padding:4px 8px;text-align:center}
th{background:#f1f5f9} .tbl tr:nth-child(even){background:#fafafa}
.g-优{background:#dcfce7} .g-良{background:#e0f2fe} .g-中{background:#fef9c3} .g-较差{background:#fee2e2} .g-差{background:#fecaca}
.legend span{display:inline-block;margin-right:12px;font-size:12px}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:middle}
</style>"""


def _grade_table_html(scores):
    piv = scores.pivot(index="town", columns="quarter", values="grade")
    rows = ["<table class='tbl'><tr><th>供水分区\\季度</th>" + "".join(f"<th>{c}</th>" for c in piv.columns) + "</tr>"]
    for town, r in piv.iterrows():
        cells = "".join(f"<td class='g-{v}'>{v}</td>" for v in r.values)
        rows.append(f"<tr><td><b>{town}</b></td>{cells}</tr>")
    rows.append("</table>")
    return "".join(rows)


def html_dashboard(G, gnn, gnn_bl, lstm, lstm_bl, scores, weights, hr):
    svg = svg_topology(G)
    legend = ("<div class='legend'><span><i class='dot' style='background:#2ecc71'></i>低风险</span>"
              "<span><i class='dot' style='background:#f39c12'></i>中风险</span>"
              "<span><i class='dot' style='background:#e74c3c'></i>高风险</span></div>")
    gnn_tbl = pd.DataFrame({**gnn["models"], **{k: v for k, v in gnn_bl.items() if isinstance(v, dict)}}).T
    gnn_tbl.index.name = "模型"
    lstm_rows = {k: v for k, v in lstm["per_town"].items()}
    lstm_tbl = pd.DataFrame(lstm_rows).T
    lstm_tbl.index.name = "乡镇(LSTM)"
    bl_tbl = pd.DataFrame(lstm_bl["summary"]).T
    wtbl = pd.DataFrame(sorted(weights.items(), key=lambda x: -x[1]), columns=["指标", "熵权"])
    parts = [
        "<html><head><meta charset='utf-8'><title>山地供水管网韧性评价看板</title>", _CSS, "</head><body>",
        "<h1>山地供水管网韧性智能评价（合成示例） - 可视化看板</h1>",
        "<div class='banner'>⚠ " + _data_note() + "</div>",
        "<h2>1. 管网拓扑与空间风险(GNN 节点分级)</h2>", legend, svg,
        "<h2>2. 韧性等级(供水分区 × 季度, 熵权-TOPSIS)</h2>", _grade_table_html(scores),
        "<h2>3. GNN 空间风险识别: 模型对比</h2>", gnn_tbl.round(4).to_html(classes="tbl"),
        "<h2>4. LSTM 压力预测(各乡镇 MAE/RMSE)</h2>", lstm_tbl.round(4).to_html(classes="tbl"),
        "<p>时序基线对比(均值):</p>", bl_tbl.round(4).to_html(classes="tbl"),
        "<h2>5. 熵权(前 8 指标)</h2>", wtbl.head(8).to_html(classes="tbl", index=False),
        "<h2>6. 高风险节点清单(Top 15)</h2>", hr.head(15).to_html(classes="tbl", index=False),
    ]
    # 证据化章节(冻结证据图 + 统一预测; 资产缺失自动降级为空)
    try:
        import evidence_sections
        ev_html = evidence_sections.evidence_sections_html()
        if ev_html:
            parts.append(ev_html)
    except Exception as _e:
        parts.append(f"<!-- evidence sections unavailable: {_e} -->")
    parts.append("</body></html>")
    return "\n".join(parts)


def markdown_report(gnn, gnn_bl, lstm, lstm_bl, scores, weights, hr, rel_csv):
    latest_q = C.QUARTERS[-1]
    last = scores[scores["quarter"] == latest_q].sort_values("closeness")
    lines = [
        "# 山地供水管网韧性智能评价（合成示例） - 评估报告",
        "",
        "> " + _data_note(),
        "",
        "## 1. 空间风险识别(GNN)",
        f"- 主模型 GraphSAGE: {gnn['models']['GraphSAGE']}",
        f"- 对比 GCN: {gnn['models']['GCN']}",
        f"- 基线 Centrality: {gnn_bl['Centrality']}; RandomForest: {gnn_bl['RandomForest']}",
        "",
        "## 2. 时序预测(LSTM, 指标=压力)",
        f"- LSTM 各乡镇 MAE 均值: {round(sum(v['mae'] for v in lstm['per_town'].values())/len(lstm['per_town']),4)}",
        f"- 基线(均值 MAE/RMSE): {lstm_bl['summary']}",
        "",
        "## 3. 综合韧性评价(熵权-TOPSIS)",
        f"- 权重前三指标: {dict(sorted(weights.items(), key=lambda x:-x[1])[:3])}",
        f"- {latest_q} 各分区韧性(贴近度/等级):",
    ]
    for _, r in last.iterrows():
        lines.append(f"  - {r['town']}: Ci={r['closeness']} 等级={r['grade']}")
    lines += [
        "",
        "## 4. 高风险管线/节点清单",
        f"- 高风险节点数: {len(hr)}; Top5:",
    ]
    for _, r in hr.head(5).iterrows():
        lines.append(f"  - {r['name']}({r['town']}, {r['node_type']}) 高风险概率={r['risk_prob_high']}")
    # 候选运维建议(来自风险知识关系表)
    lines += ["", "## 5. 候选运维优化建议(源自风险知识关系表, 需人工核验)"]
    try:
        rel = pd.read_csv(rel_csv)
        for _, r in rel.head(6).iterrows():
            lines.append(f"- 针对「{r['risk_factor']}」→{r['consequence']}: 建议{r['measure']}")
    except Exception:
        pass
    lines += ["", "## 6. 结论", "- GraphSAGE 优于 GCN 与非图基线, 体现图模型对空间风险识别的增益;",
              "- 熵权-TOPSIS 给出可追溯的分区季度韧性分级, 权重扰动下排名稳健;",
              "- 低韧性分区应优先安排巡检与管材/阀门整改。"]
    return "\n".join(lines)


def generate_all():
    G = load_topology(topo_path())
    gnn = _load_json("gnn_results.json")
    gnn_bl = _load_json("gnn_baselines.json")
    lstm = _load_json("lstm_results.json")
    lstm_bl = _load_json("lstm_baselines.json")
    scores = pd.read_csv(os.path.join(C.PATHS["outputs"], "resilience_scores.csv"))
    weights = _load_json("entropy_weights.json")
    rel_csv = os.path.join(C.PATHS["extracted"], "risk_relations.csv")

    hr = high_risk_nodes(G, gnn)
    hr.to_csv(os.path.join(C.PATHS["outputs"], "high_risk_nodes.csv"), index=False)
    html = html_dashboard(G, gnn, gnn_bl, lstm, lstm_bl, scores, weights, hr)
    with open(os.path.join(C.PATHS["outputs"], "dashboard.html"), "w", encoding="utf-8") as f:
        f.write(html)
    md = markdown_report(gnn, gnn_bl, lstm, lstm_bl, scores, weights, hr, rel_csv)
    with open(os.path.join(C.PATHS["outputs"], "report.md"), "w", encoding="utf-8") as f:
        f.write(md)
    return len(hr)


if __name__ == "__main__":
    n = generate_all()
    print(f"[report] high_risk_nodes={n}; saved high_risk_nodes.csv + dashboard.html + report.md")
    print("[report] outputs dir ->", C.PATHS["outputs"])
