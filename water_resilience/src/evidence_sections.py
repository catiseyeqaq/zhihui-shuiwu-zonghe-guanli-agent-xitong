#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""证据化看板章节：冻结证据图可视化 + 统一预测结果。

数据来源（只读，绝不修改冻结资产）：
- ${EVIDENCE_ASSET_DIR}/frozen_evidence_graph_v1.json（默认 outputs/evidence_reports）
- .../component_supervision_profile_v1.csv
- .../unified_forecast_aggregate_metrics_v1.csv / weekly_sn_predictions_v1.csv / lstm_predictions_v1.csv
资产缺失时对应章节返回空串，看板其余部分不受影响。
"""
from __future__ import annotations

import csv
import html
import json
import os

_DEFAULT_EVIDENCE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "evidence_reports"
)
EVIDENCE_DIR = os.environ.get("EVIDENCE_ASSET_DIR", _DEFAULT_EVIDENCE_DIR)

_NODE_COLOR = {
    "PLANT": "#3498db",
    "AREA": "#9b59b6",
    "TOWN": "#2c3e50",
    "TANK": "#1abc9c",
    "PIPE": "#95a5a6",
    "MAIN_PIPE": "#7f8c8d",
}
_TYPE_LABEL = {
    "PLANT": "水厂",
    "AREA": "供水片区",
    "TOWN": "乡镇",
    "TANK": "水池",
    "PIPE": "管道",
    "MAIN_PIPE": "干管",
}


def _count_components(nodes: list, edges: list) -> int:
    """简单并查集统计连通分量数。"""
    parent = {n["id"]: n["id"] for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in edges:
        a, b = e.get("src"), e.get("dst")
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    return len({find(n["id"]) for n in nodes})


def _load_frozen_graph() -> dict | None:
    path = os.path.join(EVIDENCE_DIR, "frozen_evidence_graph_v1.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _svg_frozen_graph(graph: dict, width: int = 1060) -> str:
    """按乡镇分列布局, 输出自包含 SVG。"""
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    towns = sorted({n.get("town_scope", "") for n in nodes})
    colw = (width - 120) / max(len(towns), 1)
    pos: dict[str, tuple[float, float]] = {}
    max_rows = 1
    by_town: dict[str, list] = {t: [] for t in towns}
    for n in nodes:
        by_town.setdefault(n.get("town_scope", ""), []).append(n)
    for t, lst in by_town.items():
        max_rows = max(max_rows, len(lst))
    rowh = 30
    top = 70
    height = top + max_rows * rowh + 60
    for ti, t in enumerate(towns):
        x = 60 + ti * colw + colw / 2
        for ri, n in enumerate(by_town.get(t, [])):
            pos[n["id"]] = (x, top + ri * rowh)
    parts = [
        f"<svg viewBox='0 0 {width} {int(height)}' style='max-width:{width}px;background:#fbfcfe;border:1px solid #e3e8ef;border-radius:8px'>"
    ]
    for ti, t in enumerate(towns):
        x = 60 + ti * colw + colw / 2
        parts.append(
            f"<text x='{x:.0f}' y='34' text-anchor='middle' font-size='15' font-weight='bold' fill='#34495e'>{html.escape(t)}</text>"
        )
        parts.append(
            f"<line x1='{60 + ti * colw + 8:.0f}' y1='46' x2='{60 + (ti + 1) * colw - 8:.0f}' y2='46' stroke='#dfe6ee'/>"
        )
    # 边（贝塞尔曲线）
    for e in edges:
        p1, p2 = pos.get(e.get("src")), pos.get(e.get("dst"))
        if not p1 or not p2:
            continue
        mx = (p1[0] + p2[0]) / 2
        my = (p1[1] + p2[1]) / 2 - min(24, abs(p1[0] - p2[0]) * 0.12 + 6)
        tip = "stroke:#e67e22" if e.get("relation_type") in ("outflow_segment", "MAIN_PIPE") else "stroke:#b8c4d0"
        parts.append(
            f"<path d='M{p1[0]:.0f},{p1[1]:.0f} Q{mx:.0f},{my:.0f} {p2[0]:.0f},{p2[1]:.0f}' fill='none' {tip} stroke-width='1.1' opacity='0.75'>"
            f"<title>{html.escape(e.get('src', ''))} —[{html.escape(e.get('relation_type', ''))}]→ {html.escape(e.get('dst', ''))}</title></path>"
        )
    # 节点
    for n in nodes:
        p = pos.get(n["id"])
        if not p:
            continue
        color = _NODE_COLOR.get(n.get("type", ""), "#7f8c8d")
        label = (n.get("name") or n["id"])[:14]
        parts.append(
            f"<circle cx='{p[0]:.0f}' cy='{p[1]:.0f}' r='6.5' fill='{color}' stroke='#fff' stroke-width='1.2'>"
            f"<title>{html.escape(n.get('name', ''))}（{html.escape(_TYPE_LABEL.get(n.get('type', ''), n.get('type', '')))}）"
            f"证据等级: {html.escape(n.get('evidence_grade', ''))}</title></circle>"
        )
        parts.append(
            f"<text x='{p[0] + 10:.0f}' y='{p[1] + 3.5:.0f}' font-size='9.5' fill='#5d6d7e'>{html.escape(label)}</text>"
        )
    # 图例
    lx = 24
    for tname, color in _NODE_COLOR.items():
        parts.append(
            f"<circle cx='{lx + 6}' cy='{height - 24}' r='5' fill='{color}'/>"
            f"<text x='{lx + 15}' y='{height - 20}' font-size='10' fill='#5d6d7e'>{_TYPE_LABEL.get(tname, tname)}</text>"
        )
        lx += 78
    parts.append("</svg>")
    return "".join(parts)


def _component_profile_html() -> str:
    path = os.path.join(EVIDENCE_DIR, "component_supervision_profile_v1.csv")
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return ""
    if not rows:
        return ""
    head = "<table class='tbl'><tr><th>乡镇</th><th>节点</th><th>有候选标签</th><th>独立标签</th><th>弱派生标签</th><th>仅歧义</th><th>无标签</th><th>可用训练节点</th><th>备注</th></tr>"
    body = []
    for r in rows:
        if r.get("town") == "_SUMMARY":
            body.append(
                f"<tr style='background:#fdf2e9;font-weight:bold'><td>合计</td><td>{r.get('nodes', '')}</td><td>{r.get('labeled_nodes', '')}</td>"
                f"<td>{r.get('independent_labeled_nodes', '')}</td><td>{r.get('weak_labeled_nodes', '')}</td><td>{r.get('ambiguous_labeled_nodes', '')}</td>"
                f"<td>{r.get('unlabeled_nodes', '')}</td><td>{r.get('usable_for_training_nodes', '')}</td><td style='font-weight:normal'>{html.escape(r.get('notes', '')[:80])}</td></tr>"
            )
        else:
            body.append(
                f"<tr><td><b>{html.escape(r.get('town', ''))}</b></td><td>{r.get('nodes', '')}</td><td>{r.get('labeled_nodes', '')}</td>"
                f"<td>{r.get('independent_labeled_nodes', '')}</td><td>{r.get('weak_labeled_nodes', '')}</td><td>{r.get('ambiguous_labeled_nodes', '')}</td>"
                f"<td>{r.get('unlabeled_nodes', '')}</td><td>{r.get('usable_for_training_nodes', '')}</td><td>{html.escape(r.get('notes', ''))}</td></tr>"
            )
    return head + "".join(body) + "</table>"


def frozen_graph_section() -> str:
    graph = _load_frozen_graph()
    if not graph:
        return ""
    stats = graph.get("graph_statistics") or {}
    by_town = stats.get("by_town") or {}
    town_rows = "".join(
        f"<tr><td><b>{html.escape(t)}</b></td><td>{v.get('nodes', 0)}</td><td>{v.get('edges', 0)}</td></tr>"
        for t, v in by_town.items()
    )
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    n_nodes = len(nodes)
    n_edges = len(edges)
    n_components = _count_components(nodes, edges)
    not_claims = "".join(f"<li>{html.escape(c)}</li>" for c in (graph.get("not_claims") or []))
    svg = _svg_frozen_graph(graph)
    profile = _component_profile_html()
    profile_block = (
        f"<h3>连通分量监督剖面（标签审计）</h3>{profile}"
        "<p style='color:#b03a2e'>连通分量监督结论：<b>GraphSAGE 节点分类在当前冻结图上不成立</b>"
        "（可用训练节点为 0），GNN 仅保留无监督链路预测用途（MODE_D）。</p>"
        if profile
        else ""
    )
    return (
        f"<h2>冻结证据图（{n_nodes} 节点 / {n_edges} 边 / {n_components} 连通分量）</h2>"
        "<div class='banner' style='background:#eaf2f8;color:#1b4f72'>口径："
        + html.escape(graph.get("identity_statement_zh", ""))
        + f"<br/>冻结于 {html.escape(str(graph.get('frozen_at', '')))} · graph_id: {html.escape(graph.get('graph_id', ''))} · 全部边证据等级 STRUCTURED_SOURCE_INDEXED</div>"
        f"<div style='display:flex;gap:24px;flex-wrap:wrap'><table class='tbl'><tr><th>乡镇</th><th>节点</th><th>边</th></tr>{town_rows}"
        f"<tr><td><b>合计</b></td><td><b>{stats.get('nodes', '')}</b></td><td><b>{stats.get('edges', '')}</b></td></tr></table>"
        f"<div><b>本图不是（not claims）：</b><ul style='margin:6px 0'>{not_claims}</ul></div></div>"
        + svg
        + profile_block
    )


def _load_aggregate() -> list[dict]:
    path = os.path.join(EVIDENCE_DIR, "unified_forecast_aggregate_metrics_v1.csv")
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return [r for r in csv.DictReader(f) if r.get("aggregation") == "town_macro_average"]
    except Exception:
        return []


def _svg_forecast_curves(width: int = 1000, height: int = 260, town: str = "甲镇", horizon: int = 1, max_pts: int = 420) -> str:
    """测试段 y_true vs WeeklySN vs LSTM(seed42) 曲线（默认 甲镇 1h）。"""
    series: dict[str, list] = {"true": [], "wsn": [], "lstm": []}
    try:
        for key, fname, col in (
            ("wsn", "weekly_sn_predictions_v1.csv", "y_pred"),
            ("lstm", "lstm_predictions_v1.csv", "y_pred"),
        ):
            with open(os.path.join(EVIDENCE_DIR, fname), "r", encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    if r.get("town") == town and str(r.get("horizon")) == str(horizon) and r.get("split") == "test" and str(r.get("seed", "42")) == "42":
                        series["true"].append(float(r["y_true"]))
                        series[key].append(float(r[col]))
    except Exception:
        return ""
    n = min(len(series["true"]), len(series["wsn"]), len(series["lstm"]))
    if n < 50:
        return ""
    step = max(1, n // max_pts)
    sel = lambda arr: arr[::step][:max_pts]
    t, w, l = sel(series["true"]), sel(series["wsn"]), sel(series["lstm"])
    lo = min(min(t), min(w), min(l))
    hi = max(max(t), max(w), max(l))
    pad = (hi - lo) * 0.08 + 1e-6
    lo, hi = lo - pad, hi + pad
    m = len(t)

    def pts(arr):
        return " ".join(
            f"{20 + i * (width - 40) / max(m - 1, 1):.1f},{height - 30 - (v - lo) / (hi - lo) * (height - 60):.1f}"
            for i, v in enumerate(arr)
        )

    return (
        f"<svg viewBox='0 0 {width} {height}' style='max-width:{width}px;background:#fff;border:1px solid #e3e8ef;border-radius:8px'>"
        f"<text x='20' y='18' font-size='12' fill='#34495e'>{html.escape(town)} · horizon={horizon}h · 测试段（每 {step} 点采样）</text>"
        f"<polyline points='{pts(t)}' fill='none' stroke='#2c3e50' stroke-width='1.6'/>"
        f"<polyline points='{pts(w)}' fill='none' stroke='#27ae60' stroke-width='1.2' stroke-dasharray='5,3'/>"
        f"<polyline points='{pts(l)}' fill='none' stroke='#e74c3c' stroke-width='1.0' opacity='0.8'/>"
        "<g font-size='11'><rect x='" + str(width - 250) + "' y='30' width='230' height='58' fill='#fff' opacity='0.85'/>"
        "<line x1='" + str(width - 240) + "' y1='44' x2='" + str(width - 210) + "' y2='44' stroke='#2c3e50' stroke-width='2'/><text x='" + str(width - 204) + "' y='48' fill='#34495e'>真值 y_true</text>"
        "<line x1='" + str(width - 240) + "' y1='62' x2='" + str(width - 210) + "' y2='62' stroke='#27ae60' stroke-width='2' stroke-dasharray='5,3'/><text x='" + str(width - 204) + "' y='66' fill='#34495e'>WeeklySN（主导基线）</text>"
        "<line x1='" + str(width - 240) + "' y1='80' x2='" + str(width - 210) + "' y2='80' stroke='#e74c3c' stroke-width='2'/><text x='" + str(width - 204) + "' y='84' fill='#34495e'>LSTM168</text></g>"
        "</svg>"
    )


def forecast_section() -> str:
    rows = _load_aggregate()
    if not rows:
        return ""
    horizons = sorted({int(r["horizon_h"]) for r in rows})
    models = ["WeeklySN", "Ridge168", "LSTM168"]
    head = "<table class='tbl'><tr><th>horizon</th>" + "".join(f"<th>{m} MAE</th>" for m in models) + "".join(f"<th>{m} RMSE</th>" for m in models) + "<th>裁决</th></tr>"
    body = []
    verdict = "WeeklySN 主导（基线未被超越）"
    for h in horizons:
        cells = {m["model"]: m for m in rows if int(m["horizon_h"]) == h}
        best = min(models, key=lambda m: float(cells.get(m, {"mae": 9e9}).get("mae", 9e9)))
        tds_parts = []
        for m in models:
            style = ' style="font-weight:bold;color:#1e8449"' if m == best else ''
            mae = float(cells.get(m, {"mae": float("nan")}).get("mae", float("nan")))
            tds_parts.append(f"<td{style}>{mae:.6f}</td>")
        tds = "".join(tds_parts)
        tds += "".join(f"<td>{float(cells.get(m, {'rmse': float('nan')}).get('rmse', float('nan'))):.6f}</td>" for m in models)
        body.append(f"<tr><td>{h}h</td>{tds}<td>{best}</td></tr>")
    table = head + "".join(body) + "</table>"
    curves = _svg_forecast_curves()
    gate_dir = os.environ.get("WR_OUTPUTS_DIR") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    gate_path = os.path.join(gate_dir, "lstm_deployment_gate_recheck_v1.json")
    gate_block = ""
    try:
        with open(gate_path, "r", encoding="utf-8") as f:
            gate = json.load(f)
        passed = gate.get("deployment_gate_passed")
        gate_block = (
            f"<div class='banner' style='background:{'#e8f8f0' if passed else '#fdedec'};color:{'#145a32' if passed else '#922b21'}'>"
            f"LSTM 部署门复检（统一预测协议）：<b>{'通过 — 已切训练态' if passed else '未通过 — 维持研究态'}</b>"
            f"<br/>{html.escape(gate.get('interpretation', ''))}</div>"
        )
    except Exception:
        pass
    return (
        "<h2>统一 168h 压力预测（协议 UNIFIED_168H_FAIR_V1）</h2>"
        "<div class='banner' style='background:#eaf2f8;color:#1b4f72'>数据口径：校准仿真压力时序（SIMULATED），5 乡镇单变量 · 70/15/15 时间序切分 · direct 1h/6h/24h · "
        f"裁决：<b>{html.escape(verdict)}</b>，详见 unified_forecast_report_v1</div>"
        + table
        + (f"<h3>测试段预测曲线（示例）</h3>{curves}" if curves else "")
        + gate_block
    )


def evidence_sections_html() -> str:
    """供 report.html_dashboard 追加的证据化章节（资产缺失自动降级为空）。"""
    parts = [s for s in (frozen_graph_section(), forecast_section()) if s]
    if not parts:
        return ""
    return "<hr/>" + "\n".join(parts)


if __name__ == "__main__":
    out = evidence_sections_html()
    print(f"[evidence_sections] 生成 {len(out)} 字符; frozen={bool(frozen_graph_section())} forecast={bool(forecast_section())}")
