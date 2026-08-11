#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM 辅助的风险知识关系表(风险因素-影响对象-风险后果-处置措施)。

默认使用领域专家整理的种子关系(离线可靠); 传 --use-llm 时调用 PPU 上的 Qwen 服务
从工程文档/KG 草案补充候选关系(需人工核验)。输出 CSV + NetworkX 关系图。
"""

import csv
import json
import os
import urllib.request

import networkx as nx

import wr_config as C

# 领域专家整理的种子关系(山地乡村供水运维), 每条: 风险因素/影响对象/风险后果/处置措施
SEED_RELATIONS = [
    ("管材老化", "老旧管段", "爆管漏损", "更换管材并加密巡检"),
    ("铸铁/PVC管材", "管段", "腐蚀与脆裂", "优选球墨铸铁或PE管"),
    ("山地高程落差大", "高区管段", "压力不足或水锤", "分区调压并增设减压阀"),
    ("压力长期波动", "管网整体", "疲劳破坏", "稳压运行并加装压力监测"),
    ("阀门失效", "隔离能力", "故障扩散范围扩大", "定期维护并增设检修阀"),
    ("水质在线监测缺失", "供水水质", "水质超标未及时发现", "增设水质监测与自动加药"),
    ("巡检制度不完善", "运维管理", "故障发现延迟", "完善巡检制度与责任到人"),
    ("供水冗余不足", "单一供水路径", "停水影响范围大", "增设联络管实现环状供水"),
    ("清水池水位异常", "供水连续性", "间歇停水", "水位在线监测联动补水"),
    ("山区交通不便", "抢修响应", "恢复时间偏长", "预置抢修物资与就近队伍"),
    ("冻融或暴雨", "管段与边坡", "管道破裂", "增加埋深保温与边坡防护"),
    ("监测点覆盖不足", "运行状态感知", "异常漏报", "优化压力流量监测点布设"),
    ("人员培训不足", "运维能力", "处置不规范", "定期开展技能培训与演练"),
    ("重复故障未整改", "薄弱管段", "反复停水", "建立重复故障台账并专项整改"),
    ("加药消毒不稳定", "出厂水质", "余氯不达标", "加药设备自动化与冗余"),
]


def _llm_extract(text, settings):
    """调用 PPU 上 Qwen(OpenAI 兼容 API)抽取候选关系; 失败返回空列表。"""
    llm = settings["llm"]
    prompt = (
        "你是供水管网运维风险分析专家。请从下面文本中抽取风险知识四元组, "
        "每条包含: 风险因素, 影响对象, 风险后果, 处置措施。"
        "只输出JSON数组, 形如[{\"risk_factor\":\"\",\"affected_object\":\"\",\"consequence\":\"\",\"measure\":\"\"}]。\n\n"
        + text[:4000]
    )
    payload = {
        "model": llm["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2, "max_tokens": 1200,
    }
    req = urllib.request.Request(
        llm["endpoint"].rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=llm.get("timeout", 180)) as resp:
            data = json.loads(resp.read())
        content = data["choices"][0]["message"]["content"]
        start, end = content.find("["), content.rfind("]")
        arr = json.loads(content[start:end + 1])
        return [(d.get("risk_factor", ""), d.get("affected_object", ""),
                 d.get("consequence", ""), d.get("measure", "")) for d in arr]
    except Exception as e:
        print(f"[extract] LLM 调用失败, 仅用种子关系: {e}")
        return []


def build_relations(use_llm=False, settings=None):
    settings = settings or C.SETTINGS
    relations = [(*r, "seed") for r in SEED_RELATIONS]
    if use_llm:
        kg_draft = settings["paths"].get("kg_draft")
        text = ""
        if kg_draft and os.path.exists(kg_draft):
            text = open(kg_draft, "r", encoding="utf-8").read()
        for r in _llm_extract(text, settings):
            if all(r):
                relations.append((*r, "llm"))
    return relations


def to_graph(relations):
    G = nx.DiGraph()
    for factor, obj, cons, measure, src in relations:
        for name, ntype in [(factor, "风险因素"), (obj, "影响对象"),
                            (cons, "风险后果"), (measure, "处置措施")]:
            if name and not G.has_node(name):
                G.add_node(name, node_type=ntype)
        if factor and obj:
            G.add_edge(factor, obj, relation="影响", source=src)
        if obj and cons:
            G.add_edge(obj, cons, relation="导致", source=src)
        if cons and measure:
            G.add_edge(cons, measure, relation="处置", source=src)
    return G


def save(relations, G):
    os.makedirs(C.PATHS["extracted"], exist_ok=True)
    csv_path = os.path.join(C.PATHS["extracted"], "risk_relations.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["risk_factor", "affected_object", "consequence", "measure", "source", "verified"])
        for r in relations:
            w.writerow([*r, "pending"])  # verified=pending, 需人工核验
    with open(os.path.join(C.PATHS["extracted"], "risk_relation_graph.json"), "w", encoding="utf-8") as f:
        json.dump(nx.node_link_data(G), f, ensure_ascii=False, indent=2)
    return csv_path


if __name__ == "__main__":
    import sys
    use_llm = "--use-llm" in sys.argv
    rels = build_relations(use_llm=use_llm)
    G = to_graph(rels)
    path = save(rels, G)
    print(f"[extract] relations={len(rels)} (use_llm={use_llm}) graph nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
    print("[extract] 需人工核验(verified=pending); saved ->", path)
