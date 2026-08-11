#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""节点风险标注(山地爆管多因子风险评分公式, SIMULATED 标签)。

风险评分 = w1*高程 + w2*管龄 + w3*材质风险 + w4*介数中心性 + w5*历史维修
各因子经 min-max 归一化到 [0,1]; 三级标签(0低/1中/2高)按分数 34%/67% 分位切分。
该规则标签仅用于 GNN 监督学习的方法验证, 非真实事故标签。
"""

import csv
import os

import numpy as np

import wr_config as C
from graph.topology_builder import load_topology, save_topology, default_path

# 多因子权重(可在论文中作为可解释的风险公式)
WEIGHTS = {"elevation": 0.20, "age": 0.25, "mrisk": 0.20, "betweenness": 0.15, "repair": 0.20}


def _minmax(vals):
    lo, hi = min(vals), max(vals)
    return [(v - lo) / (hi - lo) if hi > lo else 0.0 for v in vals]


def compute_risk(G):
    nodes = list(G.nodes())
    elev = _minmax([G.nodes[n]["elevation"] for n in nodes])
    age = _minmax([G.nodes[n]["inc_pipe_avg_age"] for n in nodes])
    mrisk = [G.nodes[n]["inc_pipe_avg_mrisk"] for n in nodes]
    btw = _minmax([G.nodes[n]["betweenness"] for n in nodes])
    rep = _minmax([G.nodes[n]["inc_repair_sum"] for n in nodes])
    scores = {}
    for i, n in enumerate(nodes):
        s = (WEIGHTS["elevation"] * elev[i] + WEIGHTS["age"] * age[i]
             + WEIGHTS["mrisk"] * mrisk[i] + WEIGHTS["betweenness"] * btw[i]
             + WEIGHTS["repair"] * rep[i])
        scores[n] = round(float(s), 4)
    return scores


def to_labels(scores):
    vals = np.array(list(scores.values()))
    q1, q2 = np.quantile(vals, [0.34, 0.67])
    return {n: (2 if s >= q2 else (1 if s >= q1 else 0)) for n, s in scores.items()}


def label_topology(G):
    scores = compute_risk(G)
    labels = to_labels(scores)
    for n in G.nodes():
        G.nodes[n]["risk_score"] = scores[n]
        G.nodes[n]["risk_label"] = int(labels[n])
    return scores, labels


def save_labels(G, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "name", "node_type", "town", "risk_score", "risk_label", "SIMULATED"])
        for n in G.nodes():
            d = G.nodes[n]
            w.writerow([n, d["name"], d["node_type"], d["town"], d["risk_score"], d["risk_label"], 1])


if __name__ == "__main__":
    C.set_seed(C.SEED)
    G = load_topology(default_path())
    scores, labels = label_topology(G)
    save_topology(G, default_path())
    save_labels(G, os.path.join(C.PATHS["generated"], "node_labels.csv"))
    from collections import Counter
    print("[risk] label distribution (0低/1中/2高):", dict(Counter(labels.values())))
    print("[risk] saved node_labels.csv + annotated topology")
