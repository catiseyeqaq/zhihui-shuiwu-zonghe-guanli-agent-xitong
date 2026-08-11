#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GNN 对照基线: 图中心性加权评分基线 + RandomForest(非图)基线。

用于与 GraphSAGE/GCN 在同一数据划分上对比, 体现图模型对空间风险识别的增益。
"""

import json
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

import wr_config as C
from graph.topology_builder import load_topology, default_path
from models.gnn_train import build_tensors, split_masks


def centrality_baseline(G, y, masks):
    """以介数中心性作为风险代理, 按分位切三级。"""
    nodes = list(G.nodes())
    btw = np.array([G.nodes[n]["betweenness"] for n in nodes])
    q1, q2 = np.quantile(btw, [0.34, 0.67])
    pred = np.where(btw >= q2, 2, np.where(btw >= q1, 1, 0))
    te = masks[2].numpy()
    return {
        "test_acc": round(float(accuracy_score(y.numpy()[te], pred[te])), 4),
        "test_macro_f1": round(float(f1_score(y.numpy()[te], pred[te], average="macro")), 4),
    }


def rf_baseline(X, y, masks, seed=42):
    tr, te = masks[0].numpy(), masks[2].numpy()
    Xn = X.numpy()
    clf = RandomForestClassifier(n_estimators=200, random_state=seed)
    clf.fit(Xn[tr], y.numpy()[tr])
    pred = clf.predict(Xn[te])
    return {
        "test_acc": round(float(accuracy_score(y.numpy()[te], pred)), 4),
        "test_macro_f1": round(float(f1_score(y.numpy()[te], pred, average="macro")), 4),
    }


def run_baselines(seed=42):
    G = load_topology(default_path())
    nodes, X, y, A = build_tensors(G)
    masks = split_masks(len(nodes), seed=seed)
    return {
        "Centrality": centrality_baseline(G, y, masks),
        "RandomForest": rf_baseline(X, y, masks, seed=seed),
        "SIMULATED": True,
    }


if __name__ == "__main__":
    C.set_seed(C.SEED)
    res = run_baselines(seed=C.SEED)
    out = os.path.join(C.PATHS["outputs"], "gnn_baselines.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("[baseline] Centrality  :", res["Centrality"])
    print("[baseline] RandomForest:", res["RandomForest"])
    print("[baseline] saved ->", out)
