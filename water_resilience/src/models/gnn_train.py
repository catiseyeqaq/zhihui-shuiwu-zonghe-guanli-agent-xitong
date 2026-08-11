#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GNN 空间风险识别训练/评估(GraphSAGE 主模型, GCN 对比)。

从标注后的管网拓扑构建节点特征与邻接矩阵, 三级风险分类; CPU 全图训练。
输出各模型 accuracy/macro-F1 与主模型(GraphSAGE)逐节点预测风险, 供指标聚合使用。
"""

import csv
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

import wr_config as C
from graph.topology_builder import load_topology, default_path
from graph.topology_builder import NODE_TYPES
from models.gnn_models import build_model, normalize_adj_sym, normalize_adj_mean

NUM_FEATS = ["elevation", "degree", "betweenness", "closeness", "inc_pipe_avg_age",
             "inc_pipe_avg_mrisk", "inc_pipe_min_diameter", "inc_repair_sum", "base_demand"]


def build_tensors(G):
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    # 数值特征标准化
    num = np.array([[G.nodes[n][k] for k in NUM_FEATS] for n in nodes], dtype=np.float32)
    num = (num - num.mean(0)) / (num.std(0) + 1e-6)
    # 节点类型 one-hot
    onehot = np.zeros((len(nodes), len(NODE_TYPES)), dtype=np.float32)
    for i, n in enumerate(nodes):
        onehot[i, NODE_TYPES.index(G.nodes[n]["node_type"])] = 1.0
    X = torch.tensor(np.concatenate([num, onehot], axis=1))
    y = torch.tensor([G.nodes[n]["risk_label"] for n in nodes], dtype=torch.long)
    A = torch.zeros((len(nodes), len(nodes)), dtype=torch.float32)
    for u, v in G.edges():
        A[idx[u], idx[v]] = 1.0
        A[idx[v], idx[u]] = 1.0
    return nodes, X, y, A


def split_masks(n, seed=42, ratios=(0.6, 0.2)):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tr, n_va = int(n * ratios[0]), int(n * ratios[1])
    tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
    m = lambda ids: torch.tensor(np.isin(np.arange(n), ids))
    return m(tr), m(va), m(te)


def train_one(name, X, y, A_input, masks, epochs=250, hid=32, lr=0.01, wd=5e-4, seed=42):
    torch.manual_seed(seed)
    tr, va, te = masks
    model = build_model(name, X.size(1), hid, int(y.max()) + 1, num_layers=2, dropout=0.5)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_va, best_state = -1.0, None
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        out = model(X, A_input)
        loss = F.cross_entropy(out[tr], y[tr])
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(X, A_input).argmax(1)
            va_acc = accuracy_score(y[va], pred[va])
        if va_acc > best_va:
            best_va, best_state = va_acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(X, A_input)
        prob = F.softmax(logits, 1)
        pred = logits.argmax(1)
    metrics = {
        "test_acc": round(float(accuracy_score(y[te], pred[te])), 4),
        "test_macro_f1": round(float(f1_score(y[te], pred[te], average="macro")), 4),
        "val_acc": round(float(best_va), 4),
    }
    return metrics, pred, prob


def run_gnn(seed=42):
    G = load_topology(default_path())
    nodes, X, y, A = build_tensors(G)
    masks = split_masks(len(nodes), seed=seed)
    A_sym, A_mean = normalize_adj_sym(A), normalize_adj_mean(A)
    results = {"models": {}, "SIMULATED": True}
    sage_m, sage_pred, sage_prob = train_one("sage", X, y, A_mean, masks, seed=seed)
    gcn_m, gcn_pred, gcn_prob = train_one("gcn", X, y, A_sym, masks, seed=seed)
    results["models"]["GraphSAGE"] = sage_m
    results["models"]["GCN"] = gcn_m
    # 主模型(GraphSAGE)逐节点预测
    node_pred = {}
    for i, n in enumerate(nodes):
        node_pred[n] = {
            "name": G.nodes[n].get("name", n),
            "town": G.nodes[n]["town"], "node_type": G.nodes[n]["node_type"],
            "true_label": int(y[i]), "pred_label": int(sage_pred[i]),
            "risk_prob_high": round(float(sage_prob[i, 2]), 4),
        }
    results["node_pred"] = node_pred
    return results


if __name__ == "__main__":
    C.set_seed(C.SEED)
    res = run_gnn(seed=C.SEED)
    out = os.path.join(C.PATHS["outputs"], "gnn_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    high_risk_path = os.path.join(C.PATHS["outputs"], "high_risk_nodes.csv")
    high_risk = sorted(
        ((node_id, row) for node_id, row in res["node_pred"].items() if row["pred_label"] == 2),
        key=lambda item: item[1]["risk_prob_high"],
        reverse=True,
    )
    with open(high_risk_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "name", "town", "node_type", "pred_label", "risk_prob_high", "SIMULATED"])
        for node_id, row in high_risk:
            writer.writerow([
                node_id, row["name"], row["town"], row["node_type"],
                row["pred_label"], row["risk_prob_high"], 1,
            ])
    print("[gnn] GraphSAGE:", res["models"]["GraphSAGE"])
    print("[gnn] GCN      :", res["models"]["GCN"])
    print("[gnn] saved ->", out, "+", high_risk_path)
