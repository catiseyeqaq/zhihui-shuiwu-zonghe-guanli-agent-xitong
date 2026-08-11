#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""纯 PyTorch 实现的 GraphSAGE 与 GCN(不依赖 torch_geometric)。

管网仅百余节点, 采用全图(full-batch)前向; 在 CPU 上训练即可, 避免 PPU/PyG 兼容风险。
两者均用于节点三级风险分类。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adj_sym(A: torch.Tensor) -> torch.Tensor:
    """对称归一化 D^-1/2 (A+I) D^-1/2 (GCN 用)。"""
    n = A.size(0)
    A_hat = A + torch.eye(n)
    deg = A_hat.sum(1)
    dinv = torch.pow(deg, -0.5)
    dinv[torch.isinf(dinv)] = 0.0
    D = torch.diag(dinv)
    return D @ A_hat @ D


def normalize_adj_mean(A: torch.Tensor) -> torch.Tensor:
    """行归一化(均值聚合器, GraphSAGE 用)。"""
    deg = A.sum(1, keepdim=True)
    deg[deg == 0] = 1.0
    return A / deg


class GCN(nn.Module):
    def __init__(self, in_dim, hid, n_cls, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        dims = [in_dim] + [hid] * (num_layers - 1) + [n_cls]
        self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)])

    def forward(self, X, A_norm):
        H = X
        for i, lin in enumerate(self.lins):
            H = A_norm @ lin(H)
            if i < len(self.lins) - 1:
                H = F.relu(H)
                H = F.dropout(H, self.dropout, self.training)
        return H


class GraphSAGE(nn.Module):
    """均值聚合器 GraphSAGE: h' = W_self(h) + W_neigh(mean_{u in N(v)} h_u)。"""

    def __init__(self, in_dim, hid, n_cls, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.self_lins = nn.ModuleList()
        self.neigh_lins = nn.ModuleList()
        dims = [in_dim] + [hid] * (num_layers - 1) + [n_cls]
        for i in range(num_layers):
            self.self_lins.append(nn.Linear(dims[i], dims[i + 1]))
            self.neigh_lins.append(nn.Linear(dims[i], dims[i + 1]))

    def forward(self, X, A_mean):
        H = X
        n_layers = len(self.self_lins)
        for i in range(n_layers):
            neigh = A_mean @ H
            H = self.self_lins[i](H) + self.neigh_lins[i](neigh)
            if i < n_layers - 1:
                H = F.relu(H)
                H = F.dropout(H, self.dropout, self.training)
        return H


def build_model(name, in_dim, hid, n_cls, num_layers=2, dropout=0.5):
    name = name.lower()
    if name in ("sage", "graphsage"):
        return GraphSAGE(in_dim, hid, n_cls, num_layers, dropout)
    if name == "gcn":
        return GCN(in_dim, hid, n_cls, num_layers, dropout)
    raise ValueError(f"unknown model: {name}")
