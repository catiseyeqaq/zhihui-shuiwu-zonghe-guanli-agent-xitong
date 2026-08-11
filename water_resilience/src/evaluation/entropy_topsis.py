#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""熵权-TOPSIS 综合韧性评价 + 权重敏感性分析。

流程: 方向归一化(正/负向指标统一为"越大越好")-> 熵权法定权 -> TOPSIS 贴近度 Ci ->
韧性分级; 并对权重做扰动, 用 Spearman 秩相关衡量排名稳健性。
"""

import json
import os

import numpy as np
import pandas as pd

import wr_config as C
from indicators.compute import default_path as matrix_path

EPS = 1e-12


def direction_normalize(M, directions):
    Xn = np.zeros_like(M, dtype=float)
    for j in range(M.shape[1]):
        col = M[:, j]
        lo, hi = col.min(), col.max()
        if hi - lo < EPS:
            Xn[:, j] = 0.5
        elif directions[j] == "positive":
            Xn[:, j] = (col - lo) / (hi - lo)
        else:
            Xn[:, j] = (hi - col) / (hi - lo)
    return Xn


def entropy_weights(Xn):
    m = Xn.shape[0]
    P = Xn + EPS
    P = P / P.sum(axis=0, keepdims=True)
    k = 1.0 / np.log(m)
    e = -k * (P * np.log(P)).sum(axis=0)
    d = 1.0 - e
    return d / d.sum()


def topsis(Xn, w):
    V = Xn * w
    a_pos, a_neg = V.max(axis=0), V.min(axis=0)
    d_pos = np.sqrt(((V - a_pos) ** 2).sum(axis=1))
    d_neg = np.sqrt(((V - a_neg) ** 2).sum(axis=1))
    return d_neg / (d_pos + d_neg + EPS)


def grade(ci):
    if ci >= 0.75:
        return "优"
    if ci >= 0.60:
        return "良"
    if ci >= 0.45:
        return "中"
    if ci >= 0.30:
        return "较差"
    return "差"


def _spearman(a, b):
    ra, rb = pd.Series(a).rank().values, pd.Series(b).rank().values
    if np.std(ra) < EPS or np.std(rb) < EPS:
        return 1.0
    return float(np.corrcoef(ra, rb)[0, 1])


def evaluate():
    df = pd.read_csv(matrix_path())
    # 指标体系配置化: indicators.yaml 中 enabled: false 的指标不参与评价
    # (换水司/换指标集只需改 yaml, 代码与管线不动)
    ind = [i for i in C.load_indicators()["indicators"] if i.get("enabled", True)]
    ids = [i["id"] for i in ind]
    directions = [i["direction"] for i in ind]
    dims = {i["id"]: i["dimension"] for i in ind}
    M = df[ids].values.astype(float)

    Xn = direction_normalize(M, directions)
    w = entropy_weights(Xn)
    ci = topsis(Xn, w)

    res = df[["town", "quarter"]].copy()
    res["closeness"] = ci.round(4)
    res["grade"] = [grade(c) for c in ci]
    # 四维得分(各维度归一化指标均值)
    for dim in ["resist", "absorb", "recover", "adapt"]:
        cols = [k for k, d in enumerate(ids) if dims[ids[k]] == dim]
        res[dim] = Xn[:, cols].mean(axis=1).round(4)

    weights = {ids[j]: round(float(w[j]), 5) for j in range(len(ids))}

    # 敏感性: 每个指标权重 ±20%, 重新归一化, 计算 Ci 排名的 Spearman 相关
    sens = []
    for j in range(len(ids)):
        for delta in (0.2, -0.2):
            w2 = w.copy()
            w2[j] *= (1 + delta)
            w2 = w2 / w2.sum()
            ci2 = topsis(Xn, w2)
            sens.append({"indicator": ids[j], "delta": delta,
                         "rank_corr": round(_spearman(ci, ci2), 4)})
    sens_df = pd.DataFrame(sens)
    return res, weights, sens_df


if __name__ == "__main__":
    C.set_seed(C.SEED)
    res, weights, sens = evaluate()
    res.to_csv(os.path.join(C.PATHS["outputs"], "resilience_scores.csv"), index=False)
    with open(os.path.join(C.PATHS["outputs"], "entropy_weights.json"), "w", encoding="utf-8") as f:
        json.dump(weights, f, ensure_ascii=False, indent=2)
    sens.to_csv(os.path.join(C.PATHS["outputs"], "sensitivity.csv"), index=False)
    print("[topsis] top weights:", dict(sorted(weights.items(), key=lambda x: -x[1])[:5]))
    print("[topsis] grade dist:", res["grade"].value_counts().to_dict())
    print("[topsis] min rank_corr under +/-20% weight perturb:", round(sens["rank_corr"].min(), 4))
    print("[topsis] saved resilience_scores.csv + entropy_weights.json + sensitivity.csv")
