#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""物理管网拓扑生成器。

根据公开配置中的抽象规模和参数，规则化生成供水管网物理图：
节点包括水源、水厂、清水池、交汇点、阀门、监测点和需求点，
边包括管径、管长、材质、管龄和维修次数等演示属性。
拓扑、坐标和高程均由固定随机种子生成，仅用于方法演示。
"""

import json
import math
import os
import random

import networkx as nx

import wr_config as C

NODE_TYPES = ["source", "plant", "reservoir", "junction", "valve", "monitor", "demand"]


def _elev(rng, base):
    return round(base + rng.uniform(-40, 60), 1)


def _pipe_attrs(rng, topo, length):
    material = rng.choices(topo["materials"], weights=[0.30, 0.15, 0.25, 0.20, 0.10], k=1)[0]
    age = rng.randint(1, 30)
    mrisk = topo["material_risk"][material]
    repair = int(round((age / 30.0) * 3 + mrisk * 4 + rng.uniform(0, 1.5)))
    return {"material": material, "material_risk": mrisk, "age": age,
            "length_m": round(length, 1),
            "diameter_mm": rng.choice([100, 150, 200, 300, 400, 600]),
            "repair_count": max(0, repair)}


def build_topology(settings=None):
    settings = settings or C.SETTINGS
    topo = settings["topology"]
    towns = settings["towns"]
    rng = random.Random(int(settings.get("random_seed", 42)))
    elo, ehi = topo["elevation_range_m"]

    G = nx.Graph()
    G.add_node("SRC", name="总水源", node_type="source", town="全域",
               elevation=ehi, x=0.0, y=0.0, base_demand=0.0)

    center, base_elev = {}, {}
    for i, t in enumerate(towns):
        center[t] = ((i % 3) * 30.0, (i // 3) * 30.0)
        base_elev[t] = rng.uniform(elo, ehi - 60)

    plants = []
    for p in range(topo["n_water_plants"]):
        t = towns[p % len(towns)]
        cx, cy = center[t]
        nid = f"PLANT{p:02d}"
        G.add_node(nid, name=f"{t}水厂{p:02d}", node_type="plant", town=t,
                   elevation=_elev(rng, base_elev[t] + 40),
                   x=cx + rng.uniform(-6, 6), y=cy + rng.uniform(-6, 6), base_demand=0.0)
        plants.append(nid)

    def scatter(t, spread=10.0):
        cx, cy = center[t]
        return cx + rng.uniform(-spread, spread), cy + rng.uniform(-spread, spread)

    def dist(a, b):
        return math.hypot(G.nodes[a]["x"] - G.nodes[b]["x"], G.nodes[a]["y"] - G.nodes[b]["y"]) * 100 + 20

    def pipe(a, b):
        if a != b and not G.has_edge(a, b):
            G.add_edge(a, b, **_pipe_attrs(rng, topo, dist(a, b)))

    for nid in plants:
        pipe("SRC", nid)

    reservoirs = []
    for t in towns:
        res = f"RES_{t}"
        rx, ry = scatter(t, 4)
        G.add_node(res, name=f"{t}清水池", node_type="reservoir", town=t,
                   elevation=_elev(rng, base_elev[t] + 20), x=rx, y=ry, base_demand=0.0)
        reservoirs.append(res)
        for nid in [p for p in plants if G.nodes[p]["town"] == t]:
            pipe(nid, res)

        junctions = []
        for j in range(topo["junctions_per_town"]):
            jx, jy = scatter(t)
            nid = f"J_{t}_{j}"
            G.add_node(nid, name=f"{t}交汇{j}", node_type="junction", town=t,
                       elevation=_elev(rng, base_elev[t]), x=jx, y=jy,
                       base_demand=round(rng.uniform(5, 20), 1))
            junctions.append(nid)

        core = [res] + junctions
        connected = [res]
        for nid in junctions:
            pipe(rng.choice(connected), nid)
            connected.append(nid)
        for _ in range(max(1, len(junctions) // 3)):
            a, b = rng.sample(core, 2)
            pipe(a, b)

        for v in range(topo["valves_per_town"]):
            vx, vy = scatter(t)
            nid = f"V_{t}_{v}"
            G.add_node(nid, name=f"{t}阀门{v}", node_type="valve", town=t,
                       elevation=_elev(rng, base_elev[t]), x=vx, y=vy, base_demand=0.0)
            pipe(rng.choice(junctions), nid)
        for m in range(topo["monitors_per_town"]):
            mx, my = scatter(t)
            nid = f"M_{t}_{m}"
            G.add_node(nid, name=f"{t}监测点{m}", node_type="monitor", town=t,
                       elevation=_elev(rng, base_elev[t]), x=mx, y=my, base_demand=0.0)
            pipe(rng.choice(junctions), nid)
        for d in range(topo["demand_nodes_per_town"]):
            dx, dy = scatter(t, 12)
            nid = f"D_{t}_{d}"
            G.add_node(nid, name=f"{t}需求点{d}", node_type="demand", town=t,
                       elevation=_elev(rng, base_elev[t] - 20), x=dx, y=dy,
                       base_demand=round(rng.uniform(10, 40), 1))
            pipe(rng.choice(junctions), nid)

    for i in range(len(reservoirs)):
        pipe(reservoirs[i], reservoirs[(i + 1) % len(reservoirs)])

    _annotate(G)
    G.graph["SIMULATED"] = True
    return G


def _annotate(G):
    deg = dict(G.degree())
    btw = nx.betweenness_centrality(G)
    clo = nx.closeness_centrality(G)
    for n in G.nodes():
        G.nodes[n]["degree"] = deg[n]
        G.nodes[n]["betweenness"] = round(btw[n], 6)
        G.nodes[n]["closeness"] = round(clo[n], 6)
        ages, mrisks, dias = [], [], []
        for _, _, e in G.edges(n, data=True):
            ages.append(e["age"]); mrisks.append(e["material_risk"]); dias.append(e["diameter_mm"])
        G.nodes[n]["inc_pipe_avg_age"] = round(sum(ages) / len(ages), 2) if ages else 0.0
        G.nodes[n]["inc_pipe_avg_mrisk"] = round(sum(mrisks) / len(mrisks), 4) if mrisks else 0.0
        G.nodes[n]["inc_pipe_min_diameter"] = min(dias) if dias else 0
        G.nodes[n]["inc_repair_sum"] = sum(e["repair_count"] for _, _, e in G.edges(n, data=True))


def save_topology(G, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nx.node_link_data(G), f, ensure_ascii=False, indent=2)


def load_topology(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 兼容不同 networkx 版本的边键: 3.0 用 "links", 3.4+ node_link_data 写 "edges"
    if isinstance(data, dict) and "links" not in data and "edges" in data:
        data["links"] = data["edges"]
    return nx.node_link_graph(data)


def default_path():
    return os.path.join(C.PATHS["generated"], "topology.json")


if __name__ == "__main__":
    C.set_seed(C.SEED)
    G = build_topology()
    save_topology(G, default_path())
    from collections import Counter
    types = Counter(nx.get_node_attributes(G, "node_type").values())
    print(f"[topology] nodes={G.number_of_nodes()} edges={G.number_of_edges()} source=SIMULATED")
    print("[topology] types:", dict(types))
    print("[topology] saved ->", default_path())
