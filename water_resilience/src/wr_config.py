#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全局配置与路径加载(供 water_resilience 各模块复用)。

使用 Python 隐式命名空间包: 入口脚本(pipeline/dashboard)把本目录(src)加入 sys.path,
其余模块以 `from graph.xxx import yyy` 方式互相引用, 无需 __init__.py。
"""

import os
import random

import yaml

# 目录锚点: .../water_resilience
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(_SRC_DIR)            # water_resilience/
WORKSPACE_DIR = os.path.dirname(PROJECT_DIR)       # public repository root
CONFIG_DIR = os.path.join(PROJECT_DIR, "config")


def _abs(path: str) -> str:
    """把配置里相对工作区根目录的路径转成绝对路径。"""
    if os.path.isabs(path):
        return path
    return os.path.join(WORKSPACE_DIR, path)


def load_settings() -> dict:
    with open(os.path.join(CONFIG_DIR, "settings.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # 规整常用路径为绝对路径
    paths = cfg.setdefault("paths", {})
    for k, v in list(paths.items()):
        paths[k] = _abs(v)
    return cfg


def load_indicators() -> dict:
    with open(os.path.join(CONFIG_DIR, "indicators.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


def ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# 便捷单例
SETTINGS = load_settings()
PATHS = SETTINGS["paths"]
TOWNS = SETTINGS["towns"]
QUARTERS = SETTINGS["quarters"]
SEED = int(SETTINGS.get("random_seed", 42))

ensure_dirs(PATHS["generated"], PATHS["extracted"], PATHS["outputs"])


if __name__ == "__main__":
    import json
    print("PROJECT_DIR =", PROJECT_DIR)
    print("towns =", TOWNS)
    print("quarters =", QUARTERS)
    print("paths =", json.dumps(PATHS, ensure_ascii=False, indent=2))
    print("indicators =", len(load_indicators()["indicators"]))
