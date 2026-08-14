"""面向公开示例的智慧水务 Agent 接口。

该模块只负责编排合成数据分析流水线、读取分析结果和生成建议性清单，
不连接生产系统，不执行阀门控制、调度或维修派单。
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


class WaterManagementAgent:
    """智慧水务综合分析 Agent。

    默认以当前仓库为工作根目录，也可以通过 ``WATER_AGENT_ROOT`` 指定一个
    本地副本。该环境变量只改变本地文件位置，不包含凭据或远程连接配置。
    """

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        default_root = Path(__file__).resolve().parents[1]
        self.root = Path(root or os.environ.get("WATER_AGENT_ROOT", default_root)).resolve()
        self.src = self.root / "water_resilience" / "src"
        self.outputs = self.root / "water_resilience" / "outputs"
        self.generated = self.root / "water_resilience" / "data" / "generated"
        self.extracted = self.root / "water_resilience" / "data" / "extracted"

    def run_analysis(self, use_llm: bool = False, timeout: int = 3600) -> dict[str, Any]:
        """运行完整分析链，并返回执行摘要。

        ``use_llm=False`` 时完全离线运行，风险关系来自公开示例中的种子关系。
        若启用 LLM，调用地址由公开配置控制，调用前应由使用者自行确认数据授权。
        """

        command = [sys.executable, str(self.src / "pipeline.py")]
        if use_llm:
            command.append("--use-llm")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.src)
        completed = subprocess.run(
            command,
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "use_llm": use_llm,
            "outputs": str(self.outputs),
            "log_tail": (completed.stdout + "\n" + completed.stderr)[-4000:],
        }

    def capabilities(self) -> dict[str, Any]:
        """返回 Agent 的功能清单和数据边界。"""

        return {
            "name": "智慧水务综合管理Agent系统",
            "capabilities": [
                "合成管网拓扑生成与多因子风险标注（证据化拓扑基础）",
                "压力、流量、余氯时序仿真与校准（calibrated time-series）",
                "基线感知预测：以简单基线为参照复检 LSTM 是否超越周季节基线",
                "风险因素—对象—后果—措施关系抽取（证据约束 LLM，默认关闭）",
                "四维韧性指标计算与熵权-TOPSIS 敏感性评分（实验性）",
                "高风险节点查询、报告导出和建议性维修清单",
            ],
            "data_boundary": "默认输入均为合成数据；输出仅用于功能演示和方法验证。",
            "control_boundary": "不连接生产系统，不执行调度、阀门控制或真实派单。",
        }

    def get_resilience_scores(self, limit: int = 20) -> list[dict[str, Any]]:
        """读取分区×周期的综合韧性评价结果。"""

        return self._read_csv(self.outputs / "resilience_scores.csv", limit)

    def get_high_risk_nodes(self, limit: int = 20) -> list[dict[str, Any]]:
        """读取高风险节点清单。"""

        return self._read_csv(self.outputs / "high_risk_nodes.csv", limit)

    def get_risk_relations(self, limit: int = 50) -> list[dict[str, Any]]:
        """读取风险知识四元组及其来源标记。"""

        return self._read_csv(self.extracted / "risk_relations.csv", limit)

    def get_model_metrics(self) -> dict[str, Any]:
        """读取实验性 GNN/LSTM 与综合评价权重结果（仅方法验证，非正式结论）。"""

        return {
            "gnn": self._read_json(self.outputs / "gnn_results.json"),
            "lstm": self._read_json(self.outputs / "lstm_results.json"),
            "entropy_weights": self._read_json(self.outputs / "entropy_weights.json"),
        }

    def get_topology_summary(self) -> dict[str, Any]:
        """统计拓扑节点和管段数量，不返回完整输入文件。"""

        data = self._read_json(self.generated / "topology.json")
        nodes = data.get("nodes", []) if isinstance(data, dict) else []
        links = data.get("links", data.get("edges", [])) if isinstance(data, dict) else []
        node_types: dict[str, int] = {}
        for node in nodes:
            node_type = str(node.get("node_type", "unknown"))
            node_types[node_type] = node_types.get(node_type, 0) + 1
        return {
            "nodes": len(nodes),
            "edges": len(links),
            "node_types": node_types,
            "simulated": bool(data.get("SIMULATED", True)) if isinstance(data, dict) else True,
        }

    def generate_work_orders(self, limit: int = 20) -> list[dict[str, Any]]:
        """根据高风险节点生成建议性维修清单，不执行任何外部操作。"""

        rows = self.get_high_risk_nodes(limit)
        orders = []
        for index, row in enumerate(rows, start=1):
            probability = self._as_float(row.get("risk_prob_high"))
            orders.append({
                "order_id": f"DEMO-{index:04d}",
                "priority": "high" if probability >= 0.67 else "medium",
                "node": row.get("node_id", row.get("name", "")),
                "town": row.get("town", ""),
                "suggestion": "核查现场状态、复核监测数据并安排计划性巡检",
                "risk_probability": probability,
                "status": "recommendation_only",
            })
        return orders

    def ask(self, question: str) -> dict[str, Any]:
        """提供一个轻量级中文意图路由，便于命令行或 WebUI 适配。"""

        text = (question or "").strip()
        if any(word in text for word in ("运行", "刷新", "分析")):
            return self.run_analysis()
        if "高风险" in text or "风险节点" in text:
            return {"items": self.get_high_risk_nodes()}
        if "韧性" in text or "综合评价" in text or "评分" in text:
            return {"items": self.get_resilience_scores()}
        if "关系" in text or "知识图谱" in text:
            return {"items": self.get_risk_relations()}
        if "模型" in text or "指标" in text:
            return self.get_model_metrics()
        if "拓扑" in text or "管网" in text:
            return self.get_topology_summary()
        if "工单" in text or "维修" in text:
            return {"items": self.generate_work_orders()}
        return self.capabilities()

    @staticmethod
    def _read_json(path: Path) -> Any:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _read_csv(path: Path, limit: int) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))[: max(0, int(limit))]

    @staticmethod
    def _as_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
