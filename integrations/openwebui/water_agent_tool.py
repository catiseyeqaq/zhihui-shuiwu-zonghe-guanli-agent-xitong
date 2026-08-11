"""Open WebUI 适配示例。

将本文件作为函数导入 Open WebUI 后，可把公开 Agent 的查询能力接到聊天界面。
它只读本地合成分析结果，运行分析时也只写入仓库的 outputs/data 目录。
"""

import json
import sys
from pathlib import Path
from typing import Any

# 允许从仓库根目录直接加载公开 Agent 包。
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent import WaterManagementAgent


class Tools:
    """Open WebUI Functions 风格的最小适配层。"""

    def __init__(self) -> None:
        self.agent = WaterManagementAgent()

    def run_water_analysis(self, use_llm: bool = False) -> str:
        """运行离线合成数据分析流水线；默认不调用 LLM。"""

        return json.dumps(self.agent.run_analysis(use_llm=use_llm), ensure_ascii=False, indent=2)

    def query_water_agent(self, question: str) -> str:
        """用中文问题查询韧性、风险、拓扑、模型或建议性维修清单。"""

        return json.dumps(self.agent.ask(question), ensure_ascii=False, indent=2)

    def get_water_capabilities(self) -> str:
        """查看公开 Agent 的功能和数据边界。"""

        return json.dumps(self.agent.capabilities(), ensure_ascii=False, indent=2)
