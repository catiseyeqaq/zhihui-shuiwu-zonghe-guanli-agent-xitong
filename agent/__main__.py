"""命令行入口：python -m agent [问题]。"""

import json
import sys

from .water_management_agent import WaterManagementAgent


def main() -> None:
    question = " ".join(sys.argv[1:]).strip() or "你能做什么？"
    print(json.dumps(WaterManagementAgent().ask(question), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
