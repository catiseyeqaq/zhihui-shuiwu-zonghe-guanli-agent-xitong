#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可视化看板查看器: 重新生成自包含 HTML 看板并通过内置 http.server 提供浏览。

私有镜像无 Streamlit, 故用标准库 http.server 托管 outputs/ 目录, 浏览器打开 dashboard.html 即可。
用法: python water_resilience/app/dashboard.py [--port 8600]
"""

import argparse
import functools
import http.server
import os
import socketserver
import sys

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(APP_DIR), "src")
sys.path.insert(0, SRC)

import wr_config as C  # noqa: E402
from report import generate_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--no-serve", action="store_true", help="仅重新生成看板, 不启动服务")
    args = ap.parse_args()

    n = generate_all()
    out = C.PATHS["outputs"]
    print(f"[dashboard] 已生成看板(高风险节点 {n} 个): {os.path.join(out, 'dashboard.html')}")
    if args.no_serve:
        return
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=out)
    with socketserver.TCPServer(("0.0.0.0", args.port), handler) as httpd:
        print(f"[dashboard] 浏览器打开: http://localhost:{args.port}/dashboard.html  (Ctrl+C 停止)")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
