from __future__ import annotations

import argparse

from gridpilot.webapp import serve


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="启动 GridPilot-AGH 网页工作台")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    serve(args.host, args.port)

