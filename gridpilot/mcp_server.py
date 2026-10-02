from __future__ import annotations

import json
import sys
from pathlib import Path

from .data import SCENARIOS, load_case, validate_case
from .engine import run_closed_loop
from .reporting import write_artifacts


TOOLS = [
    {
        "name": "validate_grid_case",
        "description": "校验 IEEE 33 节点算例的拓扑、负荷和线路参数。",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "run_gridpilot_closed_loop",
        "description": "执行调度优化、交流潮流校核、异常诊断和修复重优化闭环。",
        "inputSchema": {
            "type": "object",
            "properties": {"scenario": {"type": "string", "enum": list(SCENARIOS)}},
            "required": ["scenario"],
            "additionalProperties": False,
        },
    },
    {
        "name": "compare_grid_scenarios",
        "description": "对比正常、光伏偏差和储能退出三类场景的关键指标。",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "export_gridpilot_report",
        "description": "运行指定场景并导出 HTML 报告、CSV 调度表和 JSON 证据。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "scenario": {"type": "string", "enum": list(SCENARIOS)},
                "output_dir": {"type": "string"},
            },
            "required": ["scenario"],
            "additionalProperties": False,
        },
    },
]


def _content(value: object, is_error: bool = False) -> dict:
    return {"content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False, indent=2)}], "isError": is_error}


def call_tool(name: str, arguments: dict) -> dict:
    if name == "validate_grid_case":
        case = load_case()
        errors = validate_case(case)
        return _content({"valid": not errors, "errors": errors, "buses": len(case["buses"]), "lines": len(case["lines"])})
    if name == "run_gridpilot_closed_loop":
        result = run_closed_loop(arguments.get("scenario", "normal"))
        compact = {key: result[key] for key in ["run_id", "scenario", "status", "conclusion", "metrics", "trace"]}
        if result.get("n_1"):
            compact["n_1_summary"] = {
                key: result["n_1"][key]
                for key in ["hours_checked", "contingencies_evaluated", "failed_contingencies", "security_rate_pct", "critical_contingencies"]
            }
        return _content(compact)
    if name == "compare_grid_scenarios":
        rows = []
        for scenario in ["normal", "pv_error", "battery_outage"]:
            result = run_closed_loop(scenario)
            rows.append({"scenario": scenario, "status": result["status"], **result["metrics"]})
        return _content(rows)
    if name == "export_gridpilot_report":
        result = run_closed_loop(arguments.get("scenario", "normal"))
        root = Path(arguments.get("output_dir") or Path.cwd() / "output" / result["scenario"]).resolve()
        paths = write_artifacts(result, root)
        return _content({"status": result["status"], "files": {k: str(v) for k, v in paths.items()}})
    return _content({"error": f"未知工具：{name}"}, is_error=True)


def handle(message: dict) -> dict | None:
    method = message.get("method")
    request_id = message.get("id")
    if request_id is None:
        return None
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"protocolVersion": "2024-11-05", "capabilities": {"tools": {"listChanged": False}}, "serverInfo": {"name": "gridpilot-agh", "version": "0.1.0"}}}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOLS}}
    if method == "tools/call":
        params = message.get("params", {})
        try:
            result = call_tool(params.get("name", ""), params.get("arguments") or {})
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except Exception as exc:  # protocol boundary
            return {"jsonrpc": "2.0", "id": request_id, "result": _content({"error": str(exc)}, is_error=True)}
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}


def main() -> None:
    for line in sys.stdin:
        try:
            message = json.loads(line)
            response = handle(message)
            if response is not None:
                sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
                sys.stdout.flush()
        except Exception as exc:
            sys.stderr.write(f"gridpilot MCP error: {exc}\n")
            sys.stderr.flush()


if __name__ == "__main__":
    main()
