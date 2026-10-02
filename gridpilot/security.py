from __future__ import annotations

from collections import defaultdict, deque


def contingency_risk(record: dict, case: dict) -> tuple[float, str, list[str]]:
    """Return a normalized severity score, level and human-readable drivers."""
    drivers: list[str] = []
    unserved = max(float(record.get("unserved_load_kw", 0.0)), 0.0)
    min_voltage = float(record.get("min_voltage_pu", 0.0))
    max_loading = max(float(record.get("max_line_loading_pct", 0.0)), 0.0)
    if not record.get("converged", False):
        drivers.append("潮流不收敛")
    if unserved > 1e-3:
        drivers.append(f"失供 {unserved:.0f} kW")
    if 0.0 < min_voltage < case["voltage_min_pu"]:
        drivers.append(f"最低电压 {min_voltage:.3f} pu")
    if max_loading > 100.0:
        drivers.append(f"最高负载率 {max_loading:.0f}%")

    if record.get("secure", False):
        score = 0.0
    else:
        convergence = 20.0 if not record.get("converged", False) else 0.0
        supply = min(unserved / 2500.0, 1.0) * 78.0
        voltage = min(max(case["voltage_min_pu"] - min_voltage, 0.0) / 0.12, 1.0) * 58.0
        thermal = min(max(max_loading - 100.0, 0.0) / 100.0, 1.0) * 42.0
        score = min(100.0, 20.0 + convergence + supply + voltage + thermal)
    level = "critical" if score >= 75.0 else "high" if score >= 50.0 else "medium" if score >= 20.0 else "secure"
    return round(score, 1), level, drivers or ["满足 N-1 安全边界"]


def enrich_n_1_records(n_1: dict, case: dict) -> dict:
    for record in n_1.get("records", []):
        score, level, drivers = contingency_risk(record, case)
        record["risk_score"] = score
        record["risk_level"] = level
        record["risk_drivers"] = drivers
    for record in n_1.get("critical_contingencies", []):
        match = next(
            (
                item
                for item in n_1.get("records", [])
                if item["hour"] == record["hour"] and item["outage_line"] == record["outage_line"]
            ),
            None,
        )
        if match:
            record.update({key: match[key] for key in ("risk_score", "risk_level", "risk_drivers")})
    return n_1


def generate_n_1_security_constraints(
    case: dict,
    n_1: dict,
    max_constraints: int = 6,
    excluded_keys: set[tuple[int, str, str]] | None = None,
    id_start: int = 1,
) -> dict:
    """Screen N-1 results into operational AC-risk cuts and planning actions."""
    enrich_n_1_records(n_1, case)
    failed = [record for record in n_1.get("records", []) if not record.get("secure", False)]
    operational = [
        record
        for record in failed
        if record.get("restoration_tie")
        and record.get("converged", False)
        and float(record.get("unserved_load_kw", 0.0)) < 1e-3
    ]
    operational.sort(key=lambda record: record.get("risk_score", 0.0), reverse=True)
    excluded_keys = excluded_keys or set()

    # First cover different failed branches, then fill remaining slots by risk.
    diverse: list[dict] = []
    remaining: list[dict] = []
    represented_lines: set[str] = set()
    for record in operational:
        key = (int(record["hour"]), record["outage_line"], record["restoration_tie"])
        if key in excluded_keys:
            continue
        if record["outage_line"] not in represented_lines:
            diverse.append(record)
            represented_lines.add(record["outage_line"])
        else:
            remaining.append(record)
    operational = diverse + remaining

    constraints: list[dict] = []
    seen: set[tuple[int, str, str]] = set()
    for record in operational:
        key = (int(record["hour"]), record["outage_line"], record["restoration_tie"])
        if key in seen:
            continue
        seen.add(key)
        response = case.get("contingency_response", {})
        constraints.append(
            {
                "id": f"SC-{id_start + len(constraints):02d}",
                "hour": int(record["hour"]),
                "outage_line": record["outage_line"],
                "restoration_tie": record["restoration_tie"],
                "restoration_type": record.get("restoration_type", "tie_line"),
                "risk_score": float(record["risk_score"]),
                "risk_level": record["risk_level"],
                "risk_drivers": list(record["risk_drivers"]),
                "constraint_types": [
                    "post_contingency_power_balance",
                    "post_contingency_voltage",
                    "post_contingency_line_capacity",
                    "post_contingency_corrective_resources",
                ],
                "source_min_voltage_pu": float(record["min_voltage_pu"]),
                "source_max_line_loading_pct": float(record["max_line_loading_pct"]),
                "target_min_voltage_pu": round(
                    min(case["voltage_min_pu"], float(record["min_voltage_pu"]) + 0.015),
                    4,
                ),
                "target_max_line_loading_pct": 100.0,
                "battery_response_kw": round(float(case["battery"]["power_kw"]) * float(response.get("battery_response_fraction", 0.35)), 3),
                "demand_response_fraction": float(response.get("demand_response_fraction", 0.08)),
                "grid_response_kw": float(response.get("grid_response_kw", 800.0)),
                "status": "generated",
            }
        )
        if len(constraints) >= max_constraints:
            break

    structural = [record for record in failed if record not in operational]
    structural.sort(key=lambda record: record.get("risk_score", 0.0), reverse=True)
    planning_actions = [
        {
            "hour": int(record["hour"]),
            "outage_line": record["outage_line"],
            "risk_score": float(record["risk_score"]),
            "reason": "故障后存在失供或无可行单支路恢复，属于网架/备用资源补强问题",
            "unserved_load_kw": float(record.get("unserved_load_kw", 0.0)),
            "recommended_action": "增加联络容量、配置孤岛电源或将该故障纳入规划扩建",
        }
        for record in structural[:8]
    ]
    return {
        "method": "pandapower screening + iterative LinDistFlow contingency cut generation",
        "screened_failed_contingencies": len(failed),
        "operational_candidates": len(operational),
        "generated_constraints": len(constraints),
        "planning_actions_count": len(structural),
        "constraints": constraints,
        "planning_actions": planning_actions,
    }


def parse_edge_name(name: str) -> tuple[int, int]:
    prefix = name[:1]
    if prefix not in {"L", "T", "B"}:
        raise ValueError(f"无法识别线路名称：{name}")
    left, right = name[1:].split("-", 1)
    return int(left), int(right)


def contingency_tree(case: dict, outage_line: str, restoration_tie: str) -> list[dict]:
    """Build and orient the restored radial topology from the source bus."""
    outage = parse_edge_name(outage_line)
    restoration = parse_edge_name(restoration_tie)
    raw_edges: list[dict] = []
    for line in case["lines"]:
        if (line["from"], line["to"]) == outage:
            continue
        raw_edges.append({**line, "name": f"L{line['from']}-{line['to']}", "kind": "base"})
    if restoration_tie.startswith("B"):
        backup = case.get("backup_feeder")
        if not backup or {backup["from"], backup["to"]} != {restoration[0], restoration[1]}:
            raise ValueError(f"找不到备用馈线：{restoration_tie}")
        raw_edges.append({**backup, "name": backup.get("name", restoration_tie), "kind": "backup"})
    else:
        tie = next(
            item
            for item in case.get("tie_lines", [])
            if {item["from"], item["to"]} == {restoration[0], restoration[1]}
        )
        raw_edges.append({**tie, "name": f"T{tie['from']}-{tie['to']}", "kind": "tie"})

    adjacency: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    for edge in raw_edges:
        adjacency[edge["from"]].append((edge["to"], edge))
        adjacency[edge["to"]].append((edge["from"], edge))
    parent: dict[int, int | None] = {1: None}
    queue: deque[int] = deque([1])
    oriented: list[dict] = []
    while queue:
        node = queue.popleft()
        for neighbor, edge in adjacency[node]:
            if neighbor in parent:
                continue
            parent[neighbor] = node
            queue.append(neighbor)
            oriented.append({**edge, "from": node, "to": neighbor})
    if len(parent) != len(case["buses"]):
        raise ValueError(f"{outage_line} + {restoration_tie} 不能形成连通恢复拓扑")
    if len(oriented) != len(case["buses"]) - 1:
        raise ValueError(f"{outage_line} + {restoration_tie} 未形成辐射网络")
    return oriented


def build_topology_visualization(case: dict) -> dict:
    """Provide deterministic IEEE 33-bus coordinates for browser SVG rendering."""
    positions: dict[int, tuple[int, int]] = {}
    for bus in range(1, 19):
        positions[bus] = (40 + (bus - 1) * 50, 180)
    for bus in range(19, 23):
        positions[bus] = (90 + (bus - 19) * 60, 70)
    for bus in range(23, 26):
        positions[bus] = (140 + (bus - 23) * 65, 285)
    for bus in range(26, 34):
        positions[bus] = (290 + (bus - 26) * 60, 390)
    pv_buses = {int(site["bus"]) for site in case.get("pv_sites", [])}
    battery_bus = int(case["battery"]["bus"])
    nodes = [
        {
            "bus": int(row["bus"]),
            "x": positions[int(row["bus"])][0],
            "y": positions[int(row["bus"])][1],
            "type": "source" if row["bus"] == 1 else "battery" if row["bus"] == battery_bus else "pv" if row["bus"] in pv_buses else "load",
        }
        for row in case["buses"]
    ]
    lines = [
        {"name": f"L{line['from']}-{line['to']}", "from": line["from"], "to": line["to"]}
        for line in case["lines"]
    ]
    ties = [
        {"name": f"T{line['from']}-{line['to']}", "from": line["from"], "to": line["to"]}
        for line in case.get("tie_lines", [])
    ]
    backup = case.get("backup_feeder")
    backups = [] if not backup else [{
        "name": backup.get("name", f"B{backup['from']}-{backup['to']}"),
        "from": backup["from"],
        "to": backup["to"],
        "description": backup.get("description", "独立备用馈线"),
    }]
    return {"width": 940, "height": 450, "nodes": nodes, "lines": lines, "ties": ties, "backup_feeders": backups}
