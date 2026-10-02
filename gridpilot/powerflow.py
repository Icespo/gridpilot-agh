from __future__ import annotations

from collections import defaultdict
from math import sqrt

import numpy as np


def _orders(case: dict) -> tuple[list[int], list[int], dict[int, list[int]], dict[int, dict]]:
    children: dict[int, list[int]] = defaultdict(list)
    line_to: dict[int, dict] = {}
    for line in case["lines"]:
        children[line["from"]].append(line["to"])
        line_to[line["to"]] = line

    preorder: list[int] = []
    postorder: list[int] = []

    def visit(bus: int) -> None:
        for child in children.get(bus, []):
            preorder.append(child)
            visit(child)
        postorder.append(bus)

    visit(1)
    return preorder, postorder, children, line_to


def run_power_flow(
    case: dict,
    p_kw: dict[int, float],
    q_kvar: dict[int, float],
    line_rating_multiplier: float = 1.0,
    max_iterations: int = 80,
) -> dict:
    """Backward/forward sweep for a balanced radial distribution feeder."""
    preorder, postorder, children, line_to = _orders(case)
    bus_ids = [row["bus"] for row in case["buses"]]
    v_phase = case["base_kv"] * 1000.0 / sqrt(3.0)
    voltage = {bus: complex(v_phase, 0.0) for bus in bus_ids}
    branch_current: dict[int, complex] = {}

    converged = False
    for iteration in range(1, max_iterations + 1):
        previous = voltage.copy()
        injection: dict[int, complex] = {}
        for bus in bus_ids:
            s_phase_va = complex(p_kw.get(bus, 0.0), q_kvar.get(bus, 0.0)) * 1000.0 / 3.0
            safe_v = voltage[bus] if abs(voltage[bus]) > 1.0 else complex(v_phase, 0.0)
            injection[bus] = np.conj(s_phase_va / safe_v)

        aggregate = injection.copy()
        for bus in postorder:
            if bus == 1:
                continue
            branch_current[bus] = aggregate[bus]
            parent = line_to[bus]["from"]
            aggregate[parent] = aggregate.get(parent, 0j) + aggregate[bus]

        voltage[1] = complex(v_phase, 0.0)
        for bus in preorder:
            line = line_to[bus]
            z = complex(line["r_ohm"], line["x_ohm"])
            voltage[bus] = voltage[line["from"]] - z * branch_current[bus]

        delta = max(abs(voltage[b] - previous[b]) for b in bus_ids)
        if delta < 1e-4:
            converged = True
            break

    voltage_pu = {bus: abs(voltage[bus]) / v_phase for bus in bus_ids}
    loading: dict[str, float] = {}
    for bus, current in branch_current.items():
        line = line_to[bus]
        multiplier = line_rating_multiplier if line["from"] == 1 and line["to"] == 2 else 1.0
        rating_a = line["max_ka"] * multiplier * 1000.0
        loading[f"{line['from']}-{line['to']}"] = abs(current) / rating_a

    v_min = case["voltage_min_pu"]
    v_max = case["voltage_max_pu"]
    voltage_violations = [
        {"bus": bus, "voltage_pu": value}
        for bus, value in voltage_pu.items()
        if value < v_min - 1e-6 or value > v_max + 1e-6
    ]
    thermal_violations = [
        {"line": line, "loading_pct": value * 100.0}
        for line, value in loading.items()
        if value > 1.0 + 1e-6
    ]
    return {
        "converged": converged,
        "iterations": iteration,
        "min_voltage_pu": float(min(voltage_pu.values())),
        "max_voltage_pu": float(max(voltage_pu.values())),
        "max_line_loading_pct": float(max(loading.values(), default=0.0) * 100.0),
        "voltage_pu": {str(k): float(v) for k, v in voltage_pu.items()},
        "line_loading_pct": {k: float(v * 100.0) for k, v in loading.items()},
        "voltage_violations": voltage_violations,
        "thermal_violations": thermal_violations,
    }


def hourly_injections(case: dict, load_scale: float, row: dict) -> tuple[dict[int, float], dict[int, float]]:
    p = {entry["bus"]: entry["p_kw"] * load_scale for entry in case["buses"]}
    q = {entry["bus"]: entry["q_kvar"] * load_scale for entry in case["buses"]}

    shed_by_bus = row.get("shed_by_bus_kw")
    if shed_by_bus is not None:
        for bus_text, shed_kw in shed_by_bus.items():
            bus = int(bus_text)
            original_p = max(p.get(bus, 0.0), 1e-9)
            fraction = min(float(shed_kw) / original_p, 1.0)
            p[bus] -= float(shed_kw)
            q[bus] *= 1.0 - fraction
    else:
        shed_ratio = min(row["shed_kw"] / max(row["load_kw"], 1e-9), 1.0)
        if shed_ratio > 0:
            for bus in p:
                p[bus] *= 1.0 - shed_ratio
                q[bus] *= 1.0 - shed_ratio

    pv_by_bus = row.get("pv_by_bus_kw")
    if pv_by_bus is not None:
        for bus_text, generation_kw in pv_by_bus.items():
            p[int(bus_text)] -= float(generation_kw)
    else:
        total_pv_capacity = sum(site["capacity_kw"] for site in case["pv_sites"])
        for site in case["pv_sites"]:
            allocation = row["pv_used_kw"] * site["capacity_kw"] / total_pv_capacity
            p[site["bus"]] -= allocation

    battery_bus = case["battery"]["bus"]
    p[battery_bus] -= row["battery_kw"]
    q[battery_bus] -= row.get("battery_q_kvar", 0.0)
    return p, q
