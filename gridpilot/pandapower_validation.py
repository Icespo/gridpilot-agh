from __future__ import annotations

import copy
from math import isfinite

import numpy as np
import pandapower as pp
from pandapower.topology import unsupplied_buses


def _bus_injections(case: dict, load_scale: float, row: dict) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    p_load = {entry["bus"]: float(entry["p_kw"] * load_scale) for entry in case["buses"]}
    q_load = {entry["bus"]: float(entry["q_kvar"] * load_scale) for entry in case["buses"]}
    shed_by_bus = {int(bus): float(value) for bus, value in row.get("shed_by_bus_kw", {}).items()}
    if not shed_by_bus and row.get("shed_kw", 0.0) > 0:
        ratio = min(float(row["shed_kw"]) / max(float(row["load_kw"]), 1e-9), 1.0)
        shed_by_bus = {bus: value * ratio for bus, value in p_load.items()}
    for bus, shed_kw in shed_by_bus.items():
        original = max(p_load.get(bus, 0.0), 1e-9)
        fraction = min(shed_kw / original, 1.0)
        p_load[bus] -= shed_kw
        q_load[bus] *= 1.0 - fraction
    return p_load, q_load, shed_by_bus


def _backup_feeders(case: dict) -> list[dict]:
    feeders = case.get("backup_feeders")
    if feeders:
        return list(feeders)
    return [case["backup_feeder"]] if case.get("backup_feeder") else []


def build_pandapower_network(
    case: dict,
    load_scale: float,
    row: dict,
    line_rating_multiplier: float = 1.0,
) -> tuple[pp.pandapowerNet, dict]:
    net = pp.create_empty_network(sn_mva=10.0, f_hz=50.0)
    bus_map: dict[int, int] = {}
    reverse_bus_map: dict[int, int] = {}
    for entry in case["buses"]:
        bus_index = pp.create_bus(
            net,
            vn_kv=case["base_kv"],
            name=f"Bus {entry['bus']}",
            min_vm_pu=case["voltage_min_pu"],
            max_vm_pu=case["voltage_max_pu"],
        )
        bus_map[entry["bus"]] = bus_index
        reverse_bus_map[bus_index] = entry["bus"]
    pp.create_ext_grid(
        net,
        bus=bus_map[1],
        vm_pu=float(case.get("source_voltage_pu", 1.0)),
        name="Upstream grid",
    )

    base_line_indices: list[int] = []
    for line in case["lines"]:
        multiplier = line_rating_multiplier if line["from"] == 1 and line["to"] == 2 else 1.0
        line_index = pp.create_line_from_parameters(
            net,
            from_bus=bus_map[line["from"]],
            to_bus=bus_map[line["to"]],
            length_km=1.0,
            r_ohm_per_km=line["r_ohm"],
            x_ohm_per_km=line["x_ohm"],
            c_nf_per_km=0.0,
            max_i_ka=line["max_ka"] * multiplier,
            name=f"L{line['from']}-{line['to']}",
            in_service=True,
        )
        base_line_indices.append(line_index)

    tie_line_indices: list[int] = []
    restoration_candidates: list[dict] = []
    for tie in case.get("tie_lines", []):
        line_index = pp.create_line_from_parameters(
            net,
            from_bus=bus_map[tie["from"]],
            to_bus=bus_map[tie["to"]],
            length_km=1.0,
            r_ohm_per_km=tie["r_ohm"],
            x_ohm_per_km=tie["x_ohm"],
            c_nf_per_km=0.0,
            max_i_ka=tie["max_ka"],
            name=f"T{tie['from']}-{tie['to']}",
            in_service=not tie.get("normally_open", True),
        )
        tie_line_indices.append(line_index)
        restoration_candidates.append({"index": line_index, "from": tie["from"], "to": tie["to"], "type": "tie_line"})

    backup_line_indices: list[int] = []
    for backup in _backup_feeders(case):
        backup_line_index = pp.create_line_from_parameters(
            net,
            from_bus=bus_map[backup["from"]],
            to_bus=bus_map[backup["to"]],
            length_km=1.0,
            r_ohm_per_km=backup["r_ohm"],
            x_ohm_per_km=backup["x_ohm"],
            c_nf_per_km=0.0,
            max_i_ka=backup["max_ka"],
            name=backup.get("name", f"B{backup['from']}-{backup['to']}"),
            in_service=not backup.get("normally_open", True),
        )
        backup_line_indices.append(backup_line_index)
        restoration_candidates.append(
            {
                "index": backup_line_index,
                "from": backup["from"],
                "to": backup["to"],
                "type": "backup_feeder",
                "switching_time_minutes": int(backup.get("switching_time_minutes", 10)),
                "physical_repair_hours": int(backup.get("physical_repair_hours", 3)),
            }
        )

    p_load, q_load, shed_by_bus = _bus_injections(case, load_scale, row)
    for bus in bus_map:
        if p_load[bus] > 1e-9 or abs(q_load[bus]) > 1e-9:
            pp.create_load(
                net,
                bus=bus_map[bus],
                p_mw=p_load[bus] / 1000.0,
                q_mvar=q_load[bus] / 1000.0,
                name=f"Load {bus}",
            )

    pv_by_bus = row.get("pv_by_bus_kw")
    if pv_by_bus is None:
        total_capacity = sum(site["capacity_kw"] for site in case["pv_sites"])
        pv_by_bus = {
            str(site["bus"]): row["pv_used_kw"] * site["capacity_kw"] / total_capacity
            for site in case["pv_sites"]
        }
    for bus_text, value in pv_by_bus.items():
        pp.create_sgen(net, bus=bus_map[int(bus_text)], p_mw=float(value) / 1000.0, q_mvar=0.0, name=f"PV {bus_text}")

    # Commissioned SVGs use a conservative Volt-VAR proxy proportional to the
    # current load level.  The rating is a hard ceiling; the N-1 AC assessment
    # therefore receives real reactive injection rather than a display-only flag.
    for device in case.get("reactive_support", []):
        q_kvar = min(
            float(device["capacity_kvar"]),
            float(device["capacity_kvar"]) * max(load_scale, 0.0) ** 2,
        )
        pp.create_sgen(
            net,
            bus=bus_map[int(device["bus"])],
            p_mw=0.0,
            q_mvar=q_kvar / 1000.0,
            name=str(device.get("name", f"SVG-{device['bus']}")),
        )

    battery_bus = case["battery"]["bus"]
    pp.create_sgen(
        net,
        bus=bus_map[battery_bus],
        p_mw=float(row["battery_kw"]) / 1000.0,
        q_mvar=float(row.get("battery_q_kvar", 0.0)) / 1000.0,
        name="Battery inverter",
    )

    metadata = {
        "bus_map": bus_map,
        "reverse_bus_map": reverse_bus_map,
        "base_line_indices": base_line_indices,
        "tie_line_indices": tie_line_indices,
        "backup_line_indices": backup_line_indices,
        "restoration_candidates": restoration_candidates,
        "gross_served_load_kw": p_load,
        "shed_by_bus_kw": shed_by_bus,
    }
    return net, metadata


def _evaluate_network(net: pp.pandapowerNet, metadata: dict, case: dict) -> dict:
    unsupplied = set(unsupplied_buses(net))
    unserved_kw = float(sum(metadata["gross_served_load_kw"].get(metadata["reverse_bus_map"][bus], 0.0) for bus in unsupplied))
    converged = False
    error = None
    try:
        pp.runpp(
            net,
            algorithm="nr",
            init="flat",
            calculate_voltage_angles=False,
            check_connectivity=True,
            enforce_q_lims=False,
            numba=False,
            max_iteration=30,
            tolerance_mva=1e-8,
        )
        converged = bool(net.converged)
    except Exception as exc:  # pandapower convergence boundary
        error = str(exc)

    voltage_pu: dict[str, float] = {}
    voltage_violations: list[dict] = []
    if converged and not net.res_bus.empty:
        for pp_bus, value in net.res_bus.vm_pu.items():
            if not isfinite(float(value)):
                continue
            original_bus = metadata["reverse_bus_map"][int(pp_bus)]
            voltage_pu[str(original_bus)] = float(value)
            if value < case["voltage_min_pu"] - 1e-6 or value > case["voltage_max_pu"] + 1e-6:
                voltage_violations.append({"bus": original_bus, "voltage_pu": float(value)})

    line_loading: dict[str, float] = {}
    thermal_violations: list[dict] = []
    if converged and not net.res_line.empty:
        for line_index, value in net.res_line.loading_percent.items():
            if not isfinite(float(value)) or not bool(net.line.at[line_index, "in_service"]):
                continue
            name = str(net.line.at[line_index, "name"])
            line_loading[name] = float(value)
            if value > 100.0 + 1e-6:
                thermal_violations.append({"line": name, "loading_pct": float(value)})

    return {
        "converged": converged,
        "error": error,
        "iterations": None,
        "min_voltage_pu": float(min(voltage_pu.values())) if voltage_pu else 0.0,
        "max_voltage_pu": float(max(voltage_pu.values())) if voltage_pu else 0.0,
        "max_line_loading_pct": float(max(line_loading.values(), default=0.0)),
        "voltage_pu": voltage_pu,
        "line_loading_pct": line_loading,
        "voltage_violations": voltage_violations,
        "thermal_violations": thermal_violations,
        "unsupplied_buses": sorted(metadata["reverse_bus_map"][bus] for bus in unsupplied),
        "unserved_load_kw": unserved_kw,
        "backend": "pandapower Newton-Raphson AC power flow",
    }


def run_pandapower_ac(
    case: dict,
    load_scale: float,
    row: dict,
    line_rating_multiplier: float = 1.0,
) -> dict:
    net, metadata = build_pandapower_network(case, load_scale, row, line_rating_multiplier)
    return _evaluate_network(net, metadata, case)


def validate_schedule_with_pandapower(
    case: dict,
    schedule: list[dict],
    load_profile: np.ndarray,
    load_multiplier: float,
    line_rating_multiplier: float,
) -> list[dict]:
    checks: list[dict] = []
    for row in schedule:
        scale = float(load_profile[row["hour"]] * load_multiplier)
        result = run_pandapower_ac(case, scale, row, line_rating_multiplier)
        result["hour"] = row["hour"]
        checks.append(result)
    return checks


def _contingency_score(result: dict, case: dict) -> float:
    if not result["converged"]:
        return 1e12 + result["unserved_load_kw"] * 1e6
    low_voltage = max(case["voltage_min_pu"] - result["min_voltage_pu"], 0.0)
    high_voltage = max(result["max_voltage_pu"] - case["voltage_max_pu"], 0.0)
    overload = max(result["max_line_loading_pct"] - 100.0, 0.0)
    return result["unserved_load_kw"] * 1e6 + (low_voltage + high_voltage) * 1e7 + overload * 1e4


def _eligible_restoration_candidates(case: dict, metadata: dict, outage_position: int) -> list[dict]:
    """Return only normally-open restoration branches that bridge outage islands.

    Closing a tie whose endpoints remain in the same island cannot restore supply
    and may create a loop.  Filtering those options keeps the exhaustive 24-hour
    assessment equivalent to the previous search while avoiding unnecessary AC
    power-flow solves.
    """
    adjacency: dict[int, list[int]] = {int(row["bus"]): [] for row in case["buses"]}
    for position, line in enumerate(case["lines"]):
        if position == outage_position:
            continue
        left, right = int(line["from"]), int(line["to"])
        adjacency[left].append(right)
        adjacency[right].append(left)

    source_component: set[int] = {1}
    stack = [1]
    while stack:
        node = stack.pop()
        for neighbour in adjacency[node]:
            if neighbour not in source_component:
                source_component.add(neighbour)
                stack.append(neighbour)

    eligible: list[dict] = []
    for candidate in metadata["restoration_candidates"]:
        left_in_source = int(candidate["from"]) in source_component
        right_in_source = int(candidate["to"]) in source_component
        if left_in_source != right_in_source:
            eligible.append(candidate)
    return eligible


def run_line_n_1_assessment(
    case: dict,
    schedule: list[dict],
    load_profile: np.ndarray,
    load_multiplier: float,
    line_rating_multiplier: float,
    base_checks: list[dict],
    hours_to_check: list[int] | None = None,
) -> dict:
    if hours_to_check is None:
        assessed_hours = [int(row["hour"]) for row in schedule]
        assessment_mode = "full_24h"
    else:
        valid_hours = {int(row["hour"]) for row in schedule}
        assessed_hours = list(dict.fromkeys(int(hour) for hour in hours_to_check))
        if not assessed_hours or any(hour not in valid_hours for hour in assessed_hours):
            raise ValueError("N-1校核时段必须来自当前调度计划且不能为空")
        assessment_mode = "selected_hours"
    records: list[dict] = []
    power_flow_runs = 0

    for hour in assessed_hours:
        row = schedule[hour]
        scale = float(load_profile[hour] * load_multiplier)
        base_net, metadata = build_pandapower_network(case, scale, row, line_rating_multiplier)
        for outage_position, outage_index in enumerate(metadata["base_line_indices"]):
            candidates: list[tuple[str | None, dict]] = []
            eligible_restorations = _eligible_restoration_candidates(case, metadata, outage_position)
            candidate_types: dict[str | None, str] = {None: "none"}
            candidate_settings: dict[str, dict] = {}
            for restoration in [None, *eligible_restorations]:
                net = copy.deepcopy(base_net)
                net.line.at[outage_index, "in_service"] = False
                tie_name = None
                if restoration is not None:
                    line_index = int(restoration["index"])
                    net.line.at[line_index, "in_service"] = True
                    tie_name = str(net.line.at[line_index, "name"])
                    candidate_types[tie_name] = str(restoration["type"])
                    candidate_settings[tie_name] = restoration
                result = _evaluate_network(net, metadata, case)
                power_flow_runs += 1
                candidates.append((tie_name, result))
            before_restoration = candidates[0][1]
            tie_name, best = min(candidates, key=lambda item: _contingency_score(item[1], case))
            restoration_type = candidate_types.get(tie_name, "none")
            chosen_settings = candidate_settings.get(tie_name, {})
            default_backup = _backup_feeders(case)[0] if _backup_feeders(case) else {}
            secure = bool(
                best["converged"]
                and best["unserved_load_kw"] < 1e-6
                and not best["voltage_violations"]
                and not best["thermal_violations"]
            )
            line = case["lines"][outage_position]
            records.append({
                "hour": hour,
                "outage_line": f"L{line['from']}-{line['to']}",
                "restoration_tie": tie_name,
                "restoration_type": restoration_type,
                "switching_time_minutes": (
                    int(chosen_settings.get("switching_time_minutes", 10))
                    if restoration_type == "backup_feeder"
                    else 5 if restoration_type == "tie_line" else None
                ),
                "physical_repair_hours": int(chosen_settings.get("physical_repair_hours", default_backup.get("physical_repair_hours", 3))),
                "repair_completion_hour": (hour + int(chosen_settings.get("physical_repair_hours", default_backup.get("physical_repair_hours", 3)))) % 24,
                "secure": secure,
                "converged": best["converged"],
                "unserved_load_kw": round(best["unserved_load_kw"], 3),
                "min_voltage_pu": round(best["min_voltage_pu"], 5),
                "max_line_loading_pct": round(best["max_line_loading_pct"], 3),
                "voltage_violation_count": len(best["voltage_violations"]),
                "thermal_violation_count": len(best["thermal_violations"]),
                "unsupplied_buses": best["unsupplied_buses"],
                "pre_restoration_unserved_load_kw": round(before_restoration["unserved_load_kw"], 3),
                "pre_restoration_unsupplied_buses": before_restoration["unsupplied_buses"],
            })

    failed = [record for record in records if not record["secure"]]
    restored = [record for record in records if record["secure"] and record["restoration_tie"]]
    critical = sorted(
        failed,
        key=lambda record: (
            record["unserved_load_kw"],
            max(0.0, case["voltage_min_pu"] - record["min_voltage_pu"]),
            max(0.0, record["max_line_loading_pct"] - 100.0),
        ),
        reverse=True,
    )[:10]
    return {
        "backend": "pandapower AC N-1 with corrective tie-line and backup-feeder restoration search",
        "assessment_mode": assessment_mode,
        "hours_checked": assessed_hours,
        "expected_contingencies": len(assessed_hours) * len(case["lines"]),
        "contingencies_evaluated": len(records),
        "power_flow_runs": power_flow_runs,
        "repair_policy": {
            "fault_isolation_minutes": 1,
            "tie_switching_minutes": 5,
            "backup_feeder_switching_minutes": min((int(item.get("switching_time_minutes", 10)) for item in _backup_feeders(case)), default=10),
            "assumed_physical_repair_hours": min((int(item.get("physical_repair_hours", 3)) for item in _backup_feeders(case)), default=3),
            "interpretation": "N-1为逐时假想故障；开关恢复按分钟级，物理抢修按可配置的3小时演示假设。",
        },
        "secure_contingencies": len(records) - len(failed),
        "restored_contingencies": len(restored),
        "failed_contingencies": len(failed),
        "security_rate_pct": round(100.0 * (len(records) - len(failed)) / max(len(records), 1), 2),
        "worst_unserved_load_kw": round(max((record["unserved_load_kw"] for record in records), default=0.0), 3),
        "worst_min_voltage_pu": round(min((record["min_voltage_pu"] for record in records if record["converged"]), default=0.0), 5),
        "worst_max_line_loading_pct": round(max((record["max_line_loading_pct"] for record in records), default=0.0), 3),
        "critical_contingencies": critical,
        "records": records,
    }
