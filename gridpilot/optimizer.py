from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .security import contingency_tree


@dataclass(frozen=True)
class BatteryConfig:
    capacity_kwh: float
    power_kw: float
    initial_soc: float
    min_soc: float
    max_soc: float
    eta_charge: float
    eta_discharge: float


class _Variables:
    def __init__(self) -> None:
        self.size = 0
        self.blocks: dict[str, np.ndarray] = {}

    def add(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        count = int(np.prod(shape))
        block = np.arange(self.size, self.size + count, dtype=int).reshape(shape)
        self.blocks[name] = block
        self.size += count
        return block


class _LinearRows:
    def __init__(self) -> None:
        self.rows: list[int] = []
        self.cols: list[int] = []
        self.data: list[float] = []
        self.rhs: list[float] = []

    def add(self, coefficients: dict[int, float], rhs: float) -> None:
        row = len(self.rhs)
        for column, value in coefficients.items():
            if abs(value) > 1e-12:
                self.rows.append(row)
                self.cols.append(int(column))
                self.data.append(float(value))
        self.rhs.append(float(rhs))

    def matrix(self, variable_count: int):
        if not self.rhs:
            return None
        return coo_matrix(
            (self.data, (self.rows, self.cols)),
            shape=(len(self.rhs), variable_count),
        ).tocsr()


def _solve_window(
    case: dict,
    load_scale: np.ndarray,
    pv_total_kw: np.ndarray,
    price: np.ndarray,
    battery: BatteryConfig,
    grid_limits_kw: np.ndarray,
    initial_energy_kwh: float,
    line_rating_multiplier: float,
    voltage_margin_pu: float,
    terminal_min_energy_kwh: float | None,
    terminal_value_yuan_per_kwh: float,
    window_start_hour: int,
    security_constraints: list[dict],
) -> dict:
    hours = len(load_scale)
    buses = [row["bus"] for row in case["buses"]]
    bus_position = {bus: index for index, bus in enumerate(buses)}
    lines = case["lines"]
    line_position_by_child = {line["to"]: index for index, line in enumerate(lines)}
    outgoing: dict[int, list[int]] = {bus: [] for bus in buses}
    for index, line in enumerate(lines):
        outgoing[line["from"]].append(index)

    pv_sites = case["pv_sites"]
    pv_sites_by_bus: dict[int, list[int]] = {bus: [] for bus in buses}
    for index, site in enumerate(pv_sites):
        pv_sites_by_bus[site["bus"]].append(index)
    total_pv_capacity = sum(site["capacity_kw"] for site in pv_sites)
    reactive_support_by_bus = {
        int(device["bus"]): float(device["capacity_kvar"])
        for device in case.get("reactive_support", [])
    }
    source_voltage_sq = float(case.get("source_voltage_pu", 1.0)) ** 2

    variables = _Variables()
    grid = variables.add("grid", (hours,))
    q_grid = variables.add("q_grid", (hours,))
    charge = variables.add("charge", (hours,))
    discharge = variables.add("discharge", (hours,))
    battery_q = variables.add("battery_q", (hours,))
    soc = variables.add("soc", (hours,))
    pv_used = variables.add("pv_used", (hours, len(pv_sites)))
    shed = variables.add("shed", (hours, len(buses)))
    p_flow = variables.add("p_flow", (hours, len(lines)))
    q_flow = variables.add("q_flow", (hours, len(lines)))
    voltage_sq = variables.add("voltage_sq", (hours, len(buses)))

    security_models: list[dict] = []
    for cut in security_constraints:
        global_hour = int(cut["hour"])
        if not window_start_hour <= global_hour < window_start_hour + hours:
            continue
        edges = contingency_tree(case, cut["outage_line"], cut["restoration_tie"])
        security_models.append({
            "cut": cut,
            "local_hour": global_hour - window_start_hour,
            "edges": edges,
            "p_flow": variables.add(f"security_p_{cut['id']}", (len(edges),)),
            "q_flow": variables.add(f"security_q_{cut['id']}", (len(edges),)),
            "voltage_sq": variables.add(f"security_v_{cut['id']}", (len(buses),)),
            "battery_up": variables.add(f"security_battery_up_{cut['id']}", (1,)),
            "battery_down": variables.add(f"security_battery_down_{cut['id']}", (1,)),
            "battery_q_delta": variables.add(f"security_battery_q_{cut['id']}", (1,)),
            "grid_up": variables.add(f"security_grid_up_{cut['id']}", (1,)),
            "demand_response": variables.add(f"security_dr_{cut['id']}", (len(buses),)),
        })

    objective = np.zeros(variables.size)
    bounds: list[tuple[float | None, float | None]] = [(None, None)] * variables.size
    equalities = _LinearRows()
    inequalities = _LinearRows()

    e_min = battery.capacity_kwh * battery.min_soc
    e_max = battery.capacity_kwh * battery.max_soc
    v_min = (case["voltage_min_pu"] + voltage_margin_pu) ** 2
    v_max = case["voltage_max_pu"] ** 2
    z_base_ohm = case["base_kv"] ** 2

    line_limits_kva: list[float] = []
    for line in lines:
        multiplier = line_rating_multiplier if line["from"] == 1 and line["to"] == 2 else 1.0
        line_limits_kva.append(sqrt(3.0) * case["base_kv"] * line["max_ka"] * multiplier * 1000.0)

    for hour in range(hours):
        objective[grid[hour]] = price[hour]
        objective[charge[hour]] = 0.055
        objective[discharge[hour]] = 0.055
        bounds[grid[hour]] = (0.0, float(grid_limits_kw[hour]))
        bounds[q_grid[hour]] = (-10000.0, 10000.0)
        bounds[charge[hour]] = (0.0, battery.power_kw)
        bounds[discharge[hour]] = (0.0, battery.power_kw)
        bounds[battery_q[hour]] = (-battery.power_kw, battery.power_kw)
        # Storage-inverter capability polygon: |P_battery| + |Q_battery| <= S_rated.
        for p_sign, q_sign in ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)):
            inequalities.add(
                {
                    discharge[hour]: p_sign,
                    charge[hour]: -p_sign,
                    battery_q[hour]: q_sign,
                },
                battery.power_kw,
            )
        soc_lower = e_min
        if hour == hours - 1 and terminal_min_energy_kwh is not None:
            soc_lower = max(soc_lower, terminal_min_energy_kwh)
        bounds[soc[hour]] = (soc_lower, e_max)

        for site_index, site in enumerate(pv_sites):
            available = pv_total_kw[hour] * site["capacity_kw"] / total_pv_capacity
            objective[pv_used[hour, site_index]] = -0.035
            bounds[pv_used[hour, site_index]] = (0.0, float(available))

        for bus_index, bus_row in enumerate(case["buses"]):
            load_p = bus_row["p_kw"] * load_scale[hour]
            objective[shed[hour, bus_index]] = 15.0
            bounds[shed[hour, bus_index]] = (0.0, float(load_p))
            if bus_row["bus"] == 1:
                bounds[voltage_sq[hour, bus_index]] = (source_voltage_sq, source_voltage_sq)
            else:
                bounds[voltage_sq[hour, bus_index]] = (v_min, v_max)

        for line_index, limit in enumerate(line_limits_kva):
            bounds[p_flow[hour, line_index]] = (-limit, limit)
            bounds[q_flow[hour, line_index]] = (-limit, limit)
            inequalities.add({p_flow[hour, line_index]: 1.0, q_flow[hour, line_index]: 1.0}, limit)
            inequalities.add({p_flow[hour, line_index]: 1.0, q_flow[hour, line_index]: -1.0}, limit)
            inequalities.add({p_flow[hour, line_index]: -1.0, q_flow[hour, line_index]: 1.0}, limit)
            inequalities.add({p_flow[hour, line_index]: -1.0, q_flow[hour, line_index]: -1.0}, limit)

        soc_coefficients = {
            soc[hour]: 1.0,
            charge[hour]: -battery.eta_charge,
            discharge[hour]: 1.0 / battery.eta_discharge,
        }
        if hour == 0:
            equalities.add(soc_coefficients, initial_energy_kwh)
        else:
            soc_coefficients[soc[hour - 1]] = -1.0
            equalities.add(soc_coefficients, 0.0)

        for bus_index, bus_row in enumerate(case["buses"]):
            bus = bus_row["bus"]
            load_p = bus_row["p_kw"] * load_scale[hour]
            svg_q = min(
                reactive_support_by_bus.get(bus, 0.0),
                reactive_support_by_bus.get(bus, 0.0) * max(float(load_scale[hour]), 0.0) ** 2,
            )
            load_q = bus_row["q_kvar"] * load_scale[hour] - svg_q
            active: dict[int, float] = {shed[hour, bus_index]: 1.0}
            reactive: dict[int, float] = {}
            if bus == 1:
                active[grid[hour]] = 1.0
                reactive[q_grid[hour]] = 1.0
            else:
                incoming_line = line_position_by_child[bus]
                active[p_flow[hour, incoming_line]] = 1.0
                reactive[q_flow[hour, incoming_line]] = 1.0
            for outgoing_line in outgoing[bus]:
                active[p_flow[hour, outgoing_line]] = active.get(p_flow[hour, outgoing_line], 0.0) - 1.0
                reactive[q_flow[hour, outgoing_line]] = reactive.get(q_flow[hour, outgoing_line], 0.0) - 1.0
            for site_index in pv_sites_by_bus[bus]:
                active[pv_used[hour, site_index]] = 1.0
            if bus == case["battery"]["bus"]:
                active[discharge[hour]] = 1.0
                active[charge[hour]] = -1.0
                reactive[battery_q[hour]] = 1.0
            if load_p > 1e-9:
                reactive[shed[hour, bus_index]] = load_q / load_p
            equalities.add(active, load_p)
            equalities.add(reactive, load_q)

        for line_index, line in enumerate(lines):
            parent = bus_position[line["from"]]
            child = bus_position[line["to"]]
            r_pu = line["r_ohm"] / z_base_ohm
            x_pu = line["x_ohm"] / z_base_ohm
            equalities.add(
                {
                    voltage_sq[hour, child]: 1.0,
                    voltage_sq[hour, parent]: -1.0,
                    p_flow[hour, line_index]: 2.0 * r_pu / 1000.0,
                    q_flow[hour, line_index]: 2.0 * x_pu / 1000.0,
                },
                0.0,
            )

    # Constraint-generation cuts: reuse the scheduled injections while enforcing
    # power balance, voltage and thermal limits on the restored post-contingency tree.
    for model in security_models:
        hour = model["local_hour"]
        edges = model["edges"]
        p_security = model["p_flow"]
        q_security = model["q_flow"]
        v_security = model["voltage_sq"]
        battery_up = int(model["battery_up"][0])
        battery_down = int(model["battery_down"][0])
        battery_q_delta = int(model["battery_q_delta"][0])
        grid_up = int(model["grid_up"][0])
        demand_response = model["demand_response"]
        security_voltage_floor = float(model["cut"].get("target_min_voltage_pu", case["voltage_min_pu"])) ** 2
        security_loading_factor = float(model["cut"].get("target_max_line_loading_pct", 100.0)) / 100.0
        battery_response_kw = min(float(model["cut"].get("battery_response_kw", battery.power_kw * 0.35)), battery.power_kw)
        demand_response_fraction = float(model["cut"].get("demand_response_fraction", 0.08))
        grid_response_kw = float(model["cut"].get("grid_response_kw", 800.0))
        bounds[battery_up] = (0.0, battery_response_kw)
        bounds[battery_down] = (0.0, battery_response_kw)
        bounds[battery_q_delta] = (-battery_response_kw, battery_response_kw)
        bounds[grid_up] = (0.0, grid_response_kw)
        objective[battery_up] = 0.08
        objective[battery_down] = 0.08
        objective[grid_up] = float(price[hour]) * 1.15
        inequalities.add({grid[hour]: 1.0, grid_up: 1.0}, float(grid_limits_kw[hour]))

        # Accident-response capability: corrective storage action is independent
        # of the base dispatch but the combined inverter P/Q stays within rating.
        combined_p = {
            discharge[hour]: 1.0,
            charge[hour]: -1.0,
            battery_up: 1.0,
            battery_down: -1.0,
        }
        combined_q = {battery_q[hour]: 1.0, battery_q_delta: 1.0}
        for p_sign, q_sign in ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)):
            coefficients = {index: p_sign * value for index, value in combined_p.items()}
            for index, value in combined_q.items():
                coefficients[index] = coefficients.get(index, 0.0) + q_sign * value
            inequalities.add(coefficients, battery.power_kw)
        incoming_by_bus: dict[int, int] = {}
        outgoing_by_bus: dict[int, list[int]] = {bus: [] for bus in buses}
        limits: list[float] = []
        for edge_index, edge in enumerate(edges):
            incoming_by_bus[edge["to"]] = edge_index
            outgoing_by_bus[edge["from"]].append(edge_index)
            multiplier = line_rating_multiplier if {edge["from"], edge["to"]} == {1, 2} else 1.0
            limit = (
                sqrt(3.0)
                * case["base_kv"]
                * edge["max_ka"]
                * multiplier
                * security_loading_factor
                * 1000.0
            )
            limits.append(limit)
            bounds[p_security[edge_index]] = (-limit, limit)
            bounds[q_security[edge_index]] = (-limit, limit)
            inequalities.add({p_security[edge_index]: 1.0, q_security[edge_index]: 1.0}, limit)
            inequalities.add({p_security[edge_index]: 1.0, q_security[edge_index]: -1.0}, limit)
            inequalities.add({p_security[edge_index]: -1.0, q_security[edge_index]: 1.0}, limit)
            inequalities.add({p_security[edge_index]: -1.0, q_security[edge_index]: -1.0}, limit)

        for bus_index, bus in enumerate(buses):
            bounds[v_security[bus_index]] = (
                (source_voltage_sq, source_voltage_sq) if bus == 1 else (security_voltage_floor, v_max)
            )
            bus_row = case["buses"][bus_index]
            load_p = bus_row["p_kw"] * load_scale[hour]
            svg_q = min(
                reactive_support_by_bus.get(bus, 0.0),
                reactive_support_by_bus.get(bus, 0.0) * max(float(load_scale[hour]), 0.0) ** 2,
            )
            load_q = bus_row["q_kvar"] * load_scale[hour] - svg_q
            dr_limit = max(load_p * demand_response_fraction, 0.0)
            bounds[demand_response[bus_index]] = (0.0, dr_limit)
            objective[demand_response[bus_index]] = 8.0
            # Security cuts require full gross load to remain supplied after
            # restoration; base-case shedding cannot satisfy contingency cuts.
            active: dict[int, float] = {}
            reactive: dict[int, float] = {}
            if bus == 1:
                active[grid[hour]] = 1.0
                active[grid_up] = 1.0
                reactive[q_grid[hour]] = 1.0
            else:
                incoming = incoming_by_bus[bus]
                active[p_security[incoming]] = 1.0
                reactive[q_security[incoming]] = 1.0
            for outgoing_index in outgoing_by_bus[bus]:
                active[p_security[outgoing_index]] = active.get(p_security[outgoing_index], 0.0) - 1.0
                reactive[q_security[outgoing_index]] = reactive.get(q_security[outgoing_index], 0.0) - 1.0
            for site_index in pv_sites_by_bus[bus]:
                active[pv_used[hour, site_index]] = 1.0
            if bus == case["battery"]["bus"]:
                active[discharge[hour]] = 1.0
                active[charge[hour]] = -1.0
                active[battery_up] = 1.0
                active[battery_down] = -1.0
                reactive[battery_q[hour]] = 1.0
                reactive[battery_q_delta] = 1.0
            active[demand_response[bus_index]] = 1.0
            if load_p > 1e-9:
                reactive[demand_response[bus_index]] = load_q / load_p
            equalities.add(active, load_p)
            equalities.add(reactive, load_q)

        for edge_index, edge in enumerate(edges):
            parent = bus_position[edge["from"]]
            child = bus_position[edge["to"]]
            r_pu = edge["r_ohm"] / z_base_ohm
            x_pu = edge["x_ohm"] / z_base_ohm
            equalities.add(
                {
                    v_security[child]: 1.0,
                    v_security[parent]: -1.0,
                    p_security[edge_index]: 2.0 * r_pu / 1000.0,
                    q_security[edge_index]: 2.0 * x_pu / 1000.0,
                },
                0.0,
            )

    if terminal_value_yuan_per_kwh > 0:
        objective[soc[-1]] -= terminal_value_yuan_per_kwh

    result = linprog(
        objective,
        A_ub=inequalities.matrix(variables.size),
        b_ub=np.asarray(inequalities.rhs) if inequalities.rhs else None,
        A_eq=equalities.matrix(variables.size),
        b_eq=np.asarray(equalities.rhs),
        bounds=bounds,
        method="highs",
        options={"presolve": True},
    )
    if not result.success:
        raise RuntimeError(f"滚动窗口网络约束优化失败：{result.message}")

    solution = result.x
    rows: list[dict] = []
    for hour in range(hours):
        pv_available = float(pv_total_kw[hour])
        pv_used_total = float(sum(solution[pv_used[hour, site]] for site in range(len(pv_sites))))
        shed_total = float(sum(solution[shed[hour, bus]] for bus in range(len(buses))))
        charge_kw = float(solution[charge[hour]])
        discharge_kw = float(solution[discharge[hour]])
        predicted_voltages = np.sqrt(np.maximum(solution[voltage_sq[hour]], 0.0))
        line_loading = []
        for line_index, limit in enumerate(line_limits_kva):
            p_value = solution[p_flow[hour, line_index]]
            q_value = solution[q_flow[hour, line_index]]
            line_loading.append(100.0 * sqrt(p_value * p_value + q_value * q_value) / max(limit, 1e-9))
        stage_cost = (
            solution[grid[hour]] * price[hour]
            + (charge_kw + discharge_kw) * 0.055
            + (pv_available - pv_used_total) * 0.035
            + shed_total * 15.0
        )
        contingency_actions = []
        for model in security_models:
            if model["local_hour"] != hour:
                continue
            dr_total = float(sum(solution[index] for index in model["demand_response"]))
            contingency_actions.append({
                "constraint_id": model["cut"]["id"],
                "outage_line": model["cut"]["outage_line"],
                "restoration": model["cut"]["restoration_tie"],
                "restoration_type": model["cut"].get("restoration_type", "tie_line"),
                "battery_up_kw": float(solution[int(model["battery_up"][0])]),
                "battery_down_kw": float(solution[int(model["battery_down"][0])]),
                "battery_q_delta_kvar": float(solution[int(model["battery_q_delta"][0])]),
                "grid_up_kw": float(solution[int(model["grid_up"][0])]),
                "demand_response_kw": dr_total,
            })
        rows.append({
            "hour": hour,
            "load_kw": float(sum(entry["p_kw"] for entry in case["buses"]) * load_scale[hour]),
            "pv_available_kw": pv_available,
            "pv_used_kw": pv_used_total,
            "curtailed_kw": max(pv_available - pv_used_total, 0.0),
            "grid_kw": float(solution[grid[hour]]),
            "grid_q_kvar": float(solution[q_grid[hour]]),
            "battery_charge_kw": charge_kw,
            "battery_discharge_kw": discharge_kw,
            "battery_kw": discharge_kw - charge_kw,
            "battery_q_kvar": float(solution[battery_q[hour]]),
            "soc_kwh": float(solution[soc[hour]]),
            "soc_pct": float(100.0 * solution[soc[hour]] / battery.capacity_kwh),
            "shed_kw": max(shed_total, 0.0),
            "shed_by_bus_kw": {
                str(bus): max(float(solution[shed[hour, bus_index]]), 0.0)
                for bus_index, bus in enumerate(buses)
                if solution[shed[hour, bus_index]] > 1e-8
            },
            "pv_by_bus_kw": {
                str(site["bus"]): float(solution[pv_used[hour, site_index]])
                for site_index, site in enumerate(pv_sites)
            },
            "grid_limit_kw": float(grid_limits_kw[hour]),
            "price": float(price[hour]),
            "predicted_min_voltage_pu": float(np.min(predicted_voltages)),
            "predicted_max_line_loading_pct": float(max(line_loading, default=0.0)),
            "stage_cost": float(stage_cost),
            "active_security_constraints": [
                model["cut"]["id"] for model in security_models if model["local_hour"] == hour
            ],
            "contingency_actions": contingency_actions,
        })

    return {
        "rows": rows,
        "solver_objective": float(result.fun),
        "solver_iterations": int(result.nit),
        "variables": int(variables.size),
        "equalities": len(equalities.rhs),
        "inequalities": len(inequalities.rhs),
        "security_constraints": len(security_models),
    }


def optimize_rolling_dispatch(
    case: dict,
    load_scale: np.ndarray,
    pv_total_kw: np.ndarray,
    price: np.ndarray,
    battery: BatteryConfig,
    grid_limits_kw: np.ndarray,
    line_rating_multiplier: float = 1.0,
    horizon_hours: int = 12,
    voltage_margin_pu: float = 0.005,
    security_constraints: list[dict] | None = None,
) -> dict:
    """Network-constrained receding-horizon dispatch using LinDistFlow and HiGHS."""
    load_scale = np.asarray(load_scale, dtype=float)
    pv_total_kw = np.asarray(pv_total_kw, dtype=float)
    price = np.asarray(price, dtype=float)
    grid_limits_kw = np.asarray(grid_limits_kw, dtype=float)
    security_constraints = list(security_constraints or [])
    if not (len(load_scale) == len(pv_total_kw) == len(price) == len(grid_limits_kw)):
        raise ValueError("负荷、光伏、电价和电网限额长度必须一致")
    if np.any(load_scale < 0) or np.any(pv_total_kw < 0) or np.any(grid_limits_kw <= 0):
        raise ValueError("负荷系数/光伏不能为负，电网限额必须为正")

    horizon_total = len(load_scale)
    initial_energy = battery.capacity_kwh * battery.initial_soc
    current_energy = initial_energy
    implemented_rows: list[dict] = []
    diagnostics: list[dict] = []

    for start in range(horizon_total):
        stop = min(start + horizon_hours, horizon_total)
        if stop == horizon_total:
            terminal_min = initial_energy
            terminal_value = 0.0
        else:
            terminal_min = None
            look_ahead_stop = min(stop + horizon_hours, horizon_total)
            future_peak_price = float(np.max(price[stop:look_ahead_stop])) if look_ahead_stop > stop else float(price[stop - 1])
            terminal_value = future_peak_price * battery.eta_discharge

        window = _solve_window(
            case=case,
            load_scale=load_scale[start:stop],
            pv_total_kw=pv_total_kw[start:stop],
            price=price[start:stop],
            battery=battery,
            grid_limits_kw=grid_limits_kw[start:stop],
            initial_energy_kwh=current_energy,
            line_rating_multiplier=line_rating_multiplier,
            voltage_margin_pu=voltage_margin_pu,
            terminal_min_energy_kwh=terminal_min,
            terminal_value_yuan_per_kwh=terminal_value,
            window_start_hour=start,
            security_constraints=security_constraints,
        )
        implemented = dict(window["rows"][0])
        implemented["hour"] = start
        implemented_rows.append(implemented)
        current_energy = implemented["soc_kwh"]
        diagnostics.append({
            "start_hour": start,
            "stop_hour": stop,
            "variables": window["variables"],
            "equalities": window["equalities"],
            "inequalities": window["inequalities"],
            "solver_iterations": window["solver_iterations"],
            "security_constraints": window["security_constraints"],
        })

    objective = float(sum(row["stage_cost"] for row in implemented_rows))
    return {
        "schedule": implemented_rows,
        "objective_yuan": objective,
        "terminal_soc_pct": float(100.0 * current_energy / battery.capacity_kwh),
        "method": "滚动时域 LinDistFlow 网络约束优化（HiGHS）",
        "horizon_hours": horizon_hours,
        "rolling_windows": horizon_total,
        "solver": "SciPy HiGHS linear programming",
        "diagnostics": diagnostics,
        "voltage_margin_pu": voltage_margin_pu,
        "security_constraints": security_constraints,
        "security_constraint_count": len(security_constraints),
    }
