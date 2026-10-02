from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .data import load_case, scenario_config


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


class _Rows:
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


def _add_scaled(target: np.ndarray, expression: dict[int, float], scale: float) -> None:
    for index, value in expression.items():
        target[index] += scale * value


def _tail_mean(values: np.ndarray, confidence: float) -> float:
    threshold = float(np.quantile(values, confidence, method="higher"))
    tail = values[values >= threshold - 1e-9]
    return float(np.mean(tail))


@dataclass(frozen=True)
class _SolveResult:
    solution: np.ndarray
    first_stage: dict[str, np.ndarray]
    scenario_costs: np.ndarray
    method: dict
    evidence: dict


def _solve_extensive_form(
    result: dict,
    scenarios: dict,
    confidence: float,
    risk_weight: float,
    mode: str,
    fixed_first_stage: dict[str, np.ndarray] | None = None,
) -> _SolveResult:
    """Solve a two-stage deterministic-equivalent LP.

    First-stage variables are created once and therefore are non-anticipative by
    construction. Every forecast scenario owns a separate recourse block.
    """

    load = np.asarray(scenarios["load"], dtype=float)
    pv = np.asarray(scenarios["pv"], dtype=float)
    price = np.asarray(scenarios["price"], dtype=float)
    scenario_count, hours = load.shape
    probability = 1.0 / scenario_count
    schedule = sorted(result["schedule"], key=lambda row: int(row["hour"]))
    base_load = np.asarray(scenarios["base_load"], dtype=float)
    base_pv = np.asarray(scenarios["base_pv"], dtype=float)
    base_price = np.asarray(scenarios["base_price"], dtype=float)
    grid_limit = np.asarray([row["grid_limit_kw"] for row in schedule], dtype=float)

    case = load_case()
    config = scenario_config(result["scenario"])
    battery_row = case["battery"]
    battery_power = float(battery_row["power_kw"] if config["battery_available"] else 0.0)
    capacity = float(battery_row["capacity_kwh"])
    initial_energy = capacity * float(battery_row["initial_soc"])
    minimum_energy = capacity * float(battery_row["min_soc"])
    maximum_energy = capacity * float(battery_row["max_soc"])
    eta_charge = float(battery_row["eta_charge"])
    eta_discharge = float(battery_row["eta_discharge"])

    power_flow = sorted(result["power_flow"], key=lambda row: int(row["hour"]))
    base_voltage = np.asarray([row["min_voltage_pu"] for row in power_flow], dtype=float)
    base_loading = np.asarray([row["max_line_loading_pct"] for row in power_flow], dtype=float)
    reference_feeder_demand = np.asarray(
        [
            row["load_kw"]
            - row.get("shed_kw", 0.0)
            - row["pv_used_kw"]
            - row["battery_kw"]
            for row in schedule
        ],
        dtype=float,
    )

    variables = _Variables()
    # Here-and-now decisions: one common block shared by every scenario.
    day_ahead_grid = variables.add("day_ahead_grid", (hours,))
    base_charge = variables.add("base_charge", (hours,))
    base_discharge = variables.add("base_discharge", (hours,))
    base_soc = variables.add("base_soc", (hours,))
    reserve_up = variables.add("reserve_up", (hours,))
    reserve_down = variables.add("reserve_down", (hours,))
    base_curtailment = variables.add("base_curtailment", (hours,))
    base_shed = variables.add("base_shed", (hours,))

    # Wait-and-see decisions: an independent block for each realization.
    grid_up = variables.add("grid_up", (scenario_count, hours))
    grid_down = variables.add("grid_down", (scenario_count, hours))
    scenario_charge = variables.add("scenario_charge", (scenario_count, hours))
    scenario_discharge = variables.add("scenario_discharge", (scenario_count, hours))
    scenario_soc = variables.add("scenario_soc", (scenario_count, hours))
    curtailment = variables.add("curtailment", (scenario_count, hours))
    demand_response = variables.add("demand_response", (scenario_count, hours))
    load_shed = variables.add("load_shed", (scenario_count, hours))
    voltage_slack = variables.add("voltage_slack", (scenario_count, hours))
    thermal_slack = variables.add("thermal_slack", (scenario_count, hours))
    value_at_risk = variables.add("value_at_risk", (1,))
    cvar_excess = variables.add("cvar_excess", (scenario_count,))
    worst_cost = variables.add("worst_cost", (1,))

    bounds: list[tuple[float | None, float | None]] = [(0.0, None)] * variables.size
    equalities = _Rows()
    inequalities = _Rows()

    first_blocks = {
        "day_ahead_grid": day_ahead_grid,
        "base_charge": base_charge,
        "base_discharge": base_discharge,
        "base_soc": base_soc,
        "reserve_up": reserve_up,
        "reserve_down": reserve_down,
        "base_curtailment": base_curtailment,
        "base_shed": base_shed,
    }
    for hour in range(hours):
        bounds[day_ahead_grid[hour]] = (0.0, float(grid_limit[hour]))
        bounds[base_charge[hour]] = (0.0, battery_power)
        bounds[base_discharge[hour]] = (0.0, battery_power)
        bounds[base_soc[hour]] = (minimum_energy, maximum_energy)
        bounds[reserve_up[hour]] = (0.0, battery_power)
        bounds[reserve_down[hour]] = (0.0, battery_power)
        bounds[base_curtailment[hour]] = (0.0, float(base_pv[hour]))
        bounds[base_shed[hour]] = (0.0, float(base_load[hour]))
        equalities.add(
            {
                day_ahead_grid[hour]: 1.0,
                base_discharge[hour]: 1.0,
                base_charge[hour]: -1.0,
                base_curtailment[hour]: -1.0,
                base_shed[hour]: 1.0,
            },
            float(base_load[hour] - base_pv[hour]),
        )
        soc_row = {
            base_soc[hour]: 1.0,
            base_charge[hour]: -eta_charge,
            base_discharge[hour]: 1.0 / eta_discharge,
        }
        if hour:
            soc_row[base_soc[hour - 1]] = -1.0
            equalities.add(soc_row, 0.0)
        else:
            equalities.add(soc_row, initial_energy)
        inequalities.add({base_charge[hour]: 1.0, base_discharge[hour]: 1.0}, battery_power)
        inequalities.add(
            {base_discharge[hour]: 1.0, base_charge[hour]: -1.0, reserve_up[hour]: 1.0},
            battery_power,
        )
        inequalities.add(
            {base_charge[hour]: 1.0, base_discharge[hour]: -1.0, reserve_down[hour]: 1.0},
            battery_power,
        )
        inequalities.add(
            {reserve_up[hour]: 1.0, base_soc[hour]: -eta_discharge},
            -eta_discharge * minimum_energy,
        )
        inequalities.add(
            {reserve_down[hour]: eta_charge, base_soc[hour]: 1.0},
            maximum_energy,
        )
    inequalities.add({base_soc[-1]: -1.0}, -initial_energy)

    if fixed_first_stage is not None:
        for name, block in first_blocks.items():
            values = np.asarray(fixed_first_stage[name], dtype=float).reshape(block.shape)
            for index, value in zip(block.flat, values.flat, strict=True):
                bounds[int(index)] = (float(value), float(value))

    first_stage_cost: dict[int, float] = {}
    for hour in range(hours):
        first_stage_cost[int(day_ahead_grid[hour])] = float(base_price[hour])
        first_stage_cost[int(base_charge[hour])] = 0.055
        first_stage_cost[int(base_discharge[hour])] = 0.055
        first_stage_cost[int(reserve_up[hour])] = 0.065
        first_stage_cost[int(reserve_down[hour])] = 0.035
        first_stage_cost[int(base_curtailment[hour])] = 0.035
        first_stage_cost[int(base_shed[hour])] = 15.0

    voltage_sensitivity = 0.000030
    loading_sensitivity = 0.018
    scenario_cost_expressions: list[dict[int, float]] = []
    for scenario in range(scenario_count):
        scenario_cost = dict(first_stage_cost)
        for hour in range(hours):
            bounds[grid_up[scenario, hour]] = (0.0, float(grid_limit[hour]))
            bounds[grid_down[scenario, hour]] = (0.0, float(grid_limit[hour]))
            bounds[scenario_charge[scenario, hour]] = (0.0, battery_power)
            bounds[scenario_discharge[scenario, hour]] = (0.0, battery_power)
            bounds[scenario_soc[scenario, hour]] = (minimum_energy, maximum_energy)
            bounds[curtailment[scenario, hour]] = (0.0, float(pv[scenario, hour]))
            bounds[demand_response[scenario, hour]] = (0.0, 0.08 * float(load[scenario, hour]))
            bounds[load_shed[scenario, hour]] = (0.0, float(load[scenario, hour]))
            bounds[voltage_slack[scenario, hour]] = (0.0, 0.0)
            bounds[thermal_slack[scenario, hour]] = (0.0, 0.0)

            equalities.add(
                {
                    day_ahead_grid[hour]: 1.0,
                    grid_up[scenario, hour]: 1.0,
                    grid_down[scenario, hour]: -1.0,
                    scenario_discharge[scenario, hour]: 1.0,
                    scenario_charge[scenario, hour]: -1.0,
                    curtailment[scenario, hour]: -1.0,
                    demand_response[scenario, hour]: 1.0,
                    load_shed[scenario, hour]: 1.0,
                },
                float(load[scenario, hour] - pv[scenario, hour]),
            )
            soc_row = {
                scenario_soc[scenario, hour]: 1.0,
                scenario_charge[scenario, hour]: -eta_charge,
                scenario_discharge[scenario, hour]: 1.0 / eta_discharge,
            }
            if hour:
                soc_row[scenario_soc[scenario, hour - 1]] = -1.0
                equalities.add(soc_row, 0.0)
            else:
                equalities.add(soc_row, initial_energy)

            inequalities.add(
                {
                    day_ahead_grid[hour]: 1.0,
                    grid_up[scenario, hour]: 1.0,
                    grid_down[scenario, hour]: -1.0,
                },
                float(grid_limit[hour]),
            )
            inequalities.add(
                {
                    day_ahead_grid[hour]: -1.0,
                    grid_up[scenario, hour]: -1.0,
                    grid_down[scenario, hour]: 1.0,
                },
                0.0,
            )
            inequalities.add(
                {scenario_charge[scenario, hour]: 1.0, scenario_discharge[scenario, hour]: 1.0},
                battery_power,
            )
            # Recourse battery deviation cannot exceed capacity reserved day-ahead.
            inequalities.add(
                {
                    scenario_discharge[scenario, hour]: 1.0,
                    scenario_charge[scenario, hour]: -1.0,
                    base_discharge[hour]: -1.0,
                    base_charge[hour]: 1.0,
                    reserve_up[hour]: -1.0,
                },
                0.0,
            )
            inequalities.add(
                {
                    base_discharge[hour]: 1.0,
                    base_charge[hour]: -1.0,
                    scenario_discharge[scenario, hour]: -1.0,
                    scenario_charge[scenario, hour]: 1.0,
                    reserve_down[hour]: -1.0,
                },
                0.0,
            )

            feeder_constant = float(load[scenario, hour] - pv[scenario, hour] - reference_feeder_demand[hour])
            feeder_variables = {
                curtailment[scenario, hour]: 1.0,
                scenario_charge[scenario, hour]: 1.0,
                scenario_discharge[scenario, hour]: -1.0,
                demand_response[scenario, hour]: -1.0,
                load_shed[scenario, hour]: -1.0,
            }
            voltage_row = {index: voltage_sensitivity * value for index, value in feeder_variables.items()}
            voltage_row[voltage_slack[scenario, hour]] = -1.0
            inequalities.add(
                voltage_row,
                float(base_voltage[hour] - 0.95 - voltage_sensitivity * feeder_constant),
            )
            thermal_row = {index: loading_sensitivity * value for index, value in feeder_variables.items()}
            thermal_row[thermal_slack[scenario, hour]] = -1.0
            inequalities.add(
                thermal_row,
                float(100.0 - base_loading[hour] - loading_sensitivity * feeder_constant),
            )

            scenario_cost[int(grid_up[scenario, hour])] = float(price[scenario, hour]) * 1.10
            scenario_cost[int(grid_down[scenario, hour])] = -float(price[scenario, hour]) * 0.65
            scenario_cost[int(scenario_charge[scenario, hour])] = 0.025
            scenario_cost[int(scenario_discharge[scenario, hour])] = 0.025
            scenario_cost[int(curtailment[scenario, hour])] = 0.035
            scenario_cost[int(demand_response[scenario, hour])] = 8.0
            scenario_cost[int(load_shed[scenario, hour])] = 15.0
            scenario_cost[int(voltage_slack[scenario, hour])] = 0.0
            scenario_cost[int(thermal_slack[scenario, hour])] = 0.0
        inequalities.add({scenario_soc[scenario, -1]: -1.0}, -initial_energy)
        scenario_cost_expressions.append(scenario_cost)

    objective = np.zeros(variables.size, dtype=float)
    if mode == "stochastic":
        for expression in scenario_cost_expressions:
            _add_scaled(objective, expression, (1.0 - risk_weight) * probability)
        objective[value_at_risk[0]] = risk_weight
        for scenario, expression in enumerate(scenario_cost_expressions):
            objective[cvar_excess[scenario]] = risk_weight * probability / max(1.0 - confidence, 1e-6)
            cvar_row = dict(expression)
            cvar_row[int(value_at_risk[0])] = -1.0
            cvar_row[int(cvar_excess[scenario])] = -1.0
            inequalities.add(cvar_row, 0.0)
        bounds[value_at_risk[0]] = (None, None)
        bounds[worst_cost[0]] = (0.0, 0.0)
    elif mode == "robust":
        objective[worst_cost[0]] = 1.0
        for expression in scenario_cost_expressions:
            robust_row = dict(expression)
            robust_row[int(worst_cost[0])] = -1.0
            inequalities.add(robust_row, 0.0)
            _add_scaled(objective, expression, probability * 1e-6)
        bounds[value_at_risk[0]] = (0.0, 0.0)
        for index in cvar_excess:
            bounds[index] = (0.0, 0.0)
    else:
        for expression in scenario_cost_expressions:
            _add_scaled(objective, expression, probability)
        bounds[value_at_risk[0]] = (0.0, 0.0)
        bounds[worst_cost[0]] = (0.0, 0.0)
        for index in cvar_excess:
            bounds[index] = (0.0, 0.0)

    solved = linprog(
        objective,
        A_ub=inequalities.matrix(variables.size),
        b_ub=np.asarray(inequalities.rhs, dtype=float),
        A_eq=equalities.matrix(variables.size),
        b_eq=np.asarray(equalities.rhs, dtype=float),
        bounds=bounds,
        method="highs",
        options={"presolve": True},
    )
    if not solved.success:
        raise RuntimeError(f"两阶段场景优化失败：{solved.message}")

    solution = np.asarray(solved.x, dtype=float)
    costs = np.asarray(
        [sum(value * solution[index] for index, value in expression.items()) for expression in scenario_cost_expressions],
        dtype=float,
    )
    first_cost = float(sum(value * solution[index] for index, value in first_stage_cost.items()))
    loss_events = np.sum(solution[load_shed], axis=1) > 0.1
    scenario_voltage_shortfall = np.max(solution[voltage_slack], axis=1)
    scenario_thermal_overload = np.max(solution[thermal_slack], axis=1)
    # Ignore sub-0.0005 pu / sub-0.05 percentage-point numerical or calibration noise.
    voltage_events = scenario_voltage_shortfall > 5e-4
    thermal_events = scenario_thermal_overload > 0.05
    first_stage = {name: solution[block].copy() for name, block in first_blocks.items()}
    method = {
        "expected_cost_yuan": float(np.mean(costs)),
        "cvar_yuan": _tail_mean(costs, confidence),
        "loss_of_load_probability_pct": 100.0 * float(np.mean(loss_events)),
        "voltage_violation_probability_pct": 100.0 * float(np.mean(voltage_events)),
        "network_violation_probability_pct": 100.0 * float(np.mean(voltage_events | thermal_events)),
        "expected_unserved_energy_kwh": float(np.mean(np.sum(solution[load_shed], axis=1))),
        "expected_demand_response_kwh": float(np.mean(np.sum(solution[demand_response], axis=1))),
        "expected_voltage_shortfall_pu": float(np.mean(scenario_voltage_shortfall)),
        "max_voltage_shortfall_pu": float(np.max(scenario_voltage_shortfall)),
        "max_thermal_overload_pct": float(np.max(scenario_thermal_overload)),
        "reserve_energy_kwh": float(np.sum(solution[reserve_up])),
        "peak_reserve_kw": float(np.max(solution[reserve_up])),
        "first_stage_cost_yuan": first_cost,
        "expected_recourse_cost_yuan": float(np.mean(costs) - first_cost),
        "worst_scenario": int(np.argmax(costs)) + 1,
        "worst_scenario_cost_yuan": float(np.max(costs)),
        "solver_status": "optimal",
        "solver_iterations": int(solved.nit),
        "day_ahead_grid_kwh": float(np.sum(solution[day_ahead_grid])),
        "first_stage_grid_kw": solution[day_ahead_grid].round(3).tolist(),
        "reserve_up_kw": solution[reserve_up].round(3).tolist(),
        "reserve_down_kw": solution[reserve_down].round(3).tolist(),
        "scenario_costs_yuan": costs.round(2).tolist(),
    }
    first_stage_variable_count = sum(block.size for block in first_blocks.values())
    evidence = {
        "formulation": "two-stage deterministic-equivalent linear program",
        "solver": "SciPy HiGHS",
        "mode": mode,
        "scenario_count": scenario_count,
        "scenario_probability": round(probability, 6),
        "first_stage_variable_count": int(first_stage_variable_count),
        "second_stage_variable_count_per_scenario": int(10 * hours),
        "total_variable_count": int(variables.size),
        "equality_count": int(len(equalities.rhs)),
        "inequality_count": int(len(inequalities.rhs)),
        "implicit_nonanticipativity_links": int(first_stage_variable_count * max(scenario_count - 1, 0)),
        "risk_weight": float(risk_weight if mode == "stochastic" else (1.0 if mode == "robust" else 0.0)),
        "confidence_pct": float(confidence * 100.0),
        "network_model": "scenario-indexed AC-calibrated voltage and thermal security envelopes",
        "first_stage_decisions": ["day-ahead grid purchase", "battery base schedule and SOC", "up/down reserve capacity"],
        "second_stage_decisions": [
            "grid balancing",
            "battery recourse and scenario SOC",
            "PV curtailment",
            "demand response",
            "load shedding",
            "network violation slack",
        ],
    }
    return _SolveResult(solution, first_stage, costs, method, evidence)


def solve_two_stage_methods(result: dict, scenarios: dict, params: dict) -> tuple[list[dict], dict]:
    """Solve deterministic, risk-averse stochastic, and robust policies."""

    confidence = float(params["confidence_pct"]) / 100.0
    base_only = {
        **scenarios,
        "load": np.asarray(scenarios["base_load"], dtype=float)[None, :],
        "pv": np.asarray(scenarios["base_pv"], dtype=float)[None, :],
        "price": np.asarray(scenarios["base_price"], dtype=float)[None, :],
    }
    nominal = _solve_extensive_form(result, base_only, confidence, 0.0, "expected")
    deterministic = _solve_extensive_form(
        result,
        scenarios,
        confidence,
        0.0,
        "expected",
        fixed_first_stage=nominal.first_stage,
    )
    stochastic = _solve_extensive_form(
        result,
        scenarios,
        confidence,
        float(params["risk_aversion"]),
        "stochastic",
    )
    robust = _solve_extensive_form(result, scenarios, confidence, 1.0, "robust")

    methods: list[dict] = []
    for solved, key, label in (
        (deterministic, "deterministic", "确定性日前计划"),
        (stochastic, "stochastic", "两阶段随机优化"),
        (robust, "robust", "最坏场景鲁棒优化"),
    ):
        method = {**solved.method, "key": key, "label": label}
        for field in (
            "expected_cost_yuan",
            "cvar_yuan",
            "worst_scenario_cost_yuan",
            "first_stage_cost_yuan",
            "expected_recourse_cost_yuan",
        ):
            method[field] = round(float(method[field]), 2)
        for field in (
            "loss_of_load_probability_pct",
            "voltage_violation_probability_pct",
            "network_violation_probability_pct",
        ):
            method[field] = round(float(method[field]), 2)
        for field in (
            "expected_unserved_energy_kwh",
            "expected_demand_response_kwh",
            "reserve_energy_kwh",
            "peak_reserve_kw",
            "day_ahead_grid_kwh",
            "expected_voltage_shortfall_pu",
            "max_voltage_shortfall_pu",
            "max_thermal_overload_pct",
        ):
            method[field] = round(float(method[field]), 3)
        methods.append(method)

    evidence = dict(stochastic.evidence)
    evidence.update(
        {
            "deterministic_policy": "nominal forecast optimized first, then fixed and evaluated across all scenarios",
            "stochastic_policy": "shared first-stage plan plus scenario recourse with expected-cost/CVaR objective",
            "robust_policy": "shared first-stage plan minimizing the maximum scenario cost",
            "reference": "PyPSA-style deterministic equivalent and Rockafellar-Uryasev CVaR linearization",
        }
    )
    return methods, evidence
