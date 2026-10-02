from __future__ import annotations

from statistics import NormalDist

import numpy as np


DEFAULTS = {
    "normal": {
        "pv_error_pct": 18.0,
        "load_error_pct": 7.0,
        "price_error_pct": 10.0,
        "risk_aversion": 0.45,
        "reason": "常规运行采用中等预测误差和均衡风险偏好。",
    },
    "pv_error": {
        "pv_error_pct": 30.0,
        "load_error_pct": 8.0,
        "price_error_pct": 12.0,
        "risk_aversion": 0.65,
        "reason": "光伏预测偏差场景提高光伏波动并增加尾部风险权重。",
    },
    "battery_outage": {
        "pv_error_pct": 24.0,
        "load_error_pct": 11.0,
        "price_error_pct": 15.0,
        "risk_aversion": 0.82,
        "reason": "储能退出后快速备用不足，Agent 提高负荷误差与风险厌恶。",
    },
    "line_derating": {
        "pv_error_pct": 22.0,
        "load_error_pct": 10.0,
        "price_error_pct": 14.0,
        "risk_aversion": 0.88,
        "reason": "主干线路降额时安全裕度较小，采用更保守的风险参数。",
    },
}


def recommended_parameters(scenario: str) -> dict:
    preset = dict(DEFAULTS.get(scenario, DEFAULTS["normal"]))
    preset.update(
        {
            "scenario_count": 20,
            "confidence_pct": 95.0,
            "distribution": "student_t",
            "temporal_correlation": 0.72,
            "seed": 20261002,
            "data_source": "agent_historical_error_template",
        }
    )
    return preset


def _validated_parameters(scenario: str, supplied: dict | None) -> dict:
    params = recommended_parameters(scenario)
    if supplied:
        params.update({key: value for key, value in supplied.items() if key in params})
    params["scenario_count"] = int(np.clip(int(params["scenario_count"]), 10, 100))
    params["confidence_pct"] = float(np.clip(float(params["confidence_pct"]), 80.0, 99.0))
    params["pv_error_pct"] = float(np.clip(float(params["pv_error_pct"]), 0.0, 60.0))
    params["load_error_pct"] = float(np.clip(float(params["load_error_pct"]), 0.0, 40.0))
    params["price_error_pct"] = float(np.clip(float(params["price_error_pct"]), 0.0, 60.0))
    params["risk_aversion"] = float(np.clip(float(params["risk_aversion"]), 0.0, 1.0))
    params["temporal_correlation"] = float(np.clip(float(params["temporal_correlation"]), 0.0, 0.95))
    params["seed"] = int(params["seed"])
    if params["distribution"] not in {"gaussian", "student_t", "bootstrap"}:
        params["distribution"] = "student_t"
    return params


def _innovations(rng: np.random.Generator, shape: tuple[int, int, int], distribution: str) -> np.ndarray:
    if distribution == "gaussian":
        return rng.normal(size=shape)
    if distribution == "student_t":
        return rng.standard_t(df=5, size=shape) / np.sqrt(5.0 / 3.0)
    residual_library = np.array(
        [-2.25, -1.65, -1.20, -0.90, -0.62, -0.38, -0.18, 0.0, 0.16, 0.34, 0.58, 0.86, 1.16, 1.58, 2.18]
    )
    return rng.choice(residual_library, size=shape, replace=True)


def _generate_scenarios(result: dict, params: dict) -> dict:
    schedule = result["schedule"]
    base_load = np.array([row["load_kw"] for row in schedule], dtype=float)
    base_pv = np.array([row["pv_available_kw"] for row in schedule], dtype=float)
    base_price = np.array([row["price"] for row in schedule], dtype=float)
    count = params["scenario_count"]
    rng = np.random.default_rng(params["seed"])
    raw = _innovations(rng, (count, len(schedule), 3), params["distribution"])
    correlation = np.array([[1.0, -0.28, -0.10], [-0.28, 1.0, 0.35], [-0.10, 0.35, 1.0]])
    correlated = raw @ np.linalg.cholesky(correlation).T
    rho = params["temporal_correlation"]
    errors = np.zeros_like(correlated)
    errors[:, 0, :] = correlated[:, 0, :]
    innovation_scale = np.sqrt(max(1.0 - rho * rho, 1e-9))
    for hour in range(1, len(schedule)):
        errors[:, hour, :] = rho * errors[:, hour - 1, :] + innovation_scale * correlated[:, hour, :]

    pv_multiplier = np.clip(1.0 + errors[:, :, 0] * params["pv_error_pct"] / 100.0, 0.05, 1.65)
    load_multiplier = np.clip(1.0 + errors[:, :, 1] * params["load_error_pct"] / 100.0, 0.65, 1.50)
    price_multiplier = np.clip(1.0 + errors[:, :, 2] * params["price_error_pct"] / 100.0, 0.35, 2.00)
    pv = pv_multiplier * base_pv[None, :]
    load = load_multiplier * base_load[None, :]
    price = price_multiplier * base_price[None, :]
    net_load = load - pv
    base_net = base_load - base_pv
    deviations = net_load - base_net[None, :]
    return {
        "base_load": base_load,
        "base_pv": base_pv,
        "base_price": base_price,
        "load": load,
        "pv": pv,
        "price": price,
        "net_load": net_load,
        "deviations": deviations,
        "errors": errors,
        "correlation_target": correlation,
    }


def _tail_mean(values: np.ndarray, confidence: float) -> float:
    threshold = float(np.quantile(values, confidence, method="higher"))
    tail = values[values >= threshold - 1e-9]
    return float(np.mean(tail))


def _evaluate_reserve_policy(result: dict, scenarios: dict, reserve: np.ndarray, confidence: float) -> dict:
    schedule = result["schedule"]
    base_cost = float(result["metrics"]["operating_cost_yuan"])
    base_grid = np.array([row["grid_kw"] for row in schedule], dtype=float)
    grid_limit = np.array([row["grid_limit_kw"] for row in schedule], dtype=float)
    base_voltage = np.array(
        [item["min_voltage_pu"] for item in sorted(result["power_flow"], key=lambda row: row["hour"])],
        dtype=float,
    )
    base_loading = np.array(
        [item["max_line_loading_pct"] for item in sorted(result["power_flow"], key=lambda row: row["hour"])],
        dtype=float,
    )
    deviations = scenarios["deviations"]
    price = scenarios["price"]
    grid_headroom = np.maximum(grid_limit - base_grid, 0.0)
    positive = np.maximum(deviations, 0.0)
    negative = np.maximum(-deviations, 0.0)
    # Reserve represents local corrective power from storage, demand response or flexible generation.
    local_response = np.minimum(positive, reserve[None, :])
    remaining = positive - local_response
    grid_response = np.minimum(remaining, grid_headroom[None, :])
    unserved = np.maximum(remaining - grid_response, 0.0)
    downward_grid = np.minimum(negative, base_grid[None, :])
    reserve_capacity_cost = 0.065 * float(np.sum(reserve))
    scenario_costs = (
        base_cost
        + reserve_capacity_cost
        + np.sum(local_response * price * 1.15, axis=1)
        + np.sum(grid_response * price, axis=1)
        - np.sum(downward_grid * price * 0.65, axis=1)
        + np.sum(unserved, axis=1) * 15.0
    )
    voltage = base_voltage[None, :] - remaining * 0.000030 + local_response * 0.000010
    loading = base_loading[None, :] + remaining * 0.018
    loss_events = np.sum(unserved, axis=1) > 0.1
    voltage_events = np.any(voltage < 0.95, axis=1)
    network_events = np.any((voltage < 0.95) | (loading > 100.0), axis=1)
    return {
        "costs": scenario_costs,
        "expected_cost_yuan": float(np.mean(scenario_costs)),
        "cvar_yuan": _tail_mean(scenario_costs, confidence),
        "loss_of_load_probability_pct": 100.0 * float(np.mean(loss_events)),
        "voltage_violation_probability_pct": 100.0 * float(np.mean(voltage_events)),
        "network_violation_probability_pct": 100.0 * float(np.mean(network_events)),
        "expected_unserved_energy_kwh": float(np.mean(np.sum(unserved, axis=1))),
        "reserve_energy_kwh": float(np.sum(reserve)),
        "peak_reserve_kw": float(np.max(reserve)),
        "worst_scenario": int(np.argmax(scenario_costs)) + 1,
    }


def _method_comparison(result: dict, scenarios: dict, params: dict) -> list[dict]:
    confidence = params["confidence_pct"] / 100.0
    positive = np.maximum(scenarios["deviations"], 0.0)
    deterministic_reserve = np.zeros(positive.shape[1])
    stochastic_envelope = np.quantile(positive, 0.90, axis=0)
    robust_reserve = np.max(positive, axis=0)

    deterministic = _evaluate_reserve_policy(result, scenarios, deterministic_reserve, confidence)
    deterministic.update({"key": "deterministic", "label": "确定性调度", "reserve_factor": 0.0})

    candidates: list[tuple[float, dict]] = []
    for factor in np.linspace(0.25, 1.0, 7):
        evaluated = _evaluate_reserve_policy(result, scenarios, stochastic_envelope * factor, confidence)
        risk_objective = (
            (1.0 - params["risk_aversion"]) * evaluated["expected_cost_yuan"]
            + params["risk_aversion"] * evaluated["cvar_yuan"]
            + 12.0 * evaluated["network_violation_probability_pct"]
            + 45.0 * evaluated["loss_of_load_probability_pct"]
        )
        candidates.append((risk_objective, {**evaluated, "reserve_factor": float(factor)}))
    stochastic = min(candidates, key=lambda item: item[0])[1]
    stochastic.update({"key": "stochastic", "label": "两阶段随机优化"})

    robust = _evaluate_reserve_policy(result, scenarios, robust_reserve, confidence)
    robust.update({"key": "robust", "label": "鲁棒优化", "reserve_factor": 1.0})

    methods = [deterministic, stochastic, robust]
    for method in methods:
        method["expected_cost_yuan"] = round(method["expected_cost_yuan"], 2)
        method["cvar_yuan"] = round(method["cvar_yuan"], 2)
        method["loss_of_load_probability_pct"] = round(method["loss_of_load_probability_pct"], 2)
        method["voltage_violation_probability_pct"] = round(method["voltage_violation_probability_pct"], 2)
        method["network_violation_probability_pct"] = round(method["network_violation_probability_pct"], 2)
        method["expected_unserved_energy_kwh"] = round(method["expected_unserved_energy_kwh"], 3)
        method["reserve_energy_kwh"] = round(method["reserve_energy_kwh"], 2)
        method["peak_reserve_kw"] = round(method["peak_reserve_kw"], 2)
        method.pop("costs", None)
    return methods


def _histogram(values: np.ndarray, bins: int = 12) -> dict:
    counts, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return {"centers": centers.round(3).tolist(), "counts": counts.astype(int).tolist()}


def analyze_uncertainty(result: dict, supplied_parameters: dict | None = None) -> dict:
    params = _validated_parameters(result["scenario"], supplied_parameters)
    generated = _generate_scenarios(result, params)
    confidence = params["confidence_pct"] / 100.0
    low_q = (1.0 - confidence) / 2.0
    high_q = 1.0 - low_q
    net = generated["net_load"]
    methods = _method_comparison(result, generated, params)
    selected = min(
        methods,
        key=lambda item: (
            (1.0 - params["risk_aversion"]) * item["expected_cost_yuan"]
            + params["risk_aversion"] * item["cvar_yuan"]
            + 12.0 * item["network_violation_probability_pct"]
            + 45.0 * item["loss_of_load_probability_pct"]
        ),
    )
    empirical_correlation = np.corrcoef(generated["errors"].reshape(-1, 3), rowvar=False)
    z_value = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    return {
        "parameters": params,
        "scenario_count": params["scenario_count"],
        "confidence_pct": params["confidence_pct"],
        "selected_method": selected["key"],
        "selected_method_label": selected["label"],
        "methods": methods,
        "hours": list(range(len(result["schedule"]))),
        "fan_chart": {
            "base": (generated["base_load"] - generated["base_pv"]).round(3).tolist(),
            "p_low": np.quantile(net, low_q, axis=0).round(3).tolist(),
            "p25": np.quantile(net, 0.25, axis=0).round(3).tolist(),
            "p50": np.quantile(net, 0.50, axis=0).round(3).tolist(),
            "p75": np.quantile(net, 0.75, axis=0).round(3).tolist(),
            "p_high": np.quantile(net, high_q, axis=0).round(3).tolist(),
        },
        "scenario_paths": net.round(3).tolist(),
        "pv_paths": generated["pv"].round(3).tolist(),
        "load_paths": generated["load"].round(3).tolist(),
        "price_paths": generated["price"].round(4).tolist(),
        "error_histograms": {
            "pv": _histogram(generated["errors"][:, :, 0].ravel() * params["pv_error_pct"]),
            "load": _histogram(generated["errors"][:, :, 1].ravel() * params["load_error_pct"]),
            "price": _histogram(generated["errors"][:, :, 2].ravel() * params["price_error_pct"]),
        },
        "correlation": empirical_correlation.round(3).tolist(),
        "distribution_summary": {
            "confidence_z": round(z_value, 3),
            "net_load_min_kw": round(float(np.min(net)), 2),
            "net_load_max_kw": round(float(np.max(net)), 2),
            "mean_absolute_error_kw": round(float(np.mean(np.abs(generated["deviations"]))), 2),
        },
        "agent_contract": {
            "replaceable_fields": [
                "pv_error_pct",
                "load_error_pct",
                "price_error_pct",
                "scenario_count",
                "confidence_pct",
                "risk_aversion",
                "distribution",
                "temporal_correlation",
                "seed",
            ],
            "api": "POST /api/uncertainty",
            "mode": "agent parameters or manual parameters use the same validated schema",
        },
    }
