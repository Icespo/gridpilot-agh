from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import LOAD_PROFILE, PRICE, PV_PROFILE, apply_scenario_investments, load_case, scenario_config, validate_case
from .optimizer import BatteryConfig, optimize_rolling_dispatch
from .pandapower_validation import run_line_n_1_assessment, validate_schedule_with_pandapower
from .security import build_topology_visualization, enrich_n_1_records, generate_n_1_security_constraints


def _trace(trace: list[dict], step: str, status: str, detail: str) -> None:
    trace.append({
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "step": step,
        "status": status,
        "detail": detail,
    })


def _battery_config(case: dict, available: bool) -> BatteryConfig:
    raw = case["battery"]
    return BatteryConfig(
        capacity_kwh=float(raw["capacity_kwh"]),
        power_kw=float(raw["power_kw"] if available else 0.0),
        initial_soc=float(raw["initial_soc"]),
        min_soc=float(raw["min_soc"]),
        max_soc=float(raw["max_soc"]),
        eta_charge=float(raw["eta_charge"]),
        eta_discharge=float(raw["eta_discharge"]),
    )


def _validate_schedule(case: dict, schedule: list[dict], load_multiplier: float, line_multiplier: float) -> list[dict]:
    return validate_schedule_with_pandapower(
        case,
        schedule,
        LOAD_PROFILE,
        load_multiplier,
        line_multiplier,
    )


def _metrics(schedule: list[dict], checks: list[dict], objective: float, baseline_cost: float) -> dict:
    grid = sum(row["grid_kw"] for row in schedule)
    optimized_energy_cost = sum(row["grid_kw"] * row["price"] for row in schedule)
    battery_throughput = sum(abs(row["battery_kw"]) for row in schedule)
    pv_available = sum(row["pv_available_kw"] for row in schedule)
    curtailed = sum(row["curtailed_kw"] for row in schedule)
    shed = sum(row["shed_kw"] for row in schedule)
    if shed < 0.1:  # numerical/discretization tolerance across the 24-hour horizon
        shed = 0.0
    operating_cost = optimized_energy_cost + battery_throughput * 0.055 + curtailed * 0.035 + shed * 15.0
    violation_hours = sum(
        bool(item["voltage_violations"] or item["thermal_violations"] or not item["converged"])
        for item in checks
    )
    return {
        "objective_yuan": round(objective, 2),
        "optimized_energy_cost_yuan": round(optimized_energy_cost, 2),
        "operating_cost_yuan": round(operating_cost, 2),
        "baseline_cost_yuan": round(baseline_cost, 2),
        "cost_reduction_pct": round(100.0 * (baseline_cost - operating_cost) / max(baseline_cost, 1e-9), 2),
        "grid_energy_kwh": round(grid, 2),
        "renewable_consumption_pct": round(100.0 * (pv_available - curtailed) / max(pv_available, 1e-9), 2),
        "curtailed_energy_kwh": round(curtailed, 2),
        "unserved_energy_kwh": round(shed, 2),
        "min_voltage_pu": round(min(item["min_voltage_pu"] for item in checks), 4),
        "max_line_loading_pct": round(max(item["max_line_loading_pct"] for item in checks), 2),
        "violation_hours": int(violation_hours),
        "carbon_kg": round(grid * 0.55, 2),
    }


def run_closed_loop(scenario: str = "normal", max_repairs: int = 4, include_n_1: bool = True) -> dict:
    trace: list[dict] = []
    _trace(trace, "task_received", "ok", f"收到 {scenario} 场景的光储调度与安全校核任务")
    case = apply_scenario_investments(load_case(), scenario)
    errors = validate_case(case)
    if errors:
        _trace(trace, "data_validation", "error", "; ".join(errors))
        raise ValueError("；".join(errors))
    _trace(trace, "data_validation", "ok", "33 个节点、32 条径向支路及设备上下限检查通过")

    cfg = scenario_config(scenario)
    if cfg.get("investment_plan"):
        _trace(
            trace,
            "investment_commissioning",
            "ok",
            "规划建设资产已投运：2条远端独立备用馈线、12段走廊增容、3组SVG及主变OLTC。",
        )
    total_base_load = sum(row["p_kw"] for row in case["buses"])
    pv_capacity = sum(site["capacity_kw"] for site in case["pv_sites"])
    load = total_base_load * LOAD_PROFILE * cfg["load_multiplier"]
    pv = pv_capacity * PV_PROFILE * cfg["pv_multiplier"]
    _trace(trace, "scenario_generation", "ok", cfg["description"])

    battery = _battery_config(case, cfg["battery_available"])
    grid_limits = np.full(24, 2900.0)
    baseline_grid = np.maximum(load - pv, 0.0)
    baseline_cost = float(np.sum(baseline_grid * PRICE))

    load_scale = LOAD_PROFILE * cfg["load_multiplier"]
    voltage_margin = 0.005
    line_safety_factor = 1.0
    dispatch = optimize_rolling_dispatch(
        case,
        load_scale,
        pv,
        PRICE,
        battery,
        grid_limits,
        line_rating_multiplier=cfg["line_rating_multiplier"] * line_safety_factor,
        horizon_hours=12,
        voltage_margin_pu=voltage_margin,
    )
    first_window = dispatch["diagnostics"][0]
    _trace(
        trace,
        "network_constrained_mpc",
        "ok",
        f"完成 {dispatch['rolling_windows']} 个滚动窗口；每个完整窗口约含 {first_window['variables']} 个变量、"
        f"{first_window['equalities']} 个等式和 {first_window['inequalities']} 个线路容量不等式",
    )
    checks = _validate_schedule(case, dispatch["schedule"], cfg["load_multiplier"], cfg["line_rating_multiplier"])
    violation_hours = [item["hour"] for item in checks if item["voltage_violations"] or item["thermal_violations"]]
    _trace(trace, "pandapower_ac_validation", "ok" if not violation_hours else "warning", f"pandapower牛顿-拉夫逊AC潮流发现 {len(violation_hours)} 个异常时段")

    repairs = 0
    while violation_hours and repairs < max_repairs:
        repairs += 1
        has_voltage_violation = any(item["voltage_violations"] for item in checks)
        has_thermal_violation = any(item["thermal_violations"] for item in checks)
        if has_voltage_violation:
            voltage_margin += 0.005
        if has_thermal_violation:
            line_safety_factor *= 0.95
        dispatch = optimize_rolling_dispatch(
            case,
            load_scale,
            pv,
            PRICE,
            battery,
            grid_limits,
            line_rating_multiplier=cfg["line_rating_multiplier"] * line_safety_factor,
            horizon_hours=12,
            voltage_margin_pu=voltage_margin,
        )
        checks = _validate_schedule(case, dispatch["schedule"], cfg["load_multiplier"], cfg["line_rating_multiplier"])
        violation_hours = [item["hour"] for item in checks if item["voltage_violations"] or item["thermal_violations"]]
        _trace(
            trace,
            "ac_feedback_reoptimization",
            "ok" if not violation_hours else "warning",
            f"第 {repairs} 轮反馈：电压裕度 {voltage_margin:.3f} pu、线路安全系数 {line_safety_factor:.3f}；"
            f"AC校核剩余 {len(violation_hours)} 个异常时段",
        )

    n_1 = None
    security_generation = None
    if include_n_1:
        initial_n_1 = run_line_n_1_assessment(
            case,
            dispatch["schedule"],
            LOAD_PROFILE,
            cfg["load_multiplier"],
            cfg["line_rating_multiplier"],
            checks,
        )
        enrich_n_1_records(initial_n_1, case)
        _trace(
            trace,
            "pandapower_n_1_screening",
            "ok" if initial_n_1["failed_contingencies"] == 0 else "warning",
            f"完成24小时×32条线路的全时段N-1校核，共筛查 {initial_n_1['contingencies_evaluated']} 个工况，识别 "
            f"{initial_n_1['failed_contingencies']} 个高风险工况",
        )
        current_n_1 = initial_n_1
        cumulative_constraints: list[dict] = []
        all_generated: list[dict] = []
        excluded_keys: set[tuple[int, str, str]] = set()
        iteration_history: list[dict] = []
        planning_actions: list[dict] = []
        planning_action_count = 0
        max_security_iterations = 3

        for iteration in range(1, max_security_iterations + 1):
            batch_result = generate_n_1_security_constraints(
                case,
                current_n_1,
                max_constraints=4,
                excluded_keys=excluded_keys,
                id_start=len(all_generated) + 1,
            )
            if iteration == 1:
                planning_actions = batch_result["planning_actions"]
                planning_action_count = batch_result["planning_actions_count"]
            batch = batch_result["constraints"]
            if not batch:
                break
            for cut in batch:
                excluded_keys.add((int(cut["hour"]), cut["outage_line"], cut["restoration_tie"]))

            before_risk = round(sum(float(item.get("risk_score", 0.0)) for item in current_n_1["records"]), 1)
            feasible_constraints: list[dict] | None = None
            secured_dispatch = None
            secured_checks = None
            for new_count in range(len(batch), 0, -1):
                candidate = cumulative_constraints + batch[:new_count]
                try:
                    candidate_dispatch = optimize_rolling_dispatch(
                        case,
                        load_scale,
                        pv,
                        PRICE,
                        battery,
                        grid_limits,
                        line_rating_multiplier=cfg["line_rating_multiplier"] * line_safety_factor,
                        horizon_hours=12,
                        voltage_margin_pu=voltage_margin,
                        security_constraints=candidate,
                    )
                except RuntimeError:
                    continue
                candidate_checks = _validate_schedule(
                    case,
                    candidate_dispatch["schedule"],
                    cfg["load_multiplier"],
                    cfg["line_rating_multiplier"],
                )
                if any(item["voltage_violations"] or item["thermal_violations"] for item in candidate_checks):
                    continue
                feasible_constraints = candidate
                secured_dispatch = candidate_dispatch
                secured_checks = candidate_checks
                break

            applied_ids = {item["id"] for item in (feasible_constraints or cumulative_constraints)}
            for cut in batch:
                cut["status"] = "applied" if cut["id"] in applied_ids else "deferred_infeasible"
            all_generated.extend(batch)
            if feasible_constraints is None or secured_dispatch is None or secured_checks is None:
                iteration_history.append({
                    "iteration": iteration,
                    "generated": len(batch),
                    "newly_applied": 0,
                    "status": "infeasible",
                    "before_failed": current_n_1["failed_contingencies"],
                    "after_failed": current_n_1["failed_contingencies"],
                    "before_risk_index": before_risk,
                    "after_risk_index": before_risk,
                })
                break

            newly_applied = len(feasible_constraints) - len(cumulative_constraints)
            cumulative_constraints = feasible_constraints
            dispatch = secured_dispatch
            checks = secured_checks
            reassessed = run_line_n_1_assessment(
                case,
                dispatch["schedule"],
                LOAD_PROFILE,
                cfg["load_multiplier"],
                cfg["line_rating_multiplier"],
                checks,
            )
            enrich_n_1_records(reassessed, case)
            after_risk = round(sum(float(item.get("risk_score", 0.0)) for item in reassessed["records"]), 1)
            iteration_history.append({
                "iteration": iteration,
                "generated": len(batch),
                "newly_applied": newly_applied,
                "cumulative_applied": len(cumulative_constraints),
                "status": "applied",
                "before_failed": current_n_1["failed_contingencies"],
                "after_failed": reassessed["failed_contingencies"],
                "before_risk_index": before_risk,
                "after_risk_index": after_risk,
            })
            _trace(
                trace,
                "security_constrained_reoptimization",
                "ok",
                f"SCOPF第{iteration}轮新增 {newly_applied} 条约束，累计 {len(cumulative_constraints)} 条；"
                f"高风险工况 {current_n_1['failed_contingencies']} → {reassessed['failed_contingencies']}，"
                f"风险指数 {before_risk:.1f} → {after_risk:.1f}",
            )
            current_n_1 = reassessed
            if newly_applied == 0 or reassessed["failed_contingencies"] == 0:
                break

        n_1 = current_n_1
        security_generation = {
            "method": "iterative pandapower screening + LinDistFlow SCOPF with corrective recourse",
            "screened_failed_contingencies": initial_n_1["failed_contingencies"],
            "operational_candidates": sum(1 for row in initial_n_1["records"] if not row.get("secure", False) and row.get("restoration_tie")),
            "generated_constraints": len(all_generated),
            "planning_actions_count": planning_action_count,
            "constraints": all_generated,
            "planning_actions": planning_actions,
            "applied_constraints": len(cumulative_constraints),
            "deferred_constraints": len(all_generated) - len(cumulative_constraints),
            "iterations": len(iteration_history),
            "iteration_history": iteration_history,
        }
        _trace(
            trace,
            "n_1_constraint_generation",
            "ok" if all_generated else "warning",
            f"迭代SCOPF执行 {len(iteration_history)} 轮，共生成 {len(all_generated)} 条约束、"
            f"应用 {len(cumulative_constraints)} 条，并启用事故后储能/需求响应/无功纠正变量",
        )
        security_generation["before_security_rate_pct"] = initial_n_1["security_rate_pct"]
        security_generation["before_failed_contingencies"] = initial_n_1["failed_contingencies"]
        security_generation["before_records"] = initial_n_1["records"]

        n_1["topology"] = build_topology_visualization(case)
        security_generation["after_security_rate_pct"] = n_1["security_rate_pct"]
        security_generation["after_failed_contingencies"] = n_1["failed_contingencies"]
        security_generation["risk_reduction_count"] = (
            initial_n_1["failed_contingencies"] - n_1["failed_contingencies"]
        )
        security_generation["after_records"] = n_1["records"]
        security_generation["before_risk_index"] = round(
            sum(float(item.get("risk_score", 0.0)) for item in initial_n_1["records"]), 1
        )
        security_generation["after_risk_index"] = round(
            sum(float(item.get("risk_score", 0.0)) for item in n_1["records"]), 1
        )
        security_generation["risk_index_reduction_pct"] = round(
            100.0
            * (security_generation["before_risk_index"] - security_generation["after_risk_index"])
            / max(security_generation["before_risk_index"], 1e-9),
            2,
        )
        _trace(
            trace,
            "pandapower_n_1_reassessment",
            "ok" if n_1["failed_contingencies"] == 0 else "warning",
            f"约束生成前后安全率 {initial_n_1['security_rate_pct']:.1f}% → {n_1['security_rate_pct']:.1f}%；"
            f"高风险工况 {initial_n_1['failed_contingencies']} → {n_1['failed_contingencies']}；"
            f"综合风险指数 {security_generation['before_risk_index']:.1f} → {security_generation['after_risk_index']:.1f}",
        )

    metrics = _metrics(dispatch["schedule"], checks, dispatch["objective_yuan"], baseline_cost)
    if n_1 is not None:
        metrics["n_1_contingencies"] = n_1["contingencies_evaluated"]
        metrics["n_1_failed_contingencies"] = n_1["failed_contingencies"]
        metrics["n_1_security_rate_pct"] = n_1["security_rate_pct"]
    if metrics["violation_hours"]:
        status = "needs_attention"
        conclusion = "仍存在网络安全越限，需要人工确认或进一步削减负荷。"
    elif metrics["unserved_energy_kwh"] > 1.0:
        status = "degraded"
        conclusion = "网络安全校核通过，但出现非零失负荷，系统处于降级运行。"
    elif n_1 is not None and n_1["failed_contingencies"] > 0:
        status = "degraded"
        conclusion = (
            "基态调度通过pandapower AC校核，但线路N-1枚举仍发现 "
            f"{n_1['failed_contingencies']} 个高风险工况，需增加联络容量或备用资源。"
        )
    else:
        status = "completed"
        conclusion = "调度方案通过功率平衡、储能边界和交流潮流安全校核。"
    _trace(trace, "result_summary", status, conclusion)

    return {
        "run_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project": "GridPilot-AGH",
        "scenario": scenario,
        "scenario_label": cfg["label"],
        "scenario_description": cfg["description"],
        "status": status,
        "conclusion": conclusion,
        "method": dispatch["method"] + (" + N-1约束生成" if security_generation and security_generation["applied_constraints"] else ""),
        "optimization": {
            "solver": dispatch["solver"],
            "horizon_hours": dispatch["horizon_hours"],
            "rolling_windows": dispatch["rolling_windows"],
            "voltage_margin_pu": dispatch["voltage_margin_pu"],
            "line_safety_factor": line_safety_factor,
            "diagnostics": dispatch["diagnostics"],
            "security_constraint_count": dispatch.get("security_constraint_count", 0),
        },
        "metrics": metrics,
        "schedule": dispatch["schedule"],
        "power_flow": checks,
        "n_1": n_1,
        "security_constraint_generation": security_generation,
        "trace": trace,
        "investment_plan": case.get("investment_plan"),
        "assumptions": [
            "单时段长度为 1 小时，光伏无功设为 0。",
            "调度层使用12小时滚动时域LinDistFlow网络约束优化，每小时仅执行窗口首个控制量。",
            "优化模型显式包含节点有功/无功平衡、支路潮流、节点电压、线路容量和储能SOC约束。",
            "优化后使用pandapower牛顿-拉夫逊AC潮流进行独立校核，并反馈收紧电压或线路安全裕度。",
            "线路N-1覆盖24个时段并逐线枚举32条运行线路，共形成768个故障工况；每个工况搜索能够跨接故障孤岛的常开联络线或独立备用馈线。",
            "可恢复高风险工况通过多轮约束生成加入事故后LinDistFlow功率平衡、电压、线路容量、储能、无功、需求响应和上级电网备用约束。",
            "故障隔离与临时转供按分钟级建模，物理抢修默认按3小时情景假设建模；3小时为可配置的城市简单故障演示值，不代表所有故障的保证修复时限。",
            "失负荷惩罚显著高于购电和储能损耗成本。",
            "日末储能SOC不低于初始SOC；非末端窗口使用未来电价作为终端能量价值。",
        ],
    }


def save_json(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
