from __future__ import annotations

import json
import copy
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASE_PATH = ROOT / "data" / "ieee33.json"

LOAD_PROFILE = np.array([
    0.38, 0.36, 0.35, 0.35, 0.37, 0.41, 0.47, 0.52,
    0.55, 0.56, 0.55, 0.54, 0.53, 0.54, 0.56, 0.59,
    0.63, 0.68, 0.72, 0.70, 0.66, 0.58, 0.49, 0.42,
], dtype=float)

PV_PROFILE = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.10, 0.25,
    0.45, 0.65, 0.82, 0.94, 1.00, 0.94, 0.80, 0.62,
    0.40, 0.18, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0,
], dtype=float)

PRICE = np.array([
    0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.52, 0.68,
    0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68,
    0.92, 1.18, 1.18, 1.18, 1.18, 0.92, 0.52, 0.36,
], dtype=float)

SCENARIOS = {
    "normal": {
        "label": "正常运行",
        "description": "基准光伏预测、储能可用、线路额定容量正常。",
        "pv_multiplier": 1.0,
        "battery_available": True,
        "line_rating_multiplier": 1.0,
        "load_multiplier": 1.0,
    },
    "pv_error": {
        "label": "光伏预测偏差",
        "description": "实际光伏仅为预测值的 65%，触发滚动重调度。",
        "pv_multiplier": 0.65,
        "battery_available": True,
        "line_rating_multiplier": 1.0,
        "load_multiplier": 1.0,
    },
    "battery_outage": {
        "label": "储能退出",
        "description": "储能全天不可用，用于验证不可行诊断与降级运行。",
        "pv_multiplier": 0.65,
        "battery_available": False,
        "line_rating_multiplier": 1.0,
        "load_multiplier": 1.08,
    },
    "line_derating": {
        "label": "主干线路降额",
        "description": "首段线路容量降至额定值的 25%，验证热稳定异常。",
        "pv_multiplier": 0.8,
        "battery_available": True,
        "line_rating_multiplier": 0.25,
        "load_multiplier": 1.02,
    },
    "post_investment": {
        "label": "投资建设后",
        "description": "完成远端双备用馈线、关键走廊增容、动态无功补偿和主变有载调压建设后的运行状态。",
        "pv_multiplier": 1.0,
        "battery_available": True,
        "line_rating_multiplier": 1.0,
        "load_multiplier": 1.0,
        "investment_plan": {
            "name": "配电网韧性补强一期工程",
            "planning_horizon_years": 10,
            "total_capex_wanyuan": 1680.0,
            "annualized_cost_wanyuan": 198.0,
            "assets": [
                {"id": "INV-01", "type": "backup_feeder", "name": "B1-6", "location": "1—6", "capacity": "9.87 MVA", "capex_wanyuan": 520.0, "purpose": "主干中枢独立转供"},
                {"id": "INV-02", "type": "backup_feeder", "name": "B1-33", "location": "1—33", "capacity": "7.89 MVA", "capex_wanyuan": 480.0, "purpose": "南部馈线独立转供"},
                {"id": "INV-03", "type": "line_upgrade", "name": "关键走廊与联络线增容降阻", "location": "18 段关键支路 + 5 条联络线", "capacity": "载流量 +50% / 支路阻抗 -45%", "capex_wanyuan": 410.0, "purpose": "降低故障转供压降"},
                {"id": "INV-04", "type": "reactive_support", "name": "SVG 动态无功群", "location": "节点 18 / 25 / 33", "capacity": "3 × 600 kvar", "capex_wanyuan": 210.0, "purpose": "末端电压支撑"},
                {"id": "INV-05", "type": "oltc", "name": "主变 OLTC 改造", "location": "节点 1", "capacity": "1.03 pu / 9 档", "capex_wanyuan": 60.0, "purpose": "全网电压中枢调节"},
            ],
        },
    },
}


def apply_scenario_investments(case: dict, scenario: str) -> dict:
    """Return a case copy with explicit planning assets commissioned.

    Operational scenarios keep the original network untouched.  The investment
    scenario changes physical parameters and restoration candidates, so its
    improved N-1 result is produced by AC power flow rather than by relaxing any
    voltage or loading threshold.
    """
    upgraded = copy.deepcopy(case)
    if scenario != "post_investment":
        return upgraded

    upgraded["source_voltage_pu"] = 1.03
    upgraded["investment_plan"] = copy.deepcopy(SCENARIOS[scenario]["investment_plan"])
    upgraded["reactive_support"] = [
        {"name": "SVG-18", "bus": 18, "capacity_kvar": 600.0},
        {"name": "SVG-25", "bus": 25, "capacity_kvar": 600.0},
        {"name": "SVG-33", "bus": 33, "capacity_kvar": 600.0},
    ]
    critical_corridors = {
        (2, 3), (3, 4), (4, 5), (5, 6),
        (3, 23), (23, 24),
        (24, 25), (2, 19), (19, 20), (20, 21), (21, 22),
        (6, 7), (6, 26), (27, 28), (28, 29), (29, 30), (30, 31), (31, 32),
    }
    for line in upgraded["lines"]:
        if (int(line["from"]), int(line["to"])) in critical_corridors:
            line["r_ohm"] = round(float(line["r_ohm"]) * 0.55, 6)
            line["x_ohm"] = round(float(line["x_ohm"]) * 0.55, 6)
            line["max_ka"] = round(float(line["max_ka"]) * 1.50, 6)
            line["investment_upgraded"] = True
    for tie in upgraded.get("tie_lines", []):
        tie["r_ohm"] = round(float(tie["r_ohm"]) * 0.35, 6)
        tie["x_ohm"] = round(float(tie["x_ohm"]) * 0.35, 6)
        tie["max_ka"] = round(float(tie["max_ka"]) * 1.50, 6)
        tie["investment_upgraded"] = True

    existing = copy.deepcopy(upgraded.get("backup_feeder"))
    upgraded["backup_feeders"] = [
        existing,
        {
            "name": "B1-6", "from": 1, "to": 6,
            "r_ohm": 0.04, "x_ohm": 0.02, "max_ka": 0.45,
            "normally_open": True, "switching_time_minutes": 8,
            "physical_repair_hours": 3,
            "description": "投资新建：主站至节点6的独立备用馈线",
            "investment_new": True,
        },
        {
            "name": "B1-33", "from": 1, "to": 33,
            "r_ohm": 0.04, "x_ohm": 0.02, "max_ka": 0.45,
            "normally_open": True, "switching_time_minutes": 8,
            "physical_repair_hours": 3,
            "description": "投资新建：主站至节点33的独立备用馈线",
            "investment_new": True,
        },
    ]
    upgraded["contingency_response"].update({
        "demand_response_fraction": 0.12,
        "grid_response_kw": 1200.0,
    })
    return upgraded


def load_case() -> dict:
    with CASE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def scenario_config(name: str) -> dict:
    if name not in SCENARIOS:
        raise ValueError(f"未知场景 {name!r}，可选值：{', '.join(SCENARIOS)}")
    return dict(SCENARIOS[name])


def validate_case(case: dict) -> list[str]:
    errors: list[str] = []
    buses = case.get("buses", [])
    lines = case.get("lines", [])
    bus_ids = {row.get("bus") for row in buses}
    if len(buses) != 33:
        errors.append(f"节点数应为 33，实际为 {len(buses)}")
    if len(lines) != 32:
        errors.append(f"径向支路数应为 32，实际为 {len(lines)}")
    for row in lines:
        if row.get("from") not in bus_ids or row.get("to") not in bus_ids:
            errors.append(f"支路端点不存在：{row}")
        if row.get("r_ohm", 0) <= 0 or row.get("max_ka", 0) <= 0:
            errors.append(f"支路参数非法：{row}")
    for row in buses:
        if row.get("p_kw", 0) < 0 or row.get("q_kvar", 0) < 0:
            errors.append(f"负荷不能为负：{row}")
    backups = case.get("backup_feeders") or ([case["backup_feeder"]] if case.get("backup_feeder") else [])
    for backup in backups:
        if backup.get("from") not in bus_ids or backup.get("to") not in bus_ids:
            errors.append(f"备用馈线端点不存在：{backup}")
        if backup.get("r_ohm", 0) <= 0 or backup.get("max_ka", 0) <= 0:
            errors.append(f"备用馈线参数非法：{backup}")
        if backup.get("switching_time_minutes", 0) <= 0:
            errors.append(f"备用馈线切换时间必须为正：{backup}")
        if backup.get("physical_repair_hours", 0) <= 0:
            errors.append(f"物理抢修时间必须为正：{backup}")
    return errors
