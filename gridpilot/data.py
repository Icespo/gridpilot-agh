from __future__ import annotations

import json
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
}


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
    backup = case.get("backup_feeder")
    if backup:
        if backup.get("from") not in bus_ids or backup.get("to") not in bus_ids:
            errors.append(f"备用馈线端点不存在：{backup}")
        if backup.get("r_ohm", 0) <= 0 or backup.get("max_ka", 0) <= 0:
            errors.append(f"备用馈线参数非法：{backup}")
        if backup.get("switching_time_minutes", 0) <= 0:
            errors.append(f"备用馈线切换时间必须为正：{backup}")
        if backup.get("physical_repair_hours", 0) <= 0:
            errors.append(f"物理抢修时间必须为正：{backup}")
    return errors
