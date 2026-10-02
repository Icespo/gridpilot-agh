from __future__ import annotations

import argparse
from pathlib import Path

from gridpilot.data import SCENARIOS
from gridpilot.engine import run_closed_loop
from gridpilot.reporting import write_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 GridPilot-AGH 调度与安全校核闭环")
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="normal")
    parser.add_argument("--output", type=Path, default=Path("output/latest"))
    args = parser.parse_args()
    result = run_closed_loop(args.scenario)
    paths = write_artifacts(result, args.output)
    print(f"场景：{result['scenario_label']}")
    print(f"状态：{result['status']}")
    print(f"结论：{result['conclusion']}")
    print(f"最低电压：{result['metrics']['min_voltage_pu']:.4f} pu")
    print(f"异常时段：{result['metrics']['violation_hours']}")
    for name, path in paths.items():
        print(f"{name}: {path.resolve()}")


if __name__ == "__main__":
    main()

