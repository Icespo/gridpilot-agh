from __future__ import annotations

import unittest

from gridpilot.data import load_case, validate_case
from gridpilot.engine import run_closed_loop
from gridpilot.webapp import LANDING, load_saved_result
from gridpilot.uncertainty import analyze_uncertainty


class GridPilotTests(unittest.TestCase):
    def test_case_is_valid(self) -> None:
        self.assertEqual(validate_case(load_case()), [])

    def test_normal_scenario_has_complete_schedule(self) -> None:
        result = run_closed_loop("normal")
        self.assertEqual(len(result["schedule"]), 24)
        self.assertTrue(all(10.0 <= row["soc_pct"] <= 90.0 for row in result["schedule"]))
        self.assertEqual(result["metrics"]["violation_hours"], 0)
        self.assertIn("滚动时域", result["method"])
        self.assertEqual(result["optimization"]["rolling_windows"], 24)
        self.assertTrue(all(row["predicted_min_voltage_pu"] >= 0.95 for row in result["schedule"]))
        self.assertTrue(all(row["predicted_max_line_loading_pct"] <= 100.0 + 1e-6 for row in result["schedule"]))
        self.assertGreaterEqual(result["schedule"][-1]["soc_pct"], 60.0 - 1e-6)
        self.assertTrue(all(min(row["battery_charge_kw"], row["battery_discharge_kw"]) < 1e-6 for row in result["schedule"]))
        self.assertEqual(result["power_flow"][0]["backend"], "pandapower Newton-Raphson AC power flow")
        self.assertEqual(result["n_1"]["assessment_mode"], "full_24h")
        self.assertEqual(result["n_1"]["hours_checked"], list(range(24)))
        self.assertEqual(result["n_1"]["expected_contingencies"], 24 * 32)
        self.assertEqual(result["n_1"]["contingencies_evaluated"], 24 * 32)
        self.assertEqual(
            result["n_1"]["backend"],
            "pandapower AC N-1 with corrective tie-line and backup-feeder restoration search",
        )
        self.assertEqual(len(result["n_1"]["topology"]["nodes"]), 33)
        self.assertEqual(len(result["n_1"]["topology"]["backup_feeders"]), 1)
        root_records = [row for row in result["n_1"]["records"] if row["outage_line"] == "L1-2"]
        self.assertEqual(len(root_records), 24)
        self.assertTrue(all(row["restoration_tie"] == "B1-2" for row in root_records))
        self.assertTrue(all(row["restoration_type"] == "backup_feeder" for row in root_records))
        self.assertTrue(all(row["unserved_load_kw"] < 1e-6 for row in root_records))
        self.assertEqual(result["n_1"]["repair_policy"]["assumed_physical_repair_hours"], 3)
        generation = result["security_constraint_generation"]
        self.assertGreater(generation["generated_constraints"], 0)
        self.assertGreater(generation["applied_constraints"], 0)
        self.assertGreaterEqual(generation["iterations"], 1)
        self.assertTrue(any(row["active_security_constraints"] for row in result["schedule"]))
        self.assertTrue(any(row["contingency_actions"] for row in result["schedule"]))
        self.assertGreater(generation["before_risk_index"], 0.0)
        self.assertGreater(generation["after_risk_index"], 0.0)

    def test_pv_error_costs_more_than_normal(self) -> None:
        normal = run_closed_loop("normal", include_n_1=False)
        stressed = run_closed_loop("pv_error", include_n_1=False)
        self.assertGreater(stressed["metrics"]["grid_energy_kwh"], normal["metrics"]["grid_energy_kwh"])

    def test_battery_outage_uses_no_battery(self) -> None:
        result = run_closed_loop("battery_outage", include_n_1=False)
        self.assertTrue(all(abs(row["battery_kw"]) < 1e-9 for row in result["schedule"]))
        self.assertIn(result["status"], {"degraded", "needs_attention"})

    def test_unknown_scenario_rejected(self) -> None:
        with self.assertRaises(ValueError):
            run_closed_loop("not-a-scenario", include_n_1=False)

    def test_web_demo_loads_saved_results_and_contains_charts(self) -> None:
        result = load_saved_result("normal")
        self.assertEqual(len(result["schedule"]), 24)
        self.assertIn("演示进度", LANDING)
        self.assertIn("pandapower AC 电压", LANDING)
        self.assertIn("N-1 安全校核", LANDING)
        self.assertIn("N-1 风险热力图", LANDING)
        self.assertIn("故障拓扑与联络恢复", LANDING)

    def test_web_demo_rejects_unknown_scenario(self) -> None:
        with self.assertRaises(ValueError):
            load_saved_result("not-a-scenario")

    def test_uncertainty_analysis_has_twenty_scenarios_and_risk_metrics(self) -> None:
        analysis = analyze_uncertainty(load_saved_result("normal"))
        self.assertEqual(analysis["scenario_count"], 20)
        self.assertEqual(len(analysis["scenario_paths"]), 20)
        self.assertTrue(all(len(path) == 24 for path in analysis["scenario_paths"]))
        self.assertEqual({item["key"] for item in analysis["methods"]}, {"deterministic", "stochastic", "robust"})
        self.assertTrue(all(item["cvar_yuan"] >= item["expected_cost_yuan"] for item in analysis["methods"]))
        self.assertIn(analysis["selected_method"], {"deterministic", "stochastic", "robust"})
        self.assertIn("预测不确定性输入", LANDING)
        self.assertIn("净负荷概率扇形图", LANDING)


if __name__ == "__main__":
    unittest.main()
