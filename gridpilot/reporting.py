from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from .engine import save_json


def _polyline(values: list[float], width: int = 760, height: int = 210, color: str = "#1a73e8") -> str:
    if not values:
        return ""
    low, high = min(values), max(values)
    span = max(high - low, 1.0)
    points = []
    for index, value in enumerate(values):
        x = 34 + index * (width - 58) / max(len(values) - 1, 1)
        y = 18 + (high - value) * (height - 50) / span
        points.append(f"{x:.1f},{y:.1f}")
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="趋势图">'
        f'<line x1="34" y1="{height-28}" x2="{width-18}" y2="{height-28}" stroke="#cfd8e3"/>'
        f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3"/>'
        f'<text x="4" y="20" font-size="12" fill="#64748b">{high:.1f}</text>'
        f'<text x="4" y="{height-28}" font-size="12" fill="#64748b">{low:.1f}</text>'
        "</svg>"
    )


def render_html(result: dict) -> str:
    m = result["metrics"]
    n_1 = result.get("n_1")
    status_label = {"completed": "校核通过", "degraded": "降级运行", "needs_attention": "需要关注"}[result["status"]]
    status_class = result["status"]
    grid_chart = _polyline([row["grid_kw"] for row in result["schedule"]], color="#2563eb")
    soc_chart = _polyline([row["soc_pct"] for row in result["schedule"]], color="#0f9d72")
    trace_rows = "".join(
        f"<tr><td>{html.escape(item['step'])}</td><td><span class='pill {html.escape(item['status'])}'>{html.escape(item['status'])}</span></td><td>{html.escape(item['detail'])}</td></tr>"
        for item in result["trace"]
    )
    schedule_rows = "".join(
        f"<tr><td>{row['hour']:02d}:00</td><td>{row['load_kw']:.1f}</td><td>{row['pv_used_kw']:.1f}</td><td>{row['grid_kw']:.1f}</td><td>{row['battery_kw']:.1f}</td><td>{row['soc_pct']:.1f}%</td><td>{row['shed_kw']:.1f}</td></tr>"
        for row in result["schedule"]
    )
    n_1_card = ""
    n_1_section = ""
    investment_section = ""
    plan = result.get("investment_plan")
    if plan:
        asset_rows = "".join(
            f"<tr><td>{html.escape(item['id'])}</td><td>{html.escape(item['name'])}</td>"
            f"<td>{html.escape(item['location'])}</td><td>{html.escape(item['capacity'])}</td>"
            f"<td>{item['capex_wanyuan']:.0f}</td><td>{html.escape(item['purpose'])}</td></tr>"
            for item in plan.get("assets", [])
        )
        investment_section = f"""
<h2>投资建设方案</h2><div class="panel">
<p><strong>{html.escape(plan['name'])}</strong>：总投资 {plan['total_capex_wanyuan']:.0f} 万元，规划期 {plan['planning_horizon_years']} 年，年化投资 {plan['annualized_cost_wanyuan']:.0f} 万元。</p>
<p>安全阈值保持0.95–1.05 pu与线路负载率不超过100%，建设后重新执行基态AC潮流和全部768个N-1工况。</p>
<div class="scroll"><table><thead><tr><th>编号</th><th>资产</th><th>位置</th><th>规模</th><th>投资/万元</th><th>作用</th></tr></thead><tbody>{asset_rows}</tbody></table></div>
</div>"""
    if n_1 is not None:
        n_1_card = f'<div class="card"><div class="label">N-1高风险工况</div><div class="value">{n_1["failed_contingencies"]}/{n_1["contingencies_evaluated"]}</div></div>'
        critical_rows = "".join(
            f"<tr><td>{item['hour']:02d}:00</td><td>{html.escape(item['outage_line'])}</td>"
            f"<td>{html.escape(item['restoration_tie'] or '无可用联络恢复')}</td><td>{item['unserved_load_kw']:.1f}</td>"
            f"<td>{item['min_voltage_pu']:.4f}</td><td>{item['max_line_loading_pct']:.1f}%</td></tr>"
            for item in n_1["critical_contingencies"]
        )
        repair = n_1.get("repair_policy", {})
        n_1_section = f"""
<h2>pandapower 线路 N-1 校核</h2><div class="panel">
<p>N-1校核范围：{len(n_1['hours_checked'])} 个时段 × 32 条运行线路；枚举 {n_1['contingencies_evaluated']} 个故障工况，安全率 {n_1['security_rate_pct']:.1f}%，其中 {n_1['restored_contingencies']} 个通过常开联络线或独立备用馈线恢复后满足要求。</p>
<p>恢复时序假设：故障隔离 {repair.get('fault_isolation_minutes', 1)} 分钟，联络线切换 {repair.get('tie_switching_minutes', 5)} 分钟，备用馈线切换 {repair.get('backup_feeder_switching_minutes', 10)} 分钟；物理抢修按可配置的 {repair.get('assumed_physical_repair_hours', 3)} 小时演示值计算。</p>
<div class="scroll"><table><thead><tr><th>时刻</th><th>退出线路</th><th>最佳恢复</th><th>失供 kW</th><th>最低电压</th><th>最高负载率</th></tr></thead><tbody>{critical_rows}</tbody></table></div>
</div>"""
    payload = html.escape(json.dumps(result, ensure_ascii=False))
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GridPilot-AGH 运行报告</title>
<style>
:root{{--ink:#122033;--muted:#627087;--line:#dbe4ee;--blue:#1769e0;--green:#087f5b;--amber:#b26a00;--red:#b42318}}
*{{box-sizing:border-box}} body{{margin:0;background:#f4f7fb;color:var(--ink);font-family:"Microsoft YaHei","Noto Sans SC",sans-serif}}
.wrap{{max-width:1180px;margin:auto;padding:36px 24px 64px}} header{{background:linear-gradient(135deg,#0b274b,#1459a8);color:white;padding:34px;border-radius:20px;box-shadow:0 16px 40px #173a6530}}
h1{{margin:0 0 8px;font-size:34px}} h2{{margin:32px 0 14px;font-size:22px}} .sub{{opacity:.84}} .status{{display:inline-block;margin-top:18px;padding:7px 13px;border-radius:999px;background:#ffffff20;border:1px solid #ffffff45}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin-top:20px}} .card,.panel{{background:white;border:1px solid var(--line);border-radius:16px;padding:19px;box-shadow:0 8px 24px #24364b0c}} .value{{font-size:27px;font-weight:750;margin-top:7px}} .label{{color:var(--muted);font-size:13px}}
.charts{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} svg{{width:100%;height:auto}} table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{padding:10px 9px;border-bottom:1px solid var(--line);text-align:right}} th:first-child,td:first-child,td:nth-child(3){{text-align:left}} th{{color:var(--muted);font-weight:600;position:sticky;top:0;background:#fff}} .scroll{{max-height:440px;overflow:auto}}
.pill{{padding:3px 8px;border-radius:99px;background:#edf2f7}} .pill.ok{{color:var(--green);background:#e8f7f1}} .pill.warning{{color:var(--amber);background:#fff5df}} .pill.error{{color:var(--red);background:#feeceb}} .completed{{color:#d6fff0}} .degraded{{color:#ffe9b0}} .needs_attention{{color:#ffd1ce}} footer{{color:var(--muted);margin-top:30px;font-size:12px}} @media(max-width:760px){{.charts{{grid-template-columns:1fr}}}}
</style></head><body><main class="wrap">
<header><h1>GridPilot-AGH</h1><div class="sub">含光储配电网滚动优化调度与安全校核智能体</div><div class="status {result['status']}">{status_label} · {html.escape(result['scenario_label'])}</div></header>
<section class="cards">
<div class="card"><div class="label">综合目标值</div><div class="value">¥ {m['objective_yuan']:,.0f}</div></div>
<div class="card"><div class="label">相对基线成本变化</div><div class="value">{m['cost_reduction_pct']:.1f}%</div></div>
<div class="card"><div class="label">新能源消纳率</div><div class="value">{m['renewable_consumption_pct']:.1f}%</div></div>
<div class="card"><div class="label">最低电压</div><div class="value">{m['min_voltage_pu']:.3f} pu</div></div>
<div class="card"><div class="label">失负荷电量</div><div class="value">{m['unserved_energy_kwh']:.1f} kWh</div></div>
<div class="card"><div class="label">异常时段</div><div class="value">{m['violation_hours']}</div></div>
{n_1_card}
</section>
<h2>结论</h2><div class="panel"><strong>{html.escape(result['conclusion'])}</strong><p>{html.escape(result['scenario_description'])}</p></div>
{investment_section}
<h2>运行曲线</h2><section class="charts"><div class="panel"><div class="label">电网购电功率 / kW</div>{grid_chart}</div><div class="panel"><div class="label">储能 SOC / %</div>{soc_chart}</div></section>
<h2>Agent 执行轨迹</h2><div class="panel"><table><thead><tr><th>步骤</th><th>状态</th><th>证据</th></tr></thead><tbody>{trace_rows}</tbody></table></div>
{n_1_section}
<h2>24 小时调度计划</h2><div class="panel scroll"><table><thead><tr><th>时刻</th><th>负荷 kW</th><th>光伏 kW</th><th>购电 kW</th><th>储能 kW</th><th>SOC</th><th>失负荷 kW</th></tr></thead><tbody>{schedule_rows}</tbody></table></div>
<footer>运行编号：{html.escape(result['run_id'])} · 方法：{html.escape(result['method'])}</footer>
<script type="application/json" id="gridpilot-result">{payload}</script></main></body></html>"""


def write_artifacts(result: dict, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    csv_path = output_dir / "schedule.csv"
    html_path = output_dir / "report.html"
    save_json(result, json_path)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = list(result["schedule"][0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["schedule"])
    html_path.write_text(render_html(result), encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "html": html_path}
