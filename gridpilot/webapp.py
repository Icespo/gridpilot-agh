from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .data import SCENARIOS
from .reporting import render_html
from .uncertainty import analyze_uncertainty, recommended_parameters


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "output"


def load_saved_result(scenario: str) -> dict:
    """Load a precomputed scenario result without running the optimizer again."""
    if scenario not in SCENARIOS:
        raise ValueError(f"未知场景：{scenario}")
    result_path = OUTPUT_ROOT / scenario / "results.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"场景 {scenario} 尚无保存结果，请先运行 run_demo.py 生成结果。")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(result, dict) or not result.get("schedule") or not result.get("metrics"):
        raise ValueError(f"保存结果不完整：{result_path}")
    if scenario == "post_investment":
        baseline_path = OUTPUT_ROOT / "normal" / "results.json"
        if baseline_path.is_file():
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            before_n1 = baseline.get("n_1") or {}
            after_n1 = result.get("n_1") or {}
            before_security = baseline.get("security_constraint_generation") or {}
            after_security = result.get("security_constraint_generation") or {}
            result["investment_comparison"] = {
                "baseline_scenario": "正常运行（现状网架）",
                "before_security_rate_pct": before_n1.get("security_rate_pct", 0.0),
                "after_security_rate_pct": after_n1.get("security_rate_pct", 0.0),
                "before_failed_contingencies": before_n1.get("failed_contingencies", 0),
                "after_failed_contingencies": after_n1.get("failed_contingencies", 0),
                "before_risk_index": before_security.get("after_risk_index", 0.0),
                "after_risk_index": after_security.get("after_risk_index", 0.0),
                "before_min_voltage_pu": baseline.get("metrics", {}).get("min_voltage_pu", 0.0),
                "after_min_voltage_pu": result.get("metrics", {}).get("min_voltage_pu", 0.0),
                "before_operating_cost_yuan": baseline.get("metrics", {}).get("operating_cost_yuan", 0.0),
                "after_operating_cost_yuan": result.get("metrics", {}).get("operating_cost_yuan", 0.0),
            }
    return result


LANDING = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GridPilot-AGH</title>
<style>
:root{--bg:#071426;--panel:#0c1e34;--panel2:#102944;--line:#23425f;--text:#e9f2ff;--muted:#9bb0ca;--blue:#4f9cff;--green:#45d6a6;--amber:#fbbf24;--red:#fb7185}
*{box-sizing:border-box}body{font-family:"Microsoft YaHei","Noto Sans SC",sans-serif;background:var(--bg);color:var(--text);margin:0}.wrap{max-width:1240px;margin:auto;padding:48px 22px 70px}h1{font-size:46px;margin:0 0 10px}h2{font-size:24px;margin:30px 0 14px}h3{margin:0 0 16px}.lead{color:#a8bdd8;font-size:17px;line-height:1.75;max-width:900px}.controls{display:flex;gap:12px;flex-wrap:wrap;margin:26px 0 16px}select,button{padding:12px 16px;border-radius:10px;border:1px solid #315071;font-size:16px}select{background:#0d223b;color:white;min-width:210px}button{background:#2d7ff9;color:white;font-weight:700;cursor:pointer;min-width:130px}button:disabled{opacity:.55;cursor:not-allowed}.panel{background:var(--panel);border:1px solid #1e3c60;border-radius:16px;padding:20px;margin-top:16px;box-shadow:0 12px 34px #02091735}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:12px}.card{background:var(--panel2);border:1px solid #1b3b5e;border-radius:12px;padding:16px}.v{font-size:25px;font-weight:750;margin-top:7px}.muted,.label{color:var(--muted)}.label{font-size:13px}.ok{color:var(--green)}.warn{color:var(--amber)}.bad{color:var(--red)}a{color:#73adff}.progress-panel{display:none;max-width:900px}.progress-head{display:flex;justify-content:space-between;gap:16px;margin-bottom:12px}.progress-track{height:13px;background:#071526;border:1px solid #294b70;border-radius:999px;overflow:hidden}.progress-bar{height:100%;width:0;background:linear-gradient(90deg,#2d7ff9,#4fd1c5);border-radius:999px;transition:width .42s ease}.progress-steps{display:grid;grid-template-columns:repeat(8,1fr);gap:8px;margin-top:14px}.progress-step{font-size:12px;color:#607995;text-align:center}.progress-step.active{color:#e9f2ff}.progress-step.done{color:var(--green)}.demo-note{font-size:12px;color:#7990aa;margin-top:10px}.charts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.chart{min-width:0}.chart-title{font-weight:700;margin-bottom:2px}.chart-sub{font-size:12px;color:var(--muted);margin-bottom:10px}.chart svg{display:block;width:100%;height:auto}.legend{display:flex;gap:14px;flex-wrap:wrap;margin:8px 0 0}.legend-item{font-size:12px;color:#b7c9dd}.legend-dot{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:5px}.summary-grid{display:grid;grid-template-columns:minmax(220px,.7fr) minmax(0,1.3fr);gap:16px}.gauge{display:grid;place-items:center;min-height:210px}.gauge-ring{width:150px;height:150px;border-radius:50%;display:grid;place-items:center;position:relative}.gauge-ring:after{content:"";width:112px;height:112px;border-radius:50%;background:var(--panel);position:absolute}.gauge-value{position:relative;z-index:1;text-align:center;font-size:27px;font-weight:750}.gauge-value small{display:block;font-size:12px;color:var(--muted);font-weight:400;margin-top:3px}.scroll{overflow:auto;max-height:470px}table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:10px 9px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}td:first-child,th:first-child{text-align:left}th{color:#a8bdd8;font-weight:600;position:sticky;top:0;background:var(--panel);z-index:1}.trace td:nth-child(3){text-align:left;white-space:normal;line-height:1.55}.pill{display:inline-block;padding:3px 8px;border-radius:99px;background:#17304e}.pill.ok{color:var(--green)}.pill.warning,.pill.degraded{color:var(--amber)}.result-head{display:flex;align-items:flex-start;justify-content:space-between;gap:20px;margin-top:25px}.result-head p{margin:7px 0;color:#afc2d9;line-height:1.65}.mode-badge{white-space:nowrap;padding:7px 11px;border-radius:99px;background:#14385b;color:#88c1ff;font-size:12px;border:1px solid #27547d}.error{color:#fecaca;background:#401b28;border:1px solid #7f2c42;border-radius:12px;padding:13px 16px}.footer{color:#7188a3;margin-top:24px;font-size:12px}
.uncertainty-input{border-color:#315b86;background:linear-gradient(145deg,#0c2038,#0b1b30)}.uncertainty-head{display:flex;justify-content:space-between;gap:18px;align-items:flex-start}.uncertainty-head h2{margin:0 0 5px}.agent-switch{display:flex;align-items:center;gap:9px;background:#102d4b;border:1px solid #2a5b86;border-radius:999px;padding:8px 12px;white-space:nowrap}.agent-switch input{accent-color:#45d6a6;width:17px;height:17px}.input-grid{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:13px;margin-top:18px}.input-field{background:#09182a;border:1px solid #234a70;border-radius:12px;padding:12px}.input-field label{display:block;color:#a8bdd8;font-size:12px;margin-bottom:7px}.input-field input,.input-field select{width:100%;min-width:0;background:#102944;color:#fff;border:1px solid #315071;border-radius:8px;padding:9px;font-size:14px}.input-field input[type=range]{padding:0;border:0;accent-color:#4f9cff}.input-field input:disabled,.input-field select:disabled{opacity:.72}.field-value{float:right;color:#70c7ff;font-weight:700}.agent-reason{margin-top:14px;padding:11px 13px;border-left:3px solid #45d6a6;background:#0c2b36;color:#bcebdd;border-radius:4px 10px 10px 4px;font-size:13px;line-height:1.55}.uncertainty-actions{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px}.secondary{background:#143252;color:#9fcaff}.uncertainty-preview{margin-top:14px}.fan-grid{display:grid;grid-template-columns:1.25fr .75fr;gap:16px}.fan-chart svg,.scenario-chart svg,.histogram svg{width:100%;height:auto}.risk-bars{display:grid;gap:12px}.risk-bar-row{display:grid;grid-template-columns:120px 1fr 76px;gap:10px;align-items:center;font-size:12px}.risk-bar-track{height:10px;background:#152d47;border-radius:99px;overflow:hidden}.risk-bar-fill{height:100%;border-radius:99px}.correlation{display:grid;grid-template-columns:70px repeat(3,1fr);gap:5px;max-width:520px}.corr-cell{padding:13px 8px;border-radius:8px;text-align:center;font-size:12px}.method-card.selected{border-color:#45d6a6;box-shadow:0 0 0 1px #45d6a6}.method-table td:first-child{font-weight:700}.source-badge{display:inline-block;padding:4px 8px;border-radius:99px;background:#173652;color:#9bc8f5;font-size:11px}
.flow{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));align-items:center;gap:8px}.flow-step{background:#102944;border:1px solid #275078;border-radius:12px;padding:14px;text-align:center;min-height:92px;display:grid;place-items:center}.flow-step strong{display:block;font-size:20px;color:#8bc1ff}.flow-arrow{text-align:center;color:#4f9cff;font-size:24px}.constraint-table td:nth-child(4){text-align:left;white-space:normal}.heatmap-wrap{overflow:auto}.heatmap{border-collapse:separate;border-spacing:4px;min-width:680px}.heatmap th{position:static;background:transparent;text-align:center;border:0}.heatmap td{border:0;padding:8px 10px;border-radius:7px;text-align:center;cursor:pointer;font-weight:700;min-width:72px}.heatmap td:hover{outline:2px solid #fff}.heatmap td:first-child{background:transparent!important;color:#b8c9dc;text-align:left;cursor:default;font-weight:500}.phase-tabs{display:flex;gap:8px;margin-bottom:14px}.phase-tab{padding:7px 12px;min-width:0;font-size:13px;background:#102944}.phase-tab.active{background:#2d7ff9}.topology-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.topology-card{background:#08192c;border:1px solid #1e3c60;border-radius:14px;padding:14px}.topology-card svg{width:100%;height:auto}.topology-title{display:flex;justify-content:space-between;gap:12px;margin-bottom:8px}.topology-meta{color:#9bb0ca;font-size:12px}.topology-legend{display:flex;gap:14px;flex-wrap:wrap;margin-top:8px;font-size:12px;color:#9bb0ca}.legend-line{display:inline-block;width:24px;height:0;border-top:3px solid #54708d;margin-right:5px;vertical-align:middle}.legend-line.outage{border-color:#fb7185;border-top-style:dashed}.legend-line.tie{border-color:#45d6a6}.legend-line.backup{border-color:#22d3ee;border-top-style:double}.constraint-status{font-weight:700}.constraint-status.applied{color:#45d6a6}.constraint-status.deferred_infeasible{color:#fbbf24}
.investment-hero{background:linear-gradient(135deg,#102944,#0b3140);border-color:#2e8b78}.investment-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}.asset-card{background:#091c30;border:1px solid #246c65;border-radius:12px;padding:14px}.asset-card .asset-type{font-size:11px;color:#67e8f9;text-transform:uppercase;letter-spacing:.08em}.asset-card strong{display:block;margin:6px 0}.asset-card p{margin:4px 0;color:#9fb6cc;font-size:12px;line-height:1.5}.compare-table td:nth-child(2),.compare-table td:nth-child(3){font-weight:700}.delta-good{color:#45d6a6}.investment-note{border-left:3px solid #45d6a6;padding:10px 12px;background:#0b2d32;color:#bcebdd;border-radius:4px 10px 10px 4px;margin-top:14px;font-size:13px;line-height:1.6}.legend-line.upgraded{border-color:#fbbf24;border-top-width:5px}.svg-marker{color:#c4b5fd}
@media(max-width:900px){.input-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.fan-grid{grid-template-columns:1fr}}@media(max-width:820px){.charts,.summary-grid,.topology-grid{grid-template-columns:1fr}.flow{grid-template-columns:1fr}.flow-arrow{transform:rotate(90deg)}.progress-steps{grid-template-columns:repeat(3,1fr)}h1{font-size:36px}.result-head,.uncertainty-head{display:block}.mode-badge{display:inline-block;margin-top:8px}.agent-switch{display:inline-flex;margin-top:10px}}@media(max-width:520px){.input-grid{grid-template-columns:1fr}}
</style></head>
<body><main class="wrap">
<h1>GridPilot-AGH</h1>
<p class="lead">含光储配电网滚动优化调度与安全校核智能体。选择场景后，工作台将以演示进度依次呈现滚动时域优化、pandapower AC 潮流与线路 N-1 校核的已保存结果。</p>
<div class="controls"><select id="scenario" onchange="scenarioChanged()"><option value="normal">正常运行</option><option value="pv_error">光伏预测偏差</option><option value="battery_outage">储能退出</option><option value="line_derating">主干线路降额</option><option value="post_investment">投资建设后（电网维护完成）</option></select><button id="run-btn" onclick="runDemo()">开始演示</button></div>
<section class="panel uncertainty-input">
  <div class="uncertainty-head"><div><h2>预测不确定性输入</h2><div class="chart-sub">光伏、负荷和价格预测误差将生成相关时序场景，供确定性、随机和鲁棒策略比较。</div></div><label class="agent-switch"><input id="agent-control" type="checkbox" checked onchange="toggleAgentControl()">Agent 接管参数</label></div>
  <div class="input-grid">
    <div class="input-field"><label>光伏预测误差 <span class="field-value" id="pv-error-value">18%</span></label><input class="uncertainty-param" id="pv-error" type="range" min="0" max="60" value="18" oninput="syncInputLabels()"></div>
    <div class="input-field"><label>负荷预测误差 <span class="field-value" id="load-error-value">7%</span></label><input class="uncertainty-param" id="load-error" type="range" min="0" max="40" value="7" oninput="syncInputLabels()"></div>
    <div class="input-field"><label>价格预测误差 <span class="field-value" id="price-error-value">10%</span></label><input class="uncertainty-param" id="price-error" type="range" min="0" max="60" value="10" oninput="syncInputLabels()"></div>
    <div class="input-field"><label>风险厌恶系数 <span class="field-value" id="risk-value">0.45</span></label><input class="uncertainty-param" id="risk-aversion" type="range" min="0" max="1" step="0.05" value="0.45" oninput="syncInputLabels()"></div>
    <div class="input-field"><label>场景数量</label><input class="uncertainty-param" id="scenario-count" type="number" min="10" max="100" value="20"></div>
    <div class="input-field"><label>CVaR 置信度 / %</label><input class="uncertainty-param" id="confidence" type="number" min="80" max="99" value="95"></div>
    <div class="input-field"><label>误差分布</label><select class="uncertainty-param" id="distribution"><option value="student_t">Student-t 厚尾</option><option value="gaussian">高斯分布</option><option value="bootstrap">历史残差 Bootstrap</option></select></div>
    <div class="input-field"><label>时序相关系数 <span class="field-value" id="correlation-value">0.72</span></label><input class="uncertainty-param" id="temporal-correlation" type="range" min="0" max="0.95" step="0.01" value="0.72" oninput="syncInputLabels()"></div>
    <div class="input-field"><label>数据来源</label><select class="uncertainty-param" id="data-source"><option value="agent_historical_error_template">Agent 历史误差模板</option><option value="project_forecast_residuals">项目预测残差</option><option value="manual">手动参数</option></select></div>
    <div class="input-field"><label>随机种子</label><input class="uncertainty-param" id="uncertainty-seed" type="number" value="20261002"></div>
  </div>
  <div id="agent-reason" class="agent-reason">Agent 建议：常规运行采用中等预测误差和均衡风险偏好。</div>
  <div class="uncertainty-actions"><button class="secondary" onclick="applyAgentParameters()">重新获取 Agent 建议</button><button class="secondary" onclick="previewUncertainty()">预览 20 个场景</button><span class="source-badge">同一参数结构支持人工输入或 Agent API 替换</span></div>
  <div id="uncertainty-preview" class="uncertainty-preview"></div>
</section>
<div id="state" class="muted">请选择场景并开始演示</div>
<section id="progress" class="panel progress-panel" aria-live="polite">
  <div class="progress-head"><strong id="progress-label">准备数据</strong><span id="progress-pct">0%</span></div>
  <div class="progress-track"><div id="progress-bar" class="progress-bar"></div></div>
  <div class="progress-steps" id="progress-steps"></div>
  <div class="demo-note">演示进度：后台直接读取预计算结果，不会重复执行优化与 N-1 枚举。</div>
</section>
<div id="result"></div>
</main>
<script>
const progressStages=[
  {p:8,label:'读取场景数据',short:'场景数据',wait:380},
  {p:24,label:'校验 IEEE 33 节点网络',short:'网络校验',wait:520},
  {p:43,label:'载入滚动时域优化结果',short:'滚动优化',wait:650},
  {p:59,label:'载入 pandapower AC 潮流结果',short:'AC 潮流',wait:520},
  {p:75,label:'载入线路 N-1 校核结果',short:'N-1 校核',wait:560},
  {p:87,label:'载入高风险约束生成结果',short:'约束生成',wait:480},
  {p:95,label:'绘制风险热力图与恢复拓扑',short:'风险可视化',wait:480},
  {p:100,label:'生成可视化工作台',short:'结果展示',wait:400}
];
const colors=['#4f9cff','#45d6a6','#fbbf24','#fb7185','#a78bfa'];
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const f=(x,n=1)=>Number(x??0).toFixed(n);
const esc=value=>String(value??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
const uncertaintyFieldIds=['pv-error','load-error','price-error','risk-aversion','scenario-count','confidence','distribution','temporal-correlation','data-source','uncertainty-seed'];

function syncInputLabels(){
  document.getElementById('pv-error-value').textContent=document.getElementById('pv-error').value+'%';
  document.getElementById('load-error-value').textContent=document.getElementById('load-error').value+'%';
  document.getElementById('price-error-value').textContent=document.getElementById('price-error').value+'%';
  document.getElementById('risk-value').textContent=Number(document.getElementById('risk-aversion').value).toFixed(2);
  document.getElementById('correlation-value').textContent=Number(document.getElementById('temporal-correlation').value).toFixed(2);
}
function readUncertaintyParameters(){return {
  pv_error_pct:+document.getElementById('pv-error').value,
  load_error_pct:+document.getElementById('load-error').value,
  price_error_pct:+document.getElementById('price-error').value,
  risk_aversion:+document.getElementById('risk-aversion').value,
  scenario_count:+document.getElementById('scenario-count').value,
  confidence_pct:+document.getElementById('confidence').value,
  distribution:document.getElementById('distribution').value,
  temporal_correlation:+document.getElementById('temporal-correlation').value,
  data_source:document.getElementById('data-source').value,
  seed:+document.getElementById('uncertainty-seed').value
}}
function fillUncertaintyParameters(p){
  const map={'pv-error':p.pv_error_pct,'load-error':p.load_error_pct,'price-error':p.price_error_pct,'risk-aversion':p.risk_aversion,'scenario-count':p.scenario_count,'confidence':p.confidence_pct,'distribution':p.distribution,'temporal-correlation':p.temporal_correlation,'data-source':p.data_source,'uncertainty-seed':p.seed};
  Object.entries(map).forEach(([id,value])=>{if(value!==undefined&&document.getElementById(id))document.getElementById(id).value=value});syncInputLabels();
  if(p.reason)document.getElementById('agent-reason').textContent='Agent 建议：'+p.reason;
}
function toggleAgentControl(){
  const controlled=document.getElementById('agent-control').checked;
  uncertaintyFieldIds.forEach(id=>document.getElementById(id).disabled=controlled);
  document.getElementById('agent-reason').style.borderLeftColor=controlled?'#45d6a6':'#fbbf24';
  if(!controlled)document.getElementById('agent-reason').textContent='手动模式：当前参数由用户编辑，仍会经过后台范围校验。';
}
async function applyAgentParameters(){
  const scenario=document.getElementById('scenario').value;
  const response=await fetch(`/api/uncertainty/recommend?scenario=${encodeURIComponent(scenario)}`);const data=await response.json();
  if(!response.ok)throw new Error(data.error||'Agent 参数建议加载失败');fillUncertaintyParameters(data);
}
async function scenarioChanged(){if(document.getElementById('agent-control').checked)await applyAgentParameters()}
async function requestUncertainty(){
  const scenario=document.getElementById('scenario').value;
  const response=await fetch('/api/uncertainty',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({scenario,parameters:readUncertaintyParameters()})});
  const data=await response.json();if(!response.ok)throw new Error(data.error||'不确定性分析失败');return data;
}
async function previewUncertainty(){
  const host=document.getElementById('uncertainty-preview');host.innerHTML='<span class="muted">正在生成相关时序场景并计算风险指标…</span>';
  try{const data=await requestUncertainty();const chosen=data.methods.find(x=>x.key===data.selected_method);host.innerHTML=`<div class="cards"><div class="card"><div class="label">Agent 推荐策略</div><div class="v">${esc(data.selected_method_label)}</div></div><div class="card"><div class="label">期望成本</div><div class="v">¥${Number(chosen.expected_cost_yuan).toLocaleString('zh-CN',{maximumFractionDigits:0})}</div></div><div class="card"><div class="label">${f(data.confidence_pct,0)}% CVaR</div><div class="v">¥${Number(chosen.cvar_yuan).toLocaleString('zh-CN',{maximumFractionDigits:0})}</div></div><div class="card"><div class="label">失负荷概率</div><div class="v ${chosen.loss_of_load_probability_pct?'warn':'ok'}">${f(chosen.loss_of_load_probability_pct)}%</div></div></div>`}catch(error){host.innerHTML=`<div class="error">${esc(error.message)}</div>`}
}

function setProgress(index,stage){
  document.getElementById('progress-bar').style.width=stage.p+'%';
  document.getElementById('progress-pct').textContent=stage.p+'%';
  document.getElementById('progress-label').textContent=stage.label;
  [...document.querySelectorAll('.progress-step')].forEach((el,i)=>{el.className='progress-step '+(i<index?'done':i===index?'active':'')});
}
async function simulateProgress(){
  const panel=document.getElementById('progress');
  panel.style.display='block';
  document.getElementById('progress-bar').style.width='0%';
  document.getElementById('progress-pct').textContent='0%';
  document.getElementById('progress-steps').innerHTML=progressStages.map(s=>`<div class="progress-step">${s.short}</div>`).join('');
  for(let i=0;i<progressStages.length;i++){const stage=progressStages[i];await sleep(stage.wait);setProgress(i,stage)}
  [...document.querySelectorAll('.progress-step')].forEach(el=>el.classList.add('done'));
}

function lineChart(series,{min=null,max=null,thresholds=[]}={}){
  const width=760,height=260,left=58,right=18,top=20,bottom=38;
  const values=series.flatMap(s=>s.values).filter(Number.isFinite);
  if(!values.length)return '<div class="muted">暂无曲线数据</div>';
  let lo=min===null?Math.min(...values):min,hi=max===null?Math.max(...values):max;
  thresholds.forEach(t=>{lo=Math.min(lo,t.value);hi=Math.max(hi,t.value)});
  const pad=Math.max((hi-lo)*.08,.5);if(min===null)lo-=pad;if(max===null)hi+=pad;if(hi===lo)hi=lo+1;
  const x=i=>left+i*(width-left-right)/Math.max(series[0].values.length-1,1);
  const y=v=>top+(hi-v)*(height-top-bottom)/(hi-lo);
  let svg=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="结果趋势图">`;
  for(let i=0;i<=4;i++){const yy=top+i*(height-top-bottom)/4,val=hi-i*(hi-lo)/4;svg+=`<line x1="${left}" y1="${yy}" x2="${width-right}" y2="${yy}" stroke="#1f3a57"/><text x="${left-8}" y="${yy+4}" text-anchor="end" fill="#8299b4" font-size="11">${f(val,1)}</text>`}
  [0,6,12,18,23].forEach(i=>{if(i<series[0].values.length)svg+=`<text x="${x(i)}" y="${height-12}" text-anchor="middle" fill="#8299b4" font-size="11">${String(i).padStart(2,'0')}:00</text>`});
  thresholds.forEach(t=>{svg+=`<line x1="${left}" y1="${y(t.value)}" x2="${width-right}" y2="${y(t.value)}" stroke="${t.color||'#fb7185'}" stroke-dasharray="7 5"/><text x="${width-right-2}" y="${y(t.value)-5}" text-anchor="end" fill="${t.color||'#fb7185'}" font-size="10">${esc(t.label)}</text>`});
  series.forEach((s,si)=>{const pts=s.values.map((v,i)=>`${x(i).toFixed(1)},${y(Number(v)).toFixed(1)}`).join(' ');svg+=`<polyline points="${pts}" fill="none" stroke="${s.color||colors[si]}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>`});
  svg+='</svg><div class="legend">'+series.map((s,i)=>`<span class="legend-item"><i class="legend-dot" style="background:${s.color||colors[i]}"></i>${esc(s.name)}</span>`).join('')+'</div>';
  return svg;
}

function chartPanel(title,sub,chart){return `<section class="panel chart"><div class="chart-title">${title}</div><div class="chart-sub">${sub}</div>${chart}</section>`}

function fanChart(fan){
  const width=780,height=285,left=58,right=18,top=20,bottom=38,all=[...fan.p_low,...fan.p_high];
  const lo=Math.min(...all),hi=Math.max(...all),pad=Math.max((hi-lo)*.08,20),min=lo-pad,max=hi+pad;
  const x=i=>left+i*(width-left-right)/Math.max(fan.base.length-1,1),y=v=>top+(max-v)*(height-top-bottom)/(max-min);
  const points=values=>values.map((v,i)=>`${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  const band=(low,high,color,opacity)=>`<polygon points="${points(high)} ${[...low].reverse().map((v,j)=>`${x(low.length-1-j).toFixed(1)},${y(v).toFixed(1)}`).join(' ')}" fill="${color}" opacity="${opacity}"/>`;
  let svg=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="净负荷预测区间扇形图">`;
  for(let i=0;i<=4;i++){const yy=top+i*(height-top-bottom)/4,val=max-i*(max-min)/4;svg+=`<line x1="${left}" y1="${yy}" x2="${width-right}" y2="${yy}" stroke="#1f3a57"/><text x="${left-7}" y="${yy+4}" text-anchor="end" fill="#8299b4" font-size="11">${f(val,0)}</text>`}
  svg+=band(fan.p_low,fan.p_high,'#4f9cff',.16)+band(fan.p25,fan.p75,'#4f9cff',.30);
  svg+=`<polyline points="${points(fan.base)}" fill="none" stroke="#fbbf24" stroke-width="2.5" stroke-dasharray="7 5"/><polyline points="${points(fan.p50)}" fill="none" stroke="#6dd6ff" stroke-width="3"/>`;
  [0,6,12,18,23].forEach(i=>svg+=`<text x="${x(i)}" y="${height-12}" text-anchor="middle" fill="#8299b4" font-size="11">${String(i).padStart(2,'0')}:00</text>`);
  return svg+'</svg><div class="legend"><span class="legend-item"><i class="legend-dot" style="background:#4f9cff"></i>置信区间</span><span class="legend-item"><i class="legend-dot" style="background:#6dd6ff"></i>场景中位数</span><span class="legend-item"><i class="legend-dot" style="background:#fbbf24"></i>基准预测</span></div>';
}

function scenarioPathsChart(paths,base){
  const width=780,height=285,left=58,right=18,top=20,bottom=38,values=paths.flat(),lo=Math.min(...values),hi=Math.max(...values),pad=Math.max((hi-lo)*.05,20),min=lo-pad,max=hi+pad;
  const x=i=>left+i*(width-left-right)/Math.max(base.length-1,1),y=v=>top+(max-v)*(height-top-bottom)/(max-min),points=vals=>vals.map((v,i)=>`${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  let svg=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="20个净负荷场景轨迹">`;
  for(let i=0;i<=4;i++){const yy=top+i*(height-top-bottom)/4;svg+=`<line x1="${left}" y1="${yy}" x2="${width-right}" y2="${yy}" stroke="#1f3a57"/>`}
  paths.forEach((path,i)=>svg+=`<polyline points="${points(path)}" fill="none" stroke="${colors[i%colors.length]}" stroke-width="1.4" opacity=".30"/>`);
  svg+=`<polyline points="${points(base)}" fill="none" stroke="#fff" stroke-width="3"/>`;
  [0,6,12,18,23].forEach(i=>svg+=`<text x="${x(i)}" y="${height-12}" text-anchor="middle" fill="#8299b4" font-size="11">${String(i).padStart(2,'0')}:00</text>`);
  return svg+'</svg><div class="legend"><span class="legend-item"><i class="legend-dot" style="background:#fff"></i>基准预测</span><span class="legend-item">彩色细线：各不确定性场景</span></div>';
}

function histogramChart(hist,color){
  const width=360,height=190,left=35,right=12,top=12,bottom=32,max=Math.max(...hist.counts,1),bar=(width-left-right)/hist.counts.length;
  let svg=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="预测误差分布直方图"><line x1="${left}" y1="${height-bottom}" x2="${width-right}" y2="${height-bottom}" stroke="#36516e"/>`;
  hist.counts.forEach((count,i)=>{const h=count/max*(height-top-bottom),x=left+i*bar+2;svg+=`<rect x="${x}" y="${height-bottom-h}" width="${Math.max(bar-4,2)}" height="${h}" rx="2" fill="${color}" opacity=".82"/>`});
  svg+=`<text x="${left}" y="${height-9}" fill="#8299b4" font-size="10">${f(hist.centers[0],0)}%</text><text x="${width-right}" y="${height-9}" text-anchor="end" fill="#8299b4" font-size="10">+${f(hist.centers.at(-1),0)}%</text></svg>`;return svg;
}

function correlationGrid(matrix){
  const labels=['光伏','负荷','价格'];let html='<div class="correlation"><div></div>'+labels.map(x=>`<div class="corr-cell label">${x}</div>`).join('');
  matrix.forEach((row,i)=>{html+=`<div class="corr-cell label">${labels[i]}</div>`+row.map(v=>{const alpha=.15+Math.abs(v)*.65,color=v>=0?`rgba(69,214,166,${alpha})`:`rgba(251,113,133,${alpha})`;return `<div class="corr-cell" style="background:${color}">${f(v,2)}</div>`}).join('')});return html+'</div>';
}

function comparisonBars(methods){
  const metrics=[['expected_cost_yuan','期望成本','#4f9cff','¥'],['cvar_yuan','95% CVaR','#a78bfa','¥'],['loss_of_load_probability_pct','失负荷概率','#fb7185','%'],['voltage_violation_probability_pct','电压越限概率','#fbbf24','%']];
  return metrics.map(([key,label,color,unit])=>{const max=Math.max(...methods.map(x=>Number(x[key])),1);return `<div style="margin-bottom:18px"><div class="chart-sub">${label}</div><div class="risk-bars">${methods.map(x=>`<div class="risk-bar-row"><span>${esc(x.label)}</span><div class="risk-bar-track"><div class="risk-bar-fill" style="width:${Math.max(Number(x[key])/max*100,1)}%;background:${color}"></div></div><strong>${unit==='¥'?'¥'+Number(x[key]).toLocaleString('zh-CN',{maximumFractionDigits:0}):f(x[key])+unit}</strong></div>`).join('')}</div></div>`}).join('');
}

function uncertaintySection(u){
  if(!u)return '';
  const p=u.parameters,e=u.optimization_evidence||{},selected=u.methods.find(x=>x.key===u.selected_method)||u.methods[0];
  const methodCards=u.methods.map(x=>`<div class="card method-card ${x.key===u.selected_method?'selected':''}"><div class="label">${esc(x.label)}${x.key===u.selected_method?' · 推荐':''}</div><div class="v">¥${Number(x.expected_cost_yuan).toLocaleString('zh-CN',{maximumFractionDigits:0})}</div><div class="chart-sub">CVaR ¥${Number(x.cvar_yuan).toLocaleString('zh-CN',{maximumFractionDigits:0})} · 备用峰值 ${f(x.peak_reserve_kw,0)} kW</div><div class="chart-sub">一阶段 ¥${Number(x.first_stage_cost_yuan).toLocaleString('zh-CN',{maximumFractionDigits:0})} · 期望追索 ¥${Number(x.expected_recourse_cost_yuan).toLocaleString('zh-CN',{maximumFractionDigits:0})}</div></div>`).join('');
  const methodRows=u.methods.map(x=>`<tr><td>${esc(x.label)}</td><td>¥${Number(x.expected_cost_yuan).toLocaleString('zh-CN',{maximumFractionDigits:0})}</td><td>¥${Number(x.cvar_yuan).toLocaleString('zh-CN',{maximumFractionDigits:0})}</td><td>¥${Number(x.first_stage_cost_yuan).toLocaleString('zh-CN',{maximumFractionDigits:0})}</td><td>¥${Number(x.expected_recourse_cost_yuan).toLocaleString('zh-CN',{maximumFractionDigits:0})}</td><td>${f(x.loss_of_load_probability_pct)}%</td><td>${f(x.voltage_violation_probability_pct)}%</td><td>${f(x.expected_unserved_energy_kwh,2)}</td><td>${f(x.peak_reserve_kw,0)}</td></tr>`).join('');
  return `<h2>预测不确定性与风险指标</h2><section class="panel"><div class="result-head" style="margin:0"><div><h3>${u.scenario_count} 场景 · ${f(u.confidence_pct,0)}% CVaR · ${esc(p.distribution)}</h3><div class="chart-sub">光伏 ±${f(p.pv_error_pct,0)}%　负荷 ±${f(p.load_error_pct,0)}%　价格 ±${f(p.price_error_pct,0)}%　时序相关 ${f(p.temporal_correlation,2)}</div></div><span class="mode-badge">真正两阶段场景优化 · HiGHS</span></div></section>
  <section class="panel"><div class="chart-title">两阶段确定性等价模型证据</div><div class="chart-sub">一阶段日前决策在全部场景间共享；二阶段购电平衡、储能SOC、需求响应、弃光和失负荷按场景独立，显式满足非预见性。</div><div class="cards"><div class="card"><div class="label">共享一阶段变量</div><div class="v">${e.first_stage_variable_count??'—'}</div><div class="chart-sub">日前购电、储能基准计划、上下备用</div></div><div class="card"><div class="label">二阶段追索变量</div><div class="v">${e.scenario_count??u.scenario_count} × ${e.second_stage_variable_count_per_scenario??'—'}</div><div class="chart-sub">每个场景拥有独立追索块</div></div><div class="card"><div class="label">隐式非预见性链接</div><div class="v">${e.implicit_nonanticipativity_links??'—'}</div><div class="chart-sub">由共享变量块严格保证</div></div><div class="card"><div class="label">稀疏LP规模</div><div class="v">${e.total_variable_count??'—'} 变量</div><div class="chart-sub">${e.equality_count??'—'} 等式 · ${e.inequality_count??'—'} 不等式</div></div></div><div class="chart-sub" style="margin-top:12px">目标：期望成本与CVaR加权；网络：逐场景AC结果校准的电压/热稳定安全包络；求解器：${esc(e.solver||'SciPy HiGHS')}。</div></section>
  <section class="cards"><div class="card"><div class="label">推荐策略</div><div class="v">${esc(u.selected_method_label)}</div></div><div class="card"><div class="label">期望成本</div><div class="v">¥${Number(selected.expected_cost_yuan).toLocaleString('zh-CN',{maximumFractionDigits:0})}</div></div><div class="card"><div class="label">${f(u.confidence_pct,0)}% CVaR</div><div class="v">¥${Number(selected.cvar_yuan).toLocaleString('zh-CN',{maximumFractionDigits:0})}</div></div><div class="card"><div class="label">失负荷概率</div><div class="v ${selected.loss_of_load_probability_pct?'warn':'ok'}">${f(selected.loss_of_load_probability_pct)}%</div></div><div class="card"><div class="label">电压越限概率</div><div class="v ${selected.voltage_violation_probability_pct?'warn':'ok'}">${f(selected.voltage_violation_probability_pct)}%</div></div></section>
  <div class="fan-grid"><section class="panel fan-chart"><div class="chart-title">净负荷概率扇形图</div><div class="chart-sub">外带为 ${f(u.confidence_pct,0)}% 区间，内带为 25%–75% 分位区间 / kW</div>${fanChart(u.fan_chart)}</section><section class="panel scenario-chart"><div class="chart-title">${u.scenario_count} 条场景轨迹</div><div class="chart-sub">保留时间相关性与光伏—负荷—价格相关性</div>${scenarioPathsChart(u.scenario_paths,u.fan_chart.base)}</section></div>
  <h2 style="font-size:20px">预测误差分布与相关结构</h2><div class="charts"><section class="panel histogram"><div class="chart-title">光伏误差</div>${histogramChart(u.error_histograms.pv,'#45d6a6')}</section><section class="panel histogram"><div class="chart-title">负荷误差</div>${histogramChart(u.error_histograms.load,'#4f9cff')}</section><section class="panel histogram"><div class="chart-title">价格误差</div>${histogramChart(u.error_histograms.price,'#a78bfa')}</section><section class="panel"><div class="chart-title">经验相关系数矩阵</div><div class="chart-sub">绿色为正相关，红色为负相关</div>${correlationGrid(u.correlation)}</section></div>
  <h2 style="font-size:20px">确定性 / 两阶段随机 / 最坏场景鲁棒优化</h2><section class="cards">${methodCards}</section><div class="summary-grid"><section class="panel">${comparisonBars(u.methods)}</section><section class="panel scroll"><table class="method-table"><thead><tr><th>方法</th><th>期望成本</th><th>CVaR</th><th>一阶段成本</th><th>期望追索成本</th><th>失负荷概率</th><th>电压越限概率</th><th>期望失供 kWh</th><th>备用峰值 kW</th></tr></thead><tbody>${methodRows}</tbody></table></section></div>`;
}

let currentResult=null;
let activeRiskRecords=[];

function constraintSection(d){
  const g=d.security_constraint_generation;
  if(!g)return '';
  const cuts=g.constraints||[],planning=g.planning_actions||[];
  const cutRows=cuts.map(c=>`<tr><td>${esc(c.id)}</td><td>${String(c.hour).padStart(2,'0')}:00</td><td>${esc(c.outage_line)} → ${esc(c.restoration_tie)}</td><td>功率平衡 + 电压≥${f(c.target_min_voltage_pu,3)} pu + 线路容量 + 事故后资源响应</td><td>${f(c.risk_score,0)}</td><td><span class="constraint-status ${esc(c.status)}">${c.status==='applied'?'已嵌入':c.status==='deferred_infeasible'?'延期/不可行':'已生成'}</span></td></tr>`).join('');
  const planningRows=planning.slice(0,5).map(x=>`<tr><td>${String(x.hour).padStart(2,'0')}:00</td><td>${esc(x.outage_line)}</td><td>${f(x.unserved_load_kw)} kW</td><td>${esc(x.recommended_action)}</td></tr>`).join('');
  const iterationRows=(g.iteration_history||[]).map(x=>`<tr><td>${x.iteration}</td><td>${x.generated}</td><td>${x.newly_applied}</td><td>${x.before_failed} → ${x.after_failed}</td><td>${f(x.before_risk_index,1)} → ${f(x.after_risk_index,1)}</td><td>${esc(x.status)}</td></tr>`).join('');
  const responseRows=(d.schedule||[]).flatMap(row=>(row.contingency_actions||[]).map(x=>`<tr><td>${String(row.hour).padStart(2,'0')}:00</td><td>${esc(x.constraint_id)}</td><td>${esc(x.outage_line)} → ${esc(x.restoration)}</td><td>${f(x.battery_up_kw-x.battery_down_kw,1)}</td><td>${f(x.battery_q_delta_kvar,1)}</td><td>${f(x.grid_up_kw,1)}</td><td>${f(x.demand_response_kw,1)}</td></tr>`)).join('');
  return `<h2>N-1 高风险约束生成</h2>
  <section class="cards">
    <div class="card"><div class="label">筛查高风险</div><div class="v">${g.screened_failed_contingencies}</div></div>
    <div class="card"><div class="label">生成运行约束</div><div class="v">${g.generated_constraints}</div></div>
    <div class="card"><div class="label">已嵌入滚动优化</div><div class="v ok">${g.applied_constraints}</div></div>
    <div class="card"><div class="label">规划补强问题</div><div class="v warn">${g.planning_actions_count}</div></div>
  </section>
  <section class="panel"><div class="flow">
    <div class="flow-step"><div><strong>${g.before_failed_contingencies}</strong>初始高风险工况</div></div><div class="flow-arrow">→</div>
    <div class="flow-step"><div><strong>${g.generated_constraints}</strong>生成安全约束</div></div><div class="flow-arrow">→</div>
    <div class="flow-step"><div><strong>${g.applied_constraints}</strong>嵌入滚动优化</div></div><div class="flow-arrow">→</div>
    <div class="flow-step"><div><strong>${f(g.before_security_rate_pct)}% → ${f(g.after_security_rate_pct)}%</strong>pandapower 复核<br><span class="label">风险指数 ${f(g.before_risk_index,0)} → ${f(g.after_risk_index,0)}</span></div></div>
  </div></section>
  ${iterationRows?`<section class="panel scroll"><h3>迭代 SCOPF 过程</h3><table><thead><tr><th>轮次</th><th>生成约束</th><th>新增应用</th><th>高风险工况</th><th>风险指数</th><th>状态</th></tr></thead><tbody>${iterationRows}</tbody></table></section>`:''}
  <section class="panel scroll"><h3>运行类安全约束</h3><table class="constraint-table"><thead><tr><th>编号</th><th>时刻</th><th>故障 / 恢复</th><th>嵌入模型的约束</th><th>风险分</th><th>状态</th></tr></thead><tbody>${cutRows||'<tr><td colspan="6">没有可转化为运行约束的工况</td></tr>'}</tbody></table></section>
  ${responseRows?`<section class="panel scroll"><h3>事故后资源纠正动作</h3><table><thead><tr><th>时刻</th><th>约束</th><th>故障 / 恢复</th><th>储能有功 kW</th><th>储能无功 kvar</th><th>上级电网增援 kW</th><th>需求响应 kW</th></tr></thead><tbody>${responseRows}</tbody></table></section>`:''}
  ${planningRows?`<section class="panel scroll"><h3>需网架或备用补强的结构性风险</h3><table><thead><tr><th>时刻</th><th>退出线路</th><th>失供</th><th>建议</th></tr></thead><tbody>${planningRows}</tbody></table></section>`:''}`;
}

function investmentSection(d){
  const plan=d.investment_plan,cmp=d.investment_comparison;
  if(!plan)return '';
  const assets=(plan.assets||[]).map(a=>`<div class="asset-card"><div class="asset-type">${esc(a.id)} · ${esc(a.type)}</div><strong>${esc(a.name)}</strong><p>位置：${esc(a.location)}<br>规模：${esc(a.capacity)}<br>作用：${esc(a.purpose)}</p><div class="label">投资 ${f(a.capex_wanyuan,0)} 万元</div></div>`).join('');
  const compare=cmp?`<section class="panel scroll"><h3>建设前后物理校核对比</h3><table class="compare-table"><thead><tr><th>指标</th><th>现状网架</th><th>投资建设后</th><th>变化</th></tr></thead><tbody>
    <tr><td>N-1 安全率</td><td>${f(cmp.before_security_rate_pct)}%</td><td>${f(cmp.after_security_rate_pct)}%</td><td class="delta-good">+${f(cmp.after_security_rate_pct-cmp.before_security_rate_pct)} pct</td></tr>
    <tr><td>高风险工况 / 768</td><td>${cmp.before_failed_contingencies}</td><td>${cmp.after_failed_contingencies}</td><td class="delta-good">-${cmp.before_failed_contingencies-cmp.after_failed_contingencies}</td></tr>
    <tr><td>综合风险指数</td><td>${f(cmp.before_risk_index,0)}</td><td>${f(cmp.after_risk_index,0)}</td><td class="delta-good">${cmp.before_risk_index?'-'+f((cmp.before_risk_index-cmp.after_risk_index)/cmp.before_risk_index*100):'0'}%</td></tr>
    <tr><td>基态最低电压</td><td>${f(cmp.before_min_voltage_pu,3)} pu</td><td>${f(cmp.after_min_voltage_pu,3)} pu</td><td class="delta-good">+${f(cmp.after_min_voltage_pu-cmp.before_min_voltage_pu,3)} pu</td></tr>
    <tr><td>24h 运行成本</td><td>¥${Number(cmp.before_operating_cost_yuan).toLocaleString('zh-CN',{maximumFractionDigits:0})}</td><td>¥${Number(cmp.after_operating_cost_yuan).toLocaleString('zh-CN',{maximumFractionDigits:0})}</td><td>${cmp.after_operating_cost_yuan<=cmp.before_operating_cost_yuan?'下降':'上升'} ¥${Math.abs(cmp.after_operating_cost_yuan-cmp.before_operating_cost_yuan).toLocaleString('zh-CN',{maximumFractionDigits:0})}</td></tr>
  </tbody></table></section>`:'';
  return `<h2>投资建设方案与投运效果</h2><section class="panel investment-hero"><div class="result-head" style="margin:0"><div><h3>${esc(plan.name)}</h3><div class="chart-sub">规划期 ${plan.planning_horizon_years} 年；总投资 ${f(plan.total_capex_wanyuan,0)} 万元；年化投资 ${f(plan.annualized_cost_wanyuan,0)} 万元。</div></div><span class="mode-badge">规划资产已投运 · AC/N-1 重算</span></div><div class="investment-note">安全判据仍为 0.95–1.05 pu 与线路负载率≤100%，没有降低阈值。所有改善来自拓扑、阻抗、载流量、无功支撑和主变电压设定的物理变化。</div></section><section class="investment-grid">${assets}</section>${compare}`;
}

function riskColor(record){
  if(!record||record.secure)return '#164b45';
  const score=Math.max(0,Math.min(100,Number(record.risk_score||0)));
  const hue=48*(1-score/100);return `hsl(${hue} 67% ${score>70?34:29}%)`;
}

function renderRiskHeatmap(records){
  const host=document.getElementById('risk-heatmap');if(!host||!currentResult?.n_1)return;
  activeRiskRecords=records;
  const topology=currentResult.n_1.topology||{};
  const hours=[...new Set(records.map(x=>Number(x.hour)))].sort((a,b)=>a-b);
  const lineNames=(topology.lines||[]).map(x=>x.name);
  const lookup=new Map(records.map((x,i)=>[`${x.outage_line}|${x.hour}`,{record:x,index:i}]));
  const rows=lineNames.map(name=>`<tr><td>${esc(name)}</td>${hours.map(hour=>{const item=lookup.get(`${name}|${hour}`);if(!item)return '<td style="background:#10243b">—</td>';const r=item.record;return `<td style="background:${riskColor(r)}" onclick="selectRisk(${item.index})" title="${esc(r.risk_drivers?.join('；')||'安全')}">${r.secure?'✓':f(r.risk_score,0)}</td>`}).join('')}</tr>`).join('');
  host.innerHTML=`<table class="heatmap"><thead><tr><th>故障线路</th>${hours.map(h=>`<th>${String(h).padStart(2,'0')}:00</th>`).join('')}</tr></thead><tbody>${rows}</tbody></table>`;
  const worst=records.reduce((best,r,i)=>!best||Number(r.risk_score)>best.score?{score:Number(r.risk_score),index:i}:best,null);
  if(worst)selectRisk(worst.index);
}

function setHeatmapPhase(phase){
  const g=currentResult?.security_constraint_generation;if(!g)return;
  document.querySelectorAll('.phase-tab').forEach(el=>el.classList.toggle('active',el.dataset.phase===phase));
  renderRiskHeatmap(phase==='before'?(g.before_records||[]):(g.after_records||currentResult.n_1.records||[]));
}

function topologySvg(record,restored){
  const topo=currentResult.n_1.topology,nodeMap=new Map(topo.nodes.map(n=>[Number(n.bus),n]));
  const unsupplied=new Set((restored?record.unsupplied_buses:record.pre_restoration_unsupplied_buses||[]).map(Number));
  const edgeLine=edge=>{const a=nodeMap.get(Number(edge.from)),b=nodeMap.get(Number(edge.to)),isOut=edge.name===record.outage_line,upgraded=!!edge.investment_upgraded;const stroke=isOut?'#fb7185':upgraded?'#fbbf24':'#52708f';const dash=isOut?'8 6':'';const width=isOut?5:upgraded?4:2;return `<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" stroke="${stroke}" stroke-width="${width}" stroke-dasharray="${dash}" opacity="${isOut?0.95:0.78}"/>${isOut?`<text x="${(a.x+b.x)/2}" y="${(a.y+b.y)/2-8}" text-anchor="middle" fill="#fb7185" font-size="13">${esc(edge.name)} 退出</text>`:''}`};
  const tieLine=edge=>{const a=nodeMap.get(Number(edge.from)),b=nodeMap.get(Number(edge.to)),active=restored&&edge.name===record.restoration_tie;const cx=(a.x+b.x)/2,cy=Math.min(a.y,b.y)-35;return `<path d="M ${a.x} ${a.y} Q ${cx} ${cy} ${b.x} ${b.y}" fill="none" stroke="${active?'#45d6a6':'#36526f'}" stroke-width="${active?5:1.5}" stroke-dasharray="${active?'':'7 6'}" opacity="${active?1:.45}"/>${active?`<text x="${cx}" y="${cy-6}" text-anchor="middle" fill="#45d6a6" font-size="13">${esc(edge.name)} 合闸</text>`:''}`};
  const backupLine=(edge,index)=>{const a=nodeMap.get(Number(edge.from)),b=nodeMap.get(Number(edge.to)),active=restored&&edge.name===record.restoration_tie,lane=topo.height-32-index*25,label=active?`${edge.name} 已投入`:`${edge.name} 常开${edge.investment_new?' · 新建':''}`;return `<path d="M ${a.x} ${a.y} C ${a.x+20} ${lane}, ${b.x-20} ${lane}, ${b.x} ${b.y}" fill="none" stroke="#22d3ee" stroke-width="${active?6:edge.investment_new?3.5:2.5}" stroke-dasharray="${active?'':'10 6'}" opacity="${active?1:.78}"/><circle cx="${(a.x+b.x)/2}" cy="${lane}" r="${active?7:5}" fill="${active?'#22d3ee':'#0b2036'}" stroke="#22d3ee" stroke-width="2"/><g aria-label="${esc(label)}"><rect x="${530+index*5}" y="${14+index*27}" width="205" height="23" rx="7" fill="#071a2d" stroke="#22d3ee" opacity=".96"/><line x1="${540+index*5}" y1="${25+index*27}" x2="${560+index*5}" y2="${25+index*27}" stroke="#22d3ee" stroke-width="${active?5:3}" stroke-dasharray="${active?'':'7 5'}"/><text x="${568+index*5}" y="${29+index*27}" fill="#67e8f9" font-size="12" font-weight="700">${esc(label)}</text></g>`};
  const nodes=topo.nodes.map(n=>{const offline=unsupplied.has(Number(n.bus));const fill=offline?'#fb7185':n.type==='source'?'#4f9cff':n.type==='battery'?'#a78bfa':n.type==='pv'?'#45d6a6':'#d6e3f2';const svgRing=n.reactive_support?`<circle cx="${n.x}" cy="${n.y}" r="13" fill="none" stroke="#c4b5fd" stroke-width="3"/>`:'';return `${svgRing}<circle cx="${n.x}" cy="${n.y}" r="${offline?10:8}" fill="${fill}" stroke="#071426" stroke-width="2"/><text x="${n.x}" y="${n.y-12}" text-anchor="middle" fill="${offline?'#ffb3bf':'#afc2d9'}" font-size="11">${n.bus}</text>`}).join('');
  return `<svg viewBox="0 0 ${topo.width} ${topo.height}" role="img" aria-label="IEEE 33节点故障恢复拓扑">${topo.lines.map(edgeLine).join('')}${topo.ties.map(tieLine).join('')}${(topo.backup_feeders||[]).map(backupLine).join('')}${nodes}</svg>`;
}

function selectRisk(index){
  const record=activeRiskRecords[index],host=document.getElementById('topology-view');if(!record||!host)return;
  const beforeOff=(record.pre_restoration_unsupplied_buses||[]).length,afterOff=(record.unsupplied_buses||[]).length;
  document.getElementById('selected-risk-title').textContent=`${String(record.hour).padStart(2,'0')}:00 · ${record.outage_line} · 风险分 ${f(record.risk_score,0)}`;
  const recoveryLabel=record.restoration_type==='backup_feeder'?'备用馈线':'联络线';
  document.getElementById('selected-risk-detail').textContent=(record.risk_drivers||[]).join('；')+`；最佳恢复：${record.restoration_tie||'无'}（${recoveryLabel}）`;
  const backupActive=record.restoration_type==='backup_feeder';
  host.innerHTML=`<div class="cards"><div class="card"><div class="label">故障隔离</div><div class="v">1 min</div></div><div class="card"><div class="label">供电恢复</div><div class="v">${record.switching_time_minutes??'—'} min</div></div><div class="card"><div class="label">恢复支路</div><div class="v ${backupActive?'ok':''}">${record.restoration_tie||'无可用支路'}</div></div><div class="card"><div class="label">物理抢修假设</div><div class="v">${record.physical_repair_hours??3} h</div></div><div class="card"><div class="label">预计修复完成</div><div class="v">${String(record.repair_completion_hour??'—').padStart(2,'0')}:00</div></div></div><div class="topology-grid"><section class="topology-card"><div class="topology-title"><strong>故障后 · 恢复前</strong><span class="topology-meta">失电节点 ${beforeOff} 个</span></div>${topologySvg(record,false)}</section><section class="topology-card"><div class="topology-title"><strong>临时供电恢复后</strong><span class="topology-meta">剩余失电节点 ${afterOff} 个</span></div>${topologySvg(record,true)}</section></div><div class="topology-legend"><span><i class="legend-line"></i>运行线路</span><span><i class="legend-line outage"></i>故障退出</span><span><i class="legend-line tie"></i>投入联络线</span><span><i class="legend-line backup"></i>独立备用馈线（虚线常开 / 实线投入）</span>${topoInvestmentLegend()}<span>红色节点：失电</span><span>绿色节点：光伏</span><span>紫色节点：储能</span></div>`;
}

function topoInvestmentLegend(){const topo=currentResult?.n_1?.topology||{};return topo.investment_mode?'<span><i class="legend-line upgraded"></i>投资增容线路</span><span class="svg-marker">◎ SVG 无功节点</span>':''}

function riskVisualization(d){
  if(!d.n_1?.topology||!d.security_constraint_generation)return '';
  const policy=d.n_1.repair_policy||{};
  return `<h2>故障隔离、临时恢复与物理抢修</h2><section class="panel"><div class="cards"><div class="card"><div class="label">故障隔离</div><div class="v">${policy.fault_isolation_minutes??1} min</div></div><div class="card"><div class="label">联络线切换</div><div class="v">${policy.tie_switching_minutes??5} min</div></div><div class="card"><div class="label">备用馈线切换</div><div class="v">${policy.backup_feeder_switching_minutes??10} min</div></div><div class="card"><div class="label">物理抢修假设</div><div class="v">${policy.assumed_physical_repair_hours??3} h</div></div></div><div class="chart-sub">${esc(policy.interpretation||'逐时假想故障；开关恢复与物理抢修分开建模。')}</div></section><h2>N-1 风险热力图</h2><section class="panel"><div class="phase-tabs"><button class="phase-tab" data-phase="before" onclick="setHeatmapPhase('before')">约束生成前</button><button class="phase-tab active" data-phase="after" onclick="setHeatmapPhase('after')">约束生成后</button></div><div class="chart-sub">全24小时视图：行表示32条故障线路，列表示24个时段，共768个N-1工况；绿色为安全，数字越大风险越高。横向滚动并点击任意单元格可查看恢复拓扑。</div><div id="risk-heatmap" class="heatmap-wrap"></div></section><h2>故障拓扑与联络恢复 / 独立备用馈线恢复</h2><section class="panel"><h3 id="selected-risk-title">选择热力图中的工况</h3><div id="selected-risk-detail" class="chart-sub"></div><div id="topology-view"></div></section>`;
}

function renderResult(d,u){
  const m=d.metrics,s=d.schedule||[],pf=d.power_flow||[],n=d.n_1;
  const byHour=new Map(pf.map(x=>[Number(x.hour),x]));
  const minV=s.map(x=>Number(byHour.get(Number(x.hour))?.min_voltage_pu??x.predicted_min_voltage_pu));
  const maxV=s.map(x=>Number(byHour.get(Number(x.hour))?.max_voltage_pu??x.predicted_min_voltage_pu));
  const loading=s.map(x=>Number(byHour.get(Number(x.hour))?.max_line_loading_pct??x.predicted_max_line_loading_pct));
  const statusClass=d.status==='completed'?'ok':'warn';
  const n1Card=n?`<div class="card"><div class="label">N-1 安全率</div><div class="v ${n.security_rate_pct<80?'warn':'ok'}">${f(n.security_rate_pct)}%</div></div>`:'';
  const cards=`<section class="cards">
    <div class="card"><div class="label">综合目标值</div><div class="v">¥${Number(m.objective_yuan).toLocaleString('zh-CN',{maximumFractionDigits:0})}</div></div>
    <div class="card"><div class="label">成本改善</div><div class="v">${f(m.cost_reduction_pct)}%</div></div>
    <div class="card"><div class="label">新能源消纳率</div><div class="v">${f(m.renewable_consumption_pct)}%</div></div>
    <div class="card"><div class="label">AC 最低电压</div><div class="v">${f(m.min_voltage_pu,3)} pu</div></div>
    <div class="card"><div class="label">最大线路负载率</div><div class="v">${f(m.max_line_loading_pct)}%</div></div>
    <div class="card"><div class="label">基态异常时段</div><div class="v">${m.violation_hours}</div></div>${n1Card}</section>`;
  const charts=`<h2>结果曲线</h2><div class="charts">
    ${chartPanel('系统功率平衡','24 小时负荷、购电与光伏出力 / kW',lineChart([{name:'负荷',values:s.map(x=>+x.load_kw),color:'#fbbf24'},{name:'电网购电',values:s.map(x=>+x.grid_kw),color:'#4f9cff'},{name:'光伏利用',values:s.map(x=>+x.pv_used_kw),color:'#45d6a6'}],{min:0}))}
    ${chartPanel('储能运行状态','正值表示放电，负值表示充电 / kW',lineChart([{name:'储能功率',values:s.map(x=>+x.battery_kw),color:'#a78bfa'}]))}
    ${chartPanel('储能荷电状态','SOC 运行轨迹 / %',lineChart([{name:'SOC',values:s.map(x=>+x.soc_pct),color:'#45d6a6'}],{min:0,max:100,thresholds:[{value:10,label:'SOC 下限',color:'#fb7185'},{value:90,label:'SOC 上限',color:'#fbbf24'}]}))}
    ${chartPanel('pandapower AC 电压','全网最高/最低节点电压 / pu',lineChart([{name:'最低电压',values:minV,color:'#fb7185'},{name:'最高电压',values:maxV,color:'#4f9cff'}],{min:.93,max:1.07,thresholds:[{value:.95,label:'电压下限',color:'#fbbf24'},{value:1.05,label:'电压上限',color:'#fbbf24'}]}))}
    ${chartPanel('线路热稳定','各时段最高线路负载率 / %',lineChart([{name:'最高负载率',values:loading,color:'#4f9cff'}],{min:0,max:Math.max(110,...loading),thresholds:[{value:100,label:'热稳定限值',color:'#fb7185'}]}))}
  </div>`;
  const uncertainty=uncertaintySection(u);
  const investment=investmentSection(d);
  const constraints=constraintSection(d);
  const riskVisual=riskVisualization(d);
  let n1='';
  if(n){
    const rate=Math.max(0,Math.min(100,Number(n.security_rate_pct)));const critical=(n.critical_contingencies||[]).slice(0,12);
    n1=`<h2>N-1 安全校核</h2><div class="summary-grid">
      <section class="panel gauge"><div class="gauge-ring" style="background:conic-gradient(${rate>=80?'#45d6a6':'#fbbf24'} ${rate}%,#17324f 0)"><div class="gauge-value">${f(rate)}%<small>安全率</small></div></div><div class="muted">${n.secure_contingencies} 安全 / ${n.failed_contingencies} 高风险 / ${n.contingencies_evaluated} 总工况</div></section>
      <section class="panel scroll"><h3>关键高风险工况</h3><table><thead><tr><th>时刻</th><th>退出线路</th><th>联络恢复</th><th>失供 kW</th><th>最低电压</th><th>最高负载率</th></tr></thead><tbody>${critical.map(x=>`<tr><td>${String(x.hour).padStart(2,'0')}:00</td><td>${esc(x.outage_line)}</td><td>${esc(x.restoration_tie||'无')}</td><td>${f(x.unserved_load_kw)}</td><td>${f(x.min_voltage_pu,3)}</td><td>${f(x.max_line_loading_pct)}%</td></tr>`).join('')}</tbody></table></section>
    </div>`;
  }
  const schedule=`<h2>24 小时调度计划</h2><section class="panel scroll"><table><thead><tr><th>时刻</th><th>负荷 kW</th><th>光伏 kW</th><th>购电 kW</th><th>储能 kW</th><th>SOC</th><th>最低电压</th><th>最高负载率</th></tr></thead><tbody>${s.map((x,i)=>`<tr><td>${String(x.hour).padStart(2,'0')}:00</td><td>${f(x.load_kw)}</td><td>${f(x.pv_used_kw)}</td><td>${f(x.grid_kw)}</td><td>${f(x.battery_kw)}</td><td>${f(x.soc_pct)}%</td><td>${f(minV[i],3)}</td><td>${f(loading[i])}%</td></tr>`).join('')}</tbody></table></section>`;
  const trace=`<h2>执行轨迹</h2><section class="panel scroll"><table class="trace"><thead><tr><th>步骤</th><th>状态</th><th>说明</th></tr></thead><tbody>${(d.trace||[]).map(x=>`<tr><td>${esc(x.step)}</td><td><span class="pill ${esc(x.status)}">${esc(x.status)}</span></td><td>${esc(x.detail)}</td></tr>`).join('')}</tbody></table></section>`;
  document.getElementById('result').innerHTML=`<div class="result-head"><div><h2 style="margin:0">${esc(d.scenario_label)} · <span class="${statusClass}">${esc(d.status)}</span></h2><p>${esc(d.conclusion)}</p></div><span class="mode-badge">预计算结果 · 演示模式</span></div>${cards}${investment}${uncertainty}${constraints}${riskVisual}${charts}${n1}${schedule}${trace}<div class="footer">运行编号：${esc(d.run_id)} · <a href="/report?scenario=${encodeURIComponent(d.scenario)}" target="_blank">打开独立完整报告</a></div>`;
  currentResult=d;
  if(riskVisual)setHeatmapPhase('after');
}

async function runDemo(){
  const btn=document.getElementById('run-btn'),scenario=document.getElementById('scenario').value,state=document.getElementById('state');
  btn.disabled=true;document.getElementById('result').innerHTML='';state.textContent='正在载入已保存的分析结果…';
  try{
    const request=fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({scenario})});
    const [response,uncertainty]=await Promise.all([request,requestUncertainty(),simulateProgress()]);
    const data=await response.json();if(!response.ok)throw new Error(data.error||'结果加载失败');
    renderResult(data,uncertainty);state.innerHTML=`<span class="ok">演示完成</span> · 已读取 ${esc(data.scenario_label)} 的后台保存结果并生成 ${uncertainty.scenario_count} 个不确定性场景`;
  }catch(error){state.innerHTML=`<div class="error">${esc(error.message)}</div>`}
  finally{btn.disabled=false}
}
syncInputLabels();toggleAgentControl();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    latest: dict | None = None

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self._send(200, LANDING.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/scenarios":
            self._send(200, json.dumps(SCENARIOS, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
        elif path == "/api/uncertainty/recommend":
            try:
                scenario = parse_qs(parsed.query).get("scenario", ["normal"])[0]
                if scenario not in SCENARIOS:
                    raise ValueError(f"未知场景：{scenario}")
                body = json.dumps(recommended_parameters(scenario), ensure_ascii=False).encode("utf-8")
                self._send(200, body, "application/json; charset=utf-8")
            except ValueError as exc:
                self._send(400, json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
        elif path == "/report":
            try:
                scenario = parse_qs(parsed.query).get("scenario", [""])[0]
                result = load_saved_result(scenario) if scenario else Handler.latest
                if result is None:
                    result = load_saved_result("normal")
                self._send(200, render_html(result).encode("utf-8"), "text/html; charset=utf-8")
            except (ValueError, FileNotFoundError) as exc:
                self._send(404, str(exc).encode("utf-8"), "text/plain; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path not in {"/api/run", "/api/uncertainty"}:
            self._send(404, b"not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            result = load_saved_result(payload.get("scenario", "normal"))
            if path == "/api/uncertainty":
                body = json.dumps(
                    analyze_uncertainty(result, payload.get("parameters")),
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send(200, body, "application/json; charset=utf-8")
                return
            Handler.latest = result
            body = json.dumps(result, ensure_ascii=False).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
            body = json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8")
            self._send(400, body, "application/json; charset=utf-8")

    def log_message(self, fmt: str, *args: object) -> None:
        print("[web] " + fmt % args)


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    print(f"GridPilot-AGH 工作台：http://{host}:{port}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
